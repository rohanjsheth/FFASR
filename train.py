import argparse
import tomllib
from pathlib import Path
from typing import Any, Sequence

import torch
from datasets import Audio, Dataset, load_dataset
from transformers import (
    AutoProcessor,
    Qwen3ASRForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from data_utils.SceneDataset import SceneDataset
from data_utils.data_utils import audio_duration_seconds
from data_utils.data_collator import Qwen3ASRDataCollator
from data_utils.room_folds import FOLDS
from training_utils.epoch_callback import SceneEpochCallback

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "train.toml"
LOAD_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--fold", type=int, choices=tuple(FOLDS), default=1)
    return parser.parse_args(argv)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as config_file:
        return tomllib.load(config_file)


def load_speech_splits(
    dataset_config: dict[str, Any],
    validation_config: dict[str, Any],
    cache_dir: str,
) -> tuple[Dataset, Dataset]:
    """Load the training split and a fixed shuffled slice for validation."""
    text_column = dataset_config["text_column"]
    min_seconds = dataset_config["min_duration_seconds"]

    def prepare(split: str) -> Dataset:
        speech_ds = load_dataset(
            dataset_config["speech_id"],
            dataset_config["speech_config"],
            split=split,
            cache_dir=cache_dir,
        ).cast_column("audio", Audio(decode=False))
        if text_column != "text":
            speech_ds = speech_ds.rename_column(text_column, "text")
        # Header read only, no decode; datasets caches the result.
        return speech_ds.filter(
            lambda audio: audio_duration_seconds(audio) >= min_seconds,
            input_columns="audio",
            num_proc=8,
        )

    train_ds = prepare(dataset_config["speech_split"])
    validation_ds = prepare(dataset_config["validation_speech_split"])
    validation_ds = validation_ds.shuffle(seed=validation_config["seed"]).select(
        range(validation_config["num_examples"])
    )

    return train_ds, validation_ds


def load_noise(dataset_config: dict[str, Any], cache_dir: str) -> Dataset:
    return load_dataset(
        dataset_config["noise_id"],
        split=dataset_config["noise_split"],
        cache_dir=cache_dir,
    ).cast_column("audio", Audio(decode=False))


def load_rir_folds(
    dataset_config: dict[str, Any],
    fold: int,
    cache_dir: str,
) -> tuple[Dataset, Dataset]:
    """Split the room impulse responses into this fold's training and held-out rooms."""
    rir_ds = load_dataset(
        "parquet",
        data_files=dataset_config["rir_parquet"],
        split="train",
        cache_dir=cache_dir,
    ).cast_column("audio", Audio(decode=False))

    train_rooms, validation_rooms = FOLDS[fold]
    # input_columns keeps the filter from reading every RIR waveform off disk.
    return (
        rir_ds.filter(lambda room: room in train_rooms, input_columns="Room"),
        rir_ds.filter(lambda room: room in validation_rooms, input_columns="Room"),
    )


def resolve_load_dtype(load_dtype_name: str) -> torch.dtype:
    """Weight storage dtype, which is not the bf16/fp16 autocast compute dtype."""
    if load_dtype_name not in LOAD_DTYPES:
        raise ValueError(
            f"Unknown load_dtype {load_dtype_name!r}, expected one of "
            f"{sorted(LOAD_DTYPES)}"
        )

    model_dtype = LOAD_DTYPES[load_dtype_name]
    if model_dtype is not torch.float32:
        print(
            f"Warning: weights stored as {load_dtype_name}, so the optimizer updates "
            "low-precision parameters with no float32 master copy."
        )
    return model_dtype


def build_model(
    model_id: str,
    cache_dir: str,
    model_dtype: torch.dtype,
) -> Qwen3ASRForConditionalGeneration:
    """Load the model with only the audio tower and projector left trainable."""
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        dtype=model_dtype,
        low_cpu_mem_usage=True,
    )

    model.requires_grad_(False)
    model.model.audio_tower.requires_grad_(True)
    model.model.multi_modal_projector.requires_grad_(True)
    model.config.use_cache = False
    return model


def train(config: dict[str, Any], fold: int) -> None:
    model_config = config["model"]
    dataset_config = config["datasets"]
    path_config = config["paths"]
    scene_config = config["scene"]
    validation_config = config["validation"]
    training_config = config["training"]

    cache_dir = str(Path(path_config["cache_dir"]))
    output_dir = Path(path_config["output_root"]) / f"fold-{fold}"
    sample_rate = scene_config["sample_rate"]
    model_dtype = resolve_load_dtype(model_config["load_dtype"])

    print("Loading datasets...")
    train_speech_ds, validation_speech_ds = load_speech_splits(
        dataset_config, validation_config, cache_dir
    )
    noise_ds = load_noise(dataset_config, cache_dir)
    train_rir_ds, validation_rir_ds = load_rir_folds(dataset_config, fold, cache_dir)
    print(
        f"Dataset rows: training speech={len(train_speech_ds)}, "
        f"validation speech={len(validation_speech_ds)}, noise={len(noise_ds)}, "
        f"training RIRs={len(train_rir_ds)}, validation RIRs={len(validation_rir_ds)}"
    )

    processor = AutoProcessor.from_pretrained(model_config["model_id"], cache_dir=cache_dir)
    processor_sample_rate = int(processor.feature_extractor.sampling_rate)
    if processor_sample_rate != sample_rate:
        raise ValueError(
            f"Scene rate {sample_rate} does not match processor rate "
            f"{processor_sample_rate}"
        )

    model = build_model(model_config["model_id"], cache_dir, model_dtype)

    train_dataset = SceneDataset(
        speech_ds=train_speech_ds,
        noise_ds=noise_ds,
        rir_ds=train_rir_ds,
        base_seed=scene_config["base_seed"],
        sample_rate=sample_rate,
        number_of_noises=scene_config["number_of_noises"],
        clean_probability=scene_config["clean_probability"],
    )
    validation_dataset = SceneDataset(
        speech_ds=validation_speech_ds,
        noise_ds=noise_ds,
        rir_ds=validation_rir_ds,
        base_seed=validation_config["seed"],
        sample_rate=sample_rate,
        number_of_noises=scene_config["number_of_noises"],
        clean_probability=validation_config["clean_probability"],
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=str(output_dir), **training_config),
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=Qwen3ASRDataCollator(
            processor=processor,
            sample_rate=sample_rate,
            language=scene_config["language"],
        ),
        processing_class=processor,
        callbacks=[SceneEpochCallback(dataset=train_dataset)],
    )

    # Picks up an interrupted run on a rented box; None on a clean start.
    last_checkpoint = get_last_checkpoint(str(output_dir)) if output_dir.is_dir() else None
    if last_checkpoint is not None:
        print(f"Resuming from {last_checkpoint}")

    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model()


def main(argv: Sequence[str] | None = None) -> int:
    run_args = parse_args(argv)
    train(load_config(run_args.config), run_args.fold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
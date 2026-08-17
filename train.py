import argparse
import tomllib
from pathlib import Path
from typing import Any, Sequence

import torch
from datasets import Audio, load_dataset
from transformers import (
    AutoProcessor,
    Qwen3ASRForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from data_utils.SceneDataset import SceneDataset
from data_utils.data_collator import Qwen3ASRDataCollator
from data_utils.room_folds import FOLDS
from training_utils.wer_callback import ValidationWERCallback

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "train.toml"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--fold", type=int, choices=tuple(FOLDS), default=1)
    return parser.parse_args(argv)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as config_file:
        return tomllib.load(config_file)


run_args = parse_args()
config = load_config(run_args.config)
model_config = config["model"]
dataset_config = config["datasets"]
path_config = config["paths"]
scene_config = config["scene"]
validation_config = config["validation"]
training_config = config["training"]

model_id = model_config["model_id"]
cache_dir = str(Path(path_config["cache_dir"]))
output_dir = Path(path_config["output_root"]) / f"fold-{run_args.fold}"

if training_config["bf16"] and training_config["fp16"]:
    raise ValueError("bf16 and fp16 cannot both be enabled")
if training_config["bf16"]:
    model_dtype = torch.bfloat16
elif training_config["fp16"]:
    model_dtype = torch.float16
else:
    model_dtype = torch.float32

print("Loading training datasets...")
train_speech_ds = load_dataset(
    dataset_config["speech_id"],
    dataset_config["speech_config"],
    split=dataset_config["speech_split"],
    cache_dir=cache_dir,
).cast_column("audio", Audio(decode=False))
validation_speech_ds = load_dataset(
    dataset_config["speech_id"],
    dataset_config["speech_config"],
    split=dataset_config["validation_speech_split"],
    cache_dir=cache_dir,
).cast_column("audio", Audio(decode=False))

num_validation_examples = validation_config["num_examples"]
if num_validation_examples > len(validation_speech_ds):
    raise ValueError(
        f"Requested {num_validation_examples} validation examples from "
        f"{len(validation_speech_ds)} available records"
    )
validation_speech_ds = validation_speech_ds.shuffle(
    seed=validation_config["seed"]
).select(range(num_validation_examples))

noise_ds = load_dataset(
    dataset_config["noise_id"],
    split=dataset_config["noise_split"],
    cache_dir=cache_dir,
).cast_column("audio", Audio(decode=False))
rir_ds = load_dataset(
    "parquet",
    data_files=dataset_config["rir_parquet"],
    split="train",
    cache_dir=cache_dir,
).cast_column("audio", Audio(decode=False))

train_rooms, validation_rooms = FOLDS[run_args.fold]
train_rir_ds = rir_ds.filter(lambda row: row["Room"] in train_rooms)
validation_rir_ds = rir_ds.filter(lambda row: row["Room"] in validation_rooms)
print(
    f"Dataset rows: training speech={len(train_speech_ds)}, "
    f"validation speech={len(validation_speech_ds)}, noise={len(noise_ds)}, "
    f"training RIRs={len(train_rir_ds)}, validation RIRs={len(validation_rir_ds)}"
)

processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)

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

collator = Qwen3ASRDataCollator(
    processor=processor,
    sample_rate=scene_config["sample_rate"],
    language=scene_config["language"],
)

training_args = TrainingArguments(
    output_dir=str(output_dir),
    **training_config,
)

train_dataset = SceneDataset(
    speech_ds=train_speech_ds,
    noise_ds=noise_ds,
    rir_ds=train_rir_ds,
    base_seed=scene_config["base_seed"],
    sample_rate=scene_config["sample_rate"],
    number_of_noises=scene_config["number_of_noises"],
    clean_probability=scene_config["clean_probability"],
)

validation_dataset = SceneDataset(
    speech_ds=validation_speech_ds,
    noise_ds=noise_ds,
    rir_ds=validation_rir_ds,
    base_seed=validation_config["seed"],
    sample_rate=scene_config["sample_rate"],
    number_of_noises=scene_config["number_of_noises"],
    clean_probability=scene_config["clean_probability"],
)

if validation_config["wer_examples"] > len(validation_dataset):
    raise ValueError("wer_examples cannot exceed the validation dataset size")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=collator,
    processing_class=processor,
    callbacks=[
        ValidationWERCallback(
            dataset=validation_dataset,
            processor=processor,
            language=scene_config["language"],
            sample_rate=scene_config["sample_rate"],
            num_examples=validation_config["wer_examples"],
            batch_size=validation_config["wer_batch_size"],
            max_new_tokens=validation_config["max_new_tokens"],
        )
    ],
)

trainer.train()
trainer.save_model()

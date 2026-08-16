"""Smoke-test scene rendering and Qwen3-ASR batch collation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

    from data_utils import RenderedScene


DEFAULT_MODEL_ID = "Qwen/Qwen3-ASR-1.7B-hf"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render one clean and one simulated scene, then collate them with "
            "the real Qwen3-ASR processor."
        )
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--language", default="English")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--number-of-noises", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--noise-records", type=int, default=8)
    parser.add_argument(
        "--max-rir-records",
        type=int,
        default=256,
        help="Maximum streamed RIR rows to inspect while finding one complete group.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf_cache"))
    return parser.parse_args(argv)


def materialize_records(dataset: IterableDataset, count: int, name: str) -> Dataset:
    """Materialize a small prefix of a streaming dataset as a map-style dataset."""
    from datasets import Dataset

    records = list(dataset.take(count))
    if len(records) != count:
        raise RuntimeError(f"Requested {count} {name} records but received {len(records)}")
    return Dataset.from_list(records)


def find_complete_rir_group(
    rir_stream: IterableDataset,
    required_rirs: int,
    max_records: int,
) -> tuple[Dataset, str, str, int]:
    """Find and materialize one room/receiver group with enough RIR rows."""
    from datasets import Dataset

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for scanned, record in enumerate(rir_stream, start=1):
        room = str(record["Room"])
        receiver = str(record["Receiver Label"])
        group = groups.setdefault((room, receiver), [])
        group.append(record)

        if len(group) >= required_rirs:
            return Dataset.from_list(group), room, receiver, scanned

        if scanned >= max_records:
            break

    raise RuntimeError(
        f"No room/receiver group had {required_rirs} RIRs within the first "
        f"{max_records} rows; increase --max-rir-records"
    )


def validate_scene(
    name: str,
    scene: RenderedScene,
    expected_text: str,
    sample_rate: int,
) -> None:
    """Check the raw item contract before processor collation."""
    import numpy as np

    audio = scene["audio"]
    if not isinstance(audio, np.ndarray):
        raise TypeError(f"{name} audio is {type(audio).__name__}, not a NumPy array")
    if audio.ndim != 1:
        raise ValueError(f"{name} audio must be mono; received shape {audio.shape}")
    if audio.size == 0:
        raise ValueError(f"{name} audio is empty")
    if not np.issubdtype(audio.dtype, np.floating):
        raise TypeError(f"{name} audio must be floating point; received {audio.dtype}")
    if not np.all(np.isfinite(audio)):
        raise ValueError(f"{name} audio contains NaN or infinite values")
    if scene["text"] != expected_text:
        raise ValueError(f"{name} transcript no longer matches its speech record")
    if not isinstance(scene["metadata"], dict):
        raise TypeError(f"{name} metadata is not a dictionary")

    duration = audio.size / sample_rate
    peak = float(np.max(np.abs(audio)))
    print(
        f"  {name}: shape={audio.shape}, dtype={audio.dtype}, "
        f"duration={duration:.2f}s, peak={peak:.4f}"
    )


def validate_batch(batch: Any, batch_size: int) -> None:
    """Check the keys and primary tensor relationships expected by Qwen3-ASR."""
    import torch

    required_keys = {
        "attention_mask",
        "input_features",
        "input_features_mask",
        "input_ids",
        "labels",
    }
    missing = required_keys.difference(batch)
    if missing:
        raise KeyError(f"Processor batch is missing keys: {sorted(missing)}")

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    input_features = batch["input_features"]
    input_features_mask = batch["input_features_mask"]

    if input_ids.shape[0] != batch_size:
        raise ValueError(f"Expected batch size {batch_size}, received {input_ids.shape[0]}")
    if labels.shape != input_ids.shape:
        raise ValueError("labels and input_ids have different shapes")
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask and input_ids have different shapes")
    if input_features.shape[0] != batch_size:
        raise ValueError("input_features has the wrong batch dimension")
    if input_features_mask.shape[0] != batch_size:
        raise ValueError("input_features_mask has the wrong batch dimension")
    if not torch.isfinite(input_features).all():
        raise ValueError("input_features contains NaN or infinite values")

    supervised_tokens = (labels != -100).sum(dim=1)
    if torch.any(supervised_tokens == 0):
        raise ValueError("At least one example has no supervised label tokens")

    print("Processor batch:")
    for key, value in batch.items():
        shape = tuple(value.shape) if hasattr(value, "shape") else None
        dtype = getattr(value, "dtype", type(value).__name__)
        print(f"  {key}: shape={shape}, dtype={dtype}")
    print(f"  supervised tokens per example: {supervised_tokens.tolist()}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.noise_records < args.number_of_noises:
        raise ValueError("--noise-records must be at least --number-of-noises")
    if args.max_rir_records < 1 + args.number_of_noises:
        raise ValueError(
            "--max-rir-records must allow one speech RIR plus every noise RIR"
        )

    from datasets import Audio, Value, load_dataset
    from torch.utils.data import DataLoader
    from transformers import AutoProcessor

    from data_collator import Qwen3ASRDataCollator
    from SceneDataset import SceneDataset

    cache_dir = str(args.cache_dir)

    print("Streaming small dataset samples...")
    speech_stream = load_dataset(
        "openslr/librispeech_asr",
        "clean",
        split="test",
        streaming=True,
        cache_dir=cache_dir,
    ).cast_column("audio", Audio(decode=False))
    noise_stream = load_dataset(
        "bilguun/musan-noise",
        split="train",
        streaming=True,
        cache_dir=cache_dir,
    ).cast_column("audio", Value("string"))
    rir_stream = load_dataset(
        "treble-technologies/Treble10-RIR",
        split="rir_mono",
        streaming=True,
        cache_dir=cache_dir,
    ).cast_column("audio", Audio(decode=False))

    speech_ds = materialize_records(speech_stream, count=2, name="speech")
    noise_ds = materialize_records(
        noise_stream,
        count=args.noise_records,
        name="noise",
    )
    rir_ds, room, receiver, scanned = find_complete_rir_group(
        rir_stream,
        required_rirs=1 + args.number_of_noises,
        max_records=args.max_rir_records,
    )
    print(
        f"Selected {len(rir_ds)} RIRs for room={room!r}, receiver={receiver!r} "
        f"after scanning {scanned} rows."
    )

    common_dataset_args = {
        "speech_ds": speech_ds,
        "noise_ds": noise_ds,
        "rir_ds": rir_ds,
        "base_seed": args.seed,
        "sample_rate": args.sample_rate,
        "number_of_noises": args.number_of_noises,
    }
    clean_dataset = SceneDataset(**common_dataset_args, clean_probability=1.0)
    simulated_dataset = SceneDataset(**common_dataset_args, clean_probability=0.0)

    clean_scene = clean_dataset[0]
    simulated_scene = simulated_dataset[1]

    print("Raw scenes:")
    validate_scene(
        "clean",
        clean_scene,
        expected_text=str(speech_ds[0]["text"]),
        sample_rate=args.sample_rate,
    )
    validate_scene(
        "simulated",
        simulated_scene,
        expected_text=str(speech_ds[1]["text"]),
        sample_rate=args.sample_rate,
    )
    if clean_scene["metadata"].get("info") != "clean":
        raise ValueError("Clean scene is missing its clean metadata marker")
    if "final_snr_db" not in simulated_scene["metadata"]:
        raise ValueError("Simulated scene is missing its realized SNR metadata")

    print(f"Loading processor {args.model_id!r}...")
    processor = AutoProcessor.from_pretrained(args.model_id, cache_dir=cache_dir)
    processor_sample_rate = int(processor.feature_extractor.sampling_rate)
    if args.sample_rate != processor_sample_rate:
        raise ValueError(
            f"Scene rate {args.sample_rate} does not match processor rate "
            f"{processor_sample_rate}"
        )

    collator = Qwen3ASRDataCollator(
        processor=processor,
        sample_rate=args.sample_rate,
        language=args.language,
    )
    loader = DataLoader(
        [clean_scene, simulated_scene],
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )
    batch = next(iter(loader))
    validate_batch(batch, batch_size=2)

    print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

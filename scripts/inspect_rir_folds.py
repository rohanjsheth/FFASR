"""Inspect and validate the two Treble10 room folds."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from datasets import load_dataset

from data_utils.room_folds import FOLDS, ROOM_SETS

TREBLE_MONO_PARQUET = (
    "https://huggingface.co/datasets/treble-technologies/Treble10-RIR/"
    "resolve/main/data/rir_mono-00000-of-00001.parquet"
)
METADATA_COLUMNS = [
    "Room",
    "Room Description",
    "Room Volume [m³]",
    "T30",
    "Avg T30",
]


def mean_octave_band_t30(values: object) -> float:
    if isinstance(values, str):
        values = ast.literal_eval(values)
    bands = np.asarray(values, dtype=np.float64)
    if bands.shape != (8,):
        raise RuntimeError(f"Expected eight T30 octave bands, received {bands.shape}")
    return float(np.nanmean(bands))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_dataset(
        "parquet",
        data_files=TREBLE_MONO_PARQUET,
        split="train",
        columns=METADATA_COLUMNS,
        streaming=True,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
    )
    metadata = pd.DataFrame(rows)
    metadata["Room Volume [m³]"] = pd.to_numeric(metadata["Room Volume [m³]"])
    metadata["Avg T30"] = pd.to_numeric(metadata["Avg T30"])
    metadata["Raw T30 Mean"] = metadata["T30"].map(mean_octave_band_t30)

    t30_absolute_error = (metadata["Avg T30"] - metadata["Raw T30 Mean"]).abs()
    print(
        "Stored Avg T30 versus mean of raw T30 bands: "
        f"mean absolute difference={t30_absolute_error.mean():.6f} s, "
        f"maximum={t30_absolute_error.max():.6f} s"
    )
    if t30_absolute_error.max() > 0.01:
        raise RuntimeError(
            "Stored Avg T30 differs from the raw octave-band mean by more than "
            "the 0.01 s rounding tolerance"
        )

    description_counts = metadata.groupby("Room")["Room Description"].nunique()
    volume_counts = metadata.groupby("Room")["Room Volume [m³]"].nunique()
    if not description_counts.eq(1).all():
        raise RuntimeError("At least one room has multiple room descriptions")
    if not volume_counts.eq(1).all():
        raise RuntimeError("At least one room has multiple room volumes")

    room_summary = (
        metadata.groupby(["Room", "Room Description"], as_index=False)
        .agg(
            rir_rows=("Avg T30", "size"),
            mean_volume_m3=("Room Volume [m³]", "mean"),
            stored_mean_t30_s=("Avg T30", "mean"),
            raw_mean_t30_s=("Raw T30 Mean", "mean"),
        )
        .assign(
            room_type=lambda frame: frame["Room Description"].str.replace(
                r"\d+$", "", regex=True
            )
        )
    )

    expected_rooms = set().union(*ROOM_SETS.values())
    observed_rooms = set(room_summary["Room"])
    if expected_rooms != observed_rooms:
        raise RuntimeError(
            f"Fold rooms do not match dataset rooms: "
            f"missing={sorted(expected_rooms - observed_rooms)}, "
            f"extra={sorted(observed_rooms - expected_rooms)}"
        )

    display_columns = [
        "Room",
        "Room Description",
        "rir_rows",
        "mean_volume_m3",
        "stored_mean_t30_s",
        "raw_mean_t30_s",
    ]
    for name, rooms in ROOM_SETS.items():
        fold_summary = room_summary[room_summary["Room"].isin(rooms)].sort_values(
            "room_type"
        )
        type_counts = fold_summary["room_type"].value_counts()
        if len(type_counts) != 5 or not type_counts.eq(1).all():
            raise RuntimeError(
                f"Room set {name} does not contain exactly one room of each type: "
                f"{type_counts.to_dict()}"
            )

        print(f"\nRoom set {name}")
        print(fold_summary[display_columns].to_string(index=False))
        print(f"Mean room volume: {fold_summary['mean_volume_m3'].mean():.3f} m³")
        print(
            f"Mean room T30:    {fold_summary['stored_mean_t30_s'].mean():.3f} s "
            f"(stored), {fold_summary['raw_mean_t30_s'].mean():.3f} s (raw bands)"
        )

    print("\nCross-validation directions")
    for fold, (train_rooms, validation_rooms) in FOLDS.items():
        train_name = next(name for name, rooms in ROOM_SETS.items() if rooms == train_rooms)
        validation_name = next(
            name for name, rooms in ROOM_SETS.items() if rooms == validation_rooms
        )
        print(f"Fold {fold}: train={train_name}, validation={validation_name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

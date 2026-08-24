"""Convert the MIT Acoustical Impulse Response Survey into the RIR parquet schema.

Every RIR this project has trained or evaluated on is Treble-simulated, so a gain
measured against Treble10 cannot distinguish general far-field robustness from
learning to invert one simulator. These are real measured responses from ordinary
spaces, which is the only check available for that.

Two mappings are judgement calls, both documented rather than hidden:

`location` becomes `Room`, and every IR inside a location is treated as a source
at a single receiver. That mirrors Treble10, where the five RIRs at a receiver are
five source positions -- but here the measurements are separate recordings in the
same kind of space rather than one room's geometry. Locations with fewer than
`1 + number_of_noises` IRs are dropped, since the sampler draws that many without
replacement.

`Direct Path Length [m]` is estimated from the first significant arrival rather
than measured, because the survey ships no geometry. The RIRs retain propagation
delay (peaks land at 3-12 ms), so `first_arrival / sr * 343` recovers it to within
the +/-1 ms window `direct_index` searches.
"""

from __future__ import annotations

import argparse
import collections
import io
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from datasets import Dataset, Features, Value

REPO_PARQUET = (
    "hf://datasets/benjamin-paine/mit-impulse-response-survey-16khz/"
    "data/train-00000-of-00001.parquet"
)
SPEED_OF_SOUND = 343.0

FEATURES = Features(
    {
        "audio": {"bytes": Value("binary"), "path": Value("string")},
        "Room": Value("string"),
        "Room Description": Value("string"),
        "Receiver Label": Value("string"),
        "Direct Path Length [m]": Value("float32"),
    }
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=REPO_PARQUET)
    parser.add_argument("--output", type=Path, default=Path("data/mit_rir.parquet"))
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument(
        "--number-of-noises",
        type=int,
        default=2,
        help="Locations with fewer than 1 + this many IRs cannot serve a scene.",
    )
    parser.add_argument(
        "--arrival-threshold",
        type=float,
        default=0.2,
        help="First sample above this fraction of the peak counts as the direct arrival.",
    )
    return parser.parse_args(argv)


def direct_path_metres(samples: np.ndarray, sample_rate: int, threshold: float) -> float:
    """Distance implied by the first significant arrival.

    argmax alone is wrong here: in a reverberant measurement an early reflection
    can exceed the direct sound, which would place the source further away than
    it is. The first crossing is the direct path by definition.
    """
    magnitude = np.abs(samples)
    peak = magnitude.max()
    if peak <= 0.0:
        return 0.0
    above = np.flatnonzero(magnitude > threshold * peak)
    if above.size == 0:
        return 0.0
    return float(above[0] / sample_rate * SPEED_OF_SOUND)


def iter_rows(args: argparse.Namespace) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    source = load_dataset("parquet", data_files=args.source, split="train")
    rows = source.to_list()

    per_location: dict[Any, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in rows:
        per_location[record["location"]].append(record)

    needed = 1 + args.number_of_noises
    kept_locations = {k: v for k, v in per_location.items() if len(v) >= needed}
    dropped = len(rows) - sum(len(v) for v in kept_locations.values())
    print(
        f"{len(rows)} IRs across {len(per_location)} locations; "
        f"{len(kept_locations)} locations have >= {needed} IRs "
        f"({dropped} IRs dropped)"
    )

    for location, records in sorted(kept_locations.items(), key=lambda kv: str(kv[0])):
        description = next(
            (r["detail"] for r in records if r.get("detail")), "unknown"
        )
        for index, record in enumerate(records):
            samples, rate = sf.read(io.BytesIO(record["audio"]["bytes"]))
            if samples.ndim > 1:
                samples = samples.mean(axis=1)
            if rate != args.sample_rate:
                raise ValueError(f"expected {args.sample_rate} Hz, got {rate}")

            buffer = io.BytesIO()
            sf.write(
                buffer,
                samples.astype(np.float32),
                args.sample_rate,
                format="WAV",
                subtype="FLOAT",
            )
            yield {
                "audio": {"bytes": buffer.getvalue(), "path": f"loc{location}_{index}.wav"},
                "Room": f"MIT_loc{location}",
                "Room Description": str(description),
                "Receiver Label": "R0",
                "Direct Path Length [m]": direct_path_metres(
                    samples, args.sample_rate, args.arrival_threshold
                ),
            }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    dataset = Dataset.from_generator(
        iter_rows, features=FEATURES, gen_kwargs={"args": args}
    )
    dataset.to_parquet(args.output)

    distances = np.asarray(dataset["Direct Path Length [m]"], dtype=np.float64)
    print(f"\nwrote {args.output}  rows={len(dataset)}  "
          f"rooms={len(set(dataset['Room']))}  "
          f"{args.output.stat().st_size / 1e6:.1f} MB")
    print(f"estimated distance: min {distances.min():.2f}m  "
          f"median {np.median(distances):.2f}m  max {distances.max():.2f}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

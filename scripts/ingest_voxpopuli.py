"""Convert ArtificialAnalysis/VoxPopuli-Cleaned-AA into the eval's speech schema.

VoxPopuli is a harder speech domain than LibriSpeech test-clean -- semi-spontaneous
parliamentary English, largely non-native -- which makes it a test of whether a
far-field gain measured on read audiobooks survives a different speaker
population, and a partial recalibration of an eval that is currently much easier
than the benchmark it proxies.

The upstream repo is a JSONL manifest plus loose WAVs rather than a parquet, so
this fetches both and writes the three columns the eval actually reads: `audio`
as embedded bytes, `id`, and `text`.

`transcript` is used rather than facebook/voxpopuli's `normalized_text`. That
field has already been through someone else's normalizer, which strips
apostrophes -- and `don't` normalizes to `do not` while `dont` stays `dont`, so
every contraction would cost two word errors against a model that spells them
correctly. Passing real text through our own normalizer keeps both sides of the
comparison on one pipeline.
"""

from __future__ import annotations

import argparse
import io
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from datasets import Dataset, Features, Value
from huggingface_hub import HfFileSystem
from scipy.signal import resample_poly

REPO = "datasets/ArtificialAnalysis/VoxPopuli-Cleaned-AA"
MANIFEST = "voxpopuli_cleaned_aa_v1.jsonl"

FEATURES = Features(
    {
        # struct<bytes, path> rather than Audio(): Audio() re-encodes through
        # torchcodec, which needs FFmpeg and gains nothing over valid WAV bytes.
        "audio": {"bytes": Value("binary"), "path": Value("string")},
        "id": Value("string"),
        "text": Value("string"),
        "duration": Value("float32"),
    }
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument("--output", type=Path, default=Path("data/voxpopuli_en.parquet"))
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--language", default="en")
    parser.add_argument(
        "--min-duration-seconds",
        type=float,
        default=2.0,
        help=(
            "Matches the training filter. Reverb tails run to ~1s and noise "
            "offsets are drawn up to 3000 ms, so shorter clips never express "
            "the far-field effect being measured."
        ),
    )
    return parser.parse_args(argv)


def iter_rows(args: argparse.Namespace) -> Iterator[dict[str, Any]]:
    fs = HfFileSystem()
    with fs.open(f"{args.repo}/{args.manifest}", "r") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    kept = skipped_language = skipped_short = 0
    for record in records:
        if record.get("language") != args.language:
            skipped_language += 1
            continue
        if float(record.get("duration", 0.0)) < args.min_duration_seconds:
            skipped_short += 1
            continue

        with fs.open(f"{args.repo}/{record['url']}", "rb") as handle:
            samples, source_rate = sf.read(io.BytesIO(handle.read()))
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        if source_rate != args.sample_rate:
            samples = resample_poly(samples, args.sample_rate, source_rate)

        buffer = io.BytesIO()
        sf.write(
            buffer,
            samples.astype(np.float32),
            args.sample_rate,
            format="WAV",
            subtype="FLOAT",
        )
        kept += 1
        if kept % 100 == 0:
            print(f"  {kept} utterances", flush=True)

        yield {
            "audio": {"bytes": buffer.getvalue(), "path": record["file_name"]},
            "id": record["id"],
            "text": record["transcript"],
            "duration": float(record["duration"]),
        }

    print(
        f"kept {kept}, skipped {skipped_language} non-{args.language}, "
        f"{skipped_short} under {args.min_duration_seconds}s"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    dataset = Dataset.from_generator(
        iter_rows, features=FEATURES, gen_kwargs={"args": args}
    )
    dataset.to_parquet(args.output)

    durations = np.asarray(dataset["duration"], dtype=np.float64)
    print(f"\nwrote {args.output}  rows={len(dataset)}  "
          f"{args.output.stat().st_size / 1e6:.1f} MB")
    print(f"duration: min {durations.min():.2f}s  median "
          f"{np.median(durations):.2f}s  max {durations.max():.2f}s")
    apostrophes = sum("'" in text for text in dataset["text"])
    print(f"references with an apostrophe: {apostrophes}/{len(dataset)} "
          f"({100 * apostrophes / len(dataset):.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Assemble one speech parquet from several corpora. Run on the pod.

Training on LibriTTS alone gave a fine-tune whose far-field gain was -2.71 pp on
its own speech domain and -0.65 pp on real distant-microphone audio.
"""

from __future__ import annotations

import argparse
import io
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from datasets import Audio, Dataset, Features, Value, load_dataset
from scipy.signal import resample_poly

FEATURES = Features({
    # struct<bytes, path>, not Audio(): Audio() re-encodes through torchcodec.
    "audio": {"bytes": Value("binary"), "path": Value("string")},
    "id": Value("string"),
    "text_normalized": Value("string"),
    "domain": Value("string"),
    "duration": Value("float32"),
})

# Output column is text_normalized, matching configs/train.toml.
# Source text must be cased and punctuated or it teaches the decoder to drop both.
SOURCES: dict[str, dict[str, Any]] = {
    "libritts_r": {
        "repo": "mythicinfinity/libritts_r", "config": "clean", "split": "train.clean.360",
        "text": "text_normalized", "id": "id",
    },
    "vctk": {
        "repo": "sanchit-gandhi/vctk", "config": None, "split": "train",
        "text": "text", "id": "text_id",
    },
    "common_voice": {
        "repo": "mozilla-foundation/common_voice_17_0", "config": "en", "split": "train",
        "text": "sentence", "id": "path",
    },
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", action="append", required=True, metavar="NAME[:CAP]",
        help="Repeatable, e.g. --source vctk:40000. CAP is a max utterance count.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf_cache"))
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument(
        "--min-duration-seconds", type=float, default=2.0,
        help="Reverb tails run to ~1s and noise offsets to 3000ms, so shorter clips never express the effect.",
    )
    parser.add_argument("--max-duration-seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report per-source survival against the duration filter, write nothing.")
    parser.add_argument("--dry-run-sample", type=int, default=400)
    return parser.parse_args(argv)


def parse_sources(entries: Sequence[str]) -> list[tuple[str, int | None]]:
    parsed = []
    for entry in entries:
        name, _, cap = entry.partition(":")
        if name not in SOURCES:
            raise ValueError(f"Unknown source {name!r}; available: {sorted(SOURCES)}")
        parsed.append((name, int(cap) if cap else None))
    return parsed


def decode(raw: bytes, sample_rate: int) -> tuple[np.ndarray, float]:
    samples, rate = sf.read(io.BytesIO(raw), dtype="float64")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    duration = samples.size / rate
    if rate != sample_rate:
        samples = resample_poly(samples, sample_rate, rate)
    return samples, duration


def stream(name: str, cache_dir: Path):
    source = SOURCES[name]
    return load_dataset(
        source["repo"], source["config"], split=source["split"],
        cache_dir=str(cache_dir), streaming=True,
    ).cast_column("audio", Audio(decode=False))


def iter_rows(args: argparse.Namespace, sources: list[tuple[str, int | None]]) -> Iterator[dict[str, Any]]:
    for name, cap in sources:
        source = SOURCES[name]
        kept = seen = skipped_empty = skipped_duration = 0
        for record in stream(name, args.cache_dir):
            seen += 1
            text = str(record[source["text"]]).strip()
            if not text:
                skipped_empty += 1
                continue
            samples, duration = decode(record["audio"]["bytes"], args.sample_rate)
            if not args.min_duration_seconds <= duration <= args.max_duration_seconds:
                skipped_duration += 1
                continue
            buffer = io.BytesIO()
            sf.write(buffer, samples.astype(np.float32), args.sample_rate,
                     format="WAV", subtype="FLOAT")
            identifier = f"{name}/{record[source['id']]}"
            kept += 1
            if kept % 2000 == 0:
                print(f"  {name}: {kept} kept of {seen} seen", flush=True)
            yield {
                "audio": {"bytes": buffer.getvalue(), "path": f"{identifier}.wav"},
                "id": identifier, "text_normalized": text, "domain": name,
                "duration": float(samples.size / args.sample_rate),
            }
            if cap is not None and kept >= cap:
                break
        print(f"{name}: kept {kept} of {seen} seen "
              f"({skipped_empty} empty, {skipped_duration} outside duration range)", flush=True)


def dry_run(args: argparse.Namespace, sources: list[tuple[str, int | None]]) -> int:
    print(f"Survival against the {args.min_duration_seconds}-{args.max_duration_seconds}s filter, "
          f"first {args.dry_run_sample} utterances each:\n")
    for name, _ in sources:
        source = SOURCES[name]
        durations = []
        for index, record in enumerate(stream(name, args.cache_dir)):
            if index >= args.dry_run_sample:
                break
            _, duration = decode(record["audio"]["bytes"], args.sample_rate)
            durations.append(duration)
        d = np.asarray(durations)
        survive = ((d >= args.min_duration_seconds) & (d <= args.max_duration_seconds)).mean()
        print(f"{name:<14} n={d.size:<5} median {np.median(d):5.2f}s  "
              f"mean {d.mean():5.2f}s  survives {100 * survive:5.1f}%")
        print(f"{'':<14} sample text: {str(next(iter(stream(name, args.cache_dir)))[source['text']])[:80]!r}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    sources = parse_sources(args.source)
    if args.dry_run:
        return dry_run(args, sources)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    dataset = Dataset.from_generator(
        iter_rows, features=FEATURES, gen_kwargs={"args": args, "sources": sources}
    )
    dataset = dataset.shuffle(seed=args.seed)
    dataset.to_parquet(args.output)

    domains = np.asarray(dataset["domain"])
    durations = np.asarray(dataset["duration"], dtype=np.float64)
    print(f"\nwrote {args.output}  rows={len(dataset)}  "
          f"{args.output.stat().st_size / 1e9:.2f} GB  {durations.sum() / 3600:.1f} h")
    print(f"{'domain':<14}{'utterances':>11}{'hours':>8}{'share':>8}")
    for name in sorted(set(domains.tolist())):
        mask = domains == name
        print(f"{name:<14}{int(mask.sum()):>11}{durations[mask].sum() / 3600:>8.1f}"
              f"{100 * durations[mask].sum() / durations.sum():>7.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

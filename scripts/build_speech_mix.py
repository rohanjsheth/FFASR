"""Assemble one speech parquet from several corpora. Run on the pod.

Training on LibriTTS alone gave a fine-tune whose far-field gain was -2.71 pp on
its own speech domain and -0.65 pp on real distant-microphone audio.
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
        "repo": "mythicinfinity/libritts_r", "files": "train.clean.360",
        "text": "text_normalized", "id": "id",
    },
    # id from `file`: text_id is the prompt, and VCTK records each prompt twice
    # (mic1/mic2), so text_id collides across speakers and mics.
    "vctk": {
        "repo": "sanchit-gandhi/vctk", "files": "train",
        "text": "text", "id": "file", "id_stem": True,
    },
    # Mozilla emptied the official HF repo in Oct 2025; this public CC0 mirror
    # keeps CV's native layout, so it needs the tar+tsv reader rather than parquet.
    "common_voice": {
        "repo": "fsicoli/common_voice_17_0", "kind": "tar",
        "files": "audio/en/train/", "tsv": "transcript/en/train.tsv",
        "text": "sentence", "id": "path",
    },
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", action="append", required=True, metavar="NAME[:CAP]",
        help="Repeatable, e.g. --source vctk:40000. CAP is a max utterance count.",
    )
    parser.add_argument("--output", type=Path, help="Required unless --dry-run.")
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf_cache"))
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument(
        "--min-duration-seconds", type=float, default=2.0,
        help="Reverb tails run to ~1s and noise offsets to 3000ms, so shorter clips never express the effect.",
    )
    parser.add_argument("--max-duration-seconds", type=float, default=30.0)
    parser.add_argument("--validation-output", type=Path,
                        help="Also write a held-out slice, balanced across domains.")
    parser.add_argument("--validation-per-domain", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-proc", type=int, default=16,
                        help="Worker processes; the parquet shards are split across them.")
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


def shard_files(name: str) -> list[str]:
    """The split's shards: parquet hf:// paths, or repo-relative tar names."""
    from huggingface_hub import HfApi

    source = SOURCES[name]
    listing = HfApi().list_repo_files(source["repo"], repo_type="dataset")
    if source.get("kind") == "tar":
        files = sorted(f for f in listing if f.startswith(source["files"]) and f.endswith(".tar"))
        if not files:
            raise ValueError(f"No tars under {source['files']!r} in {source['repo']}")
        return files
    files = sorted(f for f in listing if f.endswith(".parquet") and source["files"] in f)
    if not files:
        raise ValueError(f"No parquet shards matching {source['files']!r} in {source['repo']}")
    return [f"hf://datasets/{source['repo']}/{f}" for f in files]


def tar_records(name: str, cache_dir: Path, files: Sequence[str]) -> Iterator[dict[str, Any]]:
    """Yield {audio bytes, text, id} from CV tars, joined to the split manifest."""
    import tarfile

    import pyarrow.csv as pv
    from huggingface_hub import hf_hub_download

    source = SOURCES[name]
    manifest = hf_hub_download(source["repo"], source["tsv"], repo_type="dataset",
                               cache_dir=str(cache_dir))
    # quote_char=False: CV sentences contain unbalanced quotes, and with quoting on
    # pyarrow swallows following lines into one field, yielding 15k-char "transcripts"
    # made of raw TSV. newlines_in_values must be off for the same reason.
    table = pv.read_csv(
        manifest,
        parse_options=pv.ParseOptions(delimiter="\t", quote_char=False, newlines_in_values=False),
        convert_options=pv.ConvertOptions(include_columns=[source["id"], source["text"]]),
    )
    sentences = dict(zip(table[source["id"]].to_pylist(), table[source["text"]].to_pylist()))
    del table

    for shard in files:
        path = hf_hub_download(source["repo"], shard, repo_type="dataset", cache_dir=str(cache_dir))
        with tarfile.open(path) as archive:
            for member in archive:
                if not member.isfile():
                    continue
                text = sentences.get(Path(member.name).name)
                if text is None:
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                yield {"raw": handle.read(), "text": text, "id": Path(member.name).name}


def stream(name: str, cache_dir: Path, files: Sequence[str] | None = None):
    return load_dataset(
        "parquet", data_files=list(files) if files else shard_files(name),
        split="train", cache_dir=str(cache_dir), streaming=True,
    ).cast_column("audio", Audio(decode=False))


def records(name: str, cache_dir: Path, files: Sequence[str] | None = None
            ) -> Iterator[tuple[bytes, str, str]]:
    source = SOURCES[name]
    shards = list(files) if files else shard_files(name)
    if source.get("kind") == "tar":
        for record in tar_records(name, cache_dir, shards):
            yield record["raw"], str(record["text"]), str(record["id"])
        return
    for record in stream(name, cache_dir, shards):
        identifier = str(record[source["id"]])
        if source.get("id_stem"):
            identifier = Path(identifier).stem
        yield record["audio"]["bytes"], str(record[source["text"]]), identifier


def iter_rows(args: argparse.Namespace, specs: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for spec in specs:
        name, files, cap = spec["name"], spec["files"], spec["cap"]
        source = SOURCES[name]
        kept = seen = skipped_empty = skipped_duration = 0
        for raw, raw_text, record_id in records(name, args.cache_dir, files):
            seen += 1
            text = raw_text.strip()
            if not text:
                skipped_empty += 1
                continue
            samples, duration = decode(raw, args.sample_rate)
            if not args.min_duration_seconds <= duration <= args.max_duration_seconds:
                skipped_duration += 1
                continue
            buffer = io.BytesIO()
            sf.write(buffer, samples.astype(np.float32), args.sample_rate,
                     format="WAV", subtype="FLOAT")
            identifier = f"{name}/{record_id}"
            kept += 1
            yield {
                "audio": {"bytes": buffer.getvalue(), "path": f"{identifier}.wav"},
                "id": identifier, "text_normalized": text, "domain": name,
                "duration": float(samples.size / args.sample_rate),
            }
            if cap is not None and kept >= cap:
                break
        print(f"{name} shard done: kept {kept} of {seen} "
              f"({skipped_empty} empty, {skipped_duration} outside duration range)", flush=True)


def build_specs(sources: list[tuple[str, int | None]], num_proc: int) -> list[dict[str, Any]]:
    """One spec per worker per source, so from_generator can shard the list."""
    specs = []
    for name, cap in sources:
        files = shard_files(name)
        groups = [files[i::num_proc] for i in range(num_proc)]
        groups = [g for g in groups if g]
        # A cap is per worker, so the total stays near what was asked for.
        per_worker = None if cap is None else max(1, cap // len(groups))
        specs.extend({"name": name, "files": g, "cap": per_worker} for g in groups)
        print(f"{name}: {len(files)} shards over {len(groups)} workers"
              + ("" if cap is None else f", cap {per_worker}/worker"))
    return specs


def envelope_stats(samples: np.ndarray, sample_rate: int) -> tuple[float, float] | None:
    """Syllabic-modulation share and log-envelope dynamic range; higher = drier."""
    hop = max(1, round(sample_rate * 0.01))
    usable = (samples.size // hop) * hop
    if usable < hop * 50:
        return None
    envelope = np.abs(samples[:usable]).reshape(-1, hop).mean(1)
    peak = envelope.max()
    if peak <= 0:
        return None
    envelope = envelope / peak
    speech = envelope > np.percentile(envelope, 20)
    if speech.sum() < 20:
        return None
    log_envelope = 20 * np.log10(envelope[speech] + 1e-6)
    dynamic = float(np.percentile(log_envelope, 95) - np.percentile(log_envelope, 10))

    centred = envelope - envelope.mean()
    spectrum = np.abs(np.fft.rfft(centred * np.hanning(centred.size))) ** 2
    freq = np.fft.rfftfreq(centred.size, hop / sample_rate)
    band = spectrum[(freq >= 0.5) & (freq <= 20)].sum()
    if band <= 0:
        return None
    return float(spectrum[(freq >= 3) & (freq <= 8)].sum() / band), dynamic


def dry_run(args: argparse.Namespace, sources: list[tuple[str, int | None]]) -> int:
    print(f"Survival against the {args.min_duration_seconds}-{args.max_duration_seconds}s filter, "
          f"first {args.dry_run_sample} utterances each:\n")
    for name, _ in sources:
        source = SOURCES[name]
        durations, sample_text, stats = [], "", []
        for index, (raw, raw_text, _) in enumerate(records(name, args.cache_dir)):
            if index >= args.dry_run_sample:
                break
            if not sample_text:
                sample_text = raw_text[:80]
            samples, duration = decode(raw, args.sample_rate)
            durations.append(duration)
            measured = envelope_stats(samples, args.sample_rate)
            if measured is not None:
                stats.append(measured)
        d = np.asarray(durations)
        survive = ((d >= args.min_duration_seconds) & (d <= args.max_duration_seconds)).mean()
        s = np.asarray(stats)
        print(f"{name:<14} n={d.size:<5} median {np.median(d):5.2f}s  "
              f"mean {d.mean():5.2f}s  survives {100 * survive:5.1f}%  "
              f"syll-mod {s[:, 0].mean():.3f}  dyn-range {s[:, 1].mean():5.2f} dB")
        print(f"{'':<14} sample text: {sample_text!r}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    sources = parse_sources(args.source)
    if args.dry_run:
        return dry_run(args, sources)
    if args.output is None:
        raise ValueError("--output is required unless --dry-run")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    specs = build_specs(sources, args.num_proc)
    dataset = Dataset.from_generator(
        iter_rows, features=FEATURES, num_proc=min(args.num_proc, len(specs)),
        gen_kwargs={"args": args, "specs": specs},
    )
    dataset = dataset.shuffle(seed=args.seed)

    if args.validation_output is not None:
        if args.validation_output.exists():
            raise FileExistsError(args.validation_output)
        held, remaining, counts = [], [], collections.Counter()
        for index, domain in enumerate(dataset["domain"]):
            if counts[domain] < args.validation_per_domain:
                counts[domain] += 1
                held.append(index)
            else:
                remaining.append(index)
        dataset.select(held).to_parquet(args.validation_output)
        dataset = dataset.select(remaining)
        print(f"held out {len(held)} for validation "
              f"({dict(sorted(counts.items()))}) -> {args.validation_output}")

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

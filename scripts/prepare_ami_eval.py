"""Freeze a paired AMI headset/distant-microphone corpus for far-field scoring.

Every far-field number we have so far was produced by our own renderer: clean
speech convolved with a Treble or MIT RIR and mixed with MUSAN or AID at a
target SNR. That shares its data-generating process with training, which is why
`results/noise_eval` disagreed with the FFASR leaderboard. AMI removes the
renderer entirely. It is real meeting speech captured simultaneously by a
headset (IHM) and by a single distant microphone in the same room (SDM), so the
same utterance exists in a near-field and a far-field version with **no
simulated RIR, no synthetic noise and no SNR targeting** -- the reverberation,
the babble and the level difference are whatever the room actually did.

The two microphone configurations are separate dataset configs with independently
shuffled rows, so this joins them on the segment identity encoded in `audio_id`
(`AMI_{meeting}_{mic}_{speaker}_{start}_{end}`) and re-checks meeting, speaker,
timestamps and reference text before accepting a pair. Row order is never
trusted.

Each accepted segment is annotated with the fraction of its span covered by
another speaker's segment. Cross-talk is the one degradation the headset does
not share with the distant microphone, so a distant-microphone regression that
lives entirely in the overlapped subset is a different finding from one that is
uniform, and the two are not separable after the fact.

Audio is decoded and re-encoded to canonical float32 WAV bytes with a recorded
SHA-256, so both models provably score identical samples. No gain, filtering or
resampling is applied; AMI ships 16 kHz mono, which is already the model rate.
"""

from __future__ import annotations

import argparse
import io
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download, list_repo_files

from eval_utils.text_norm import normalize_for_wer
from scripts.noise_eval_io import sha256, wav_bytes, write_parquet

REPO = "edinburghcstr/ami"
MICROPHONES = ("ihm", "sdm")
METADATA_COLUMNS = (
    "meeting_id",
    "audio_id",
    "text",
    "begin_time",
    "end_time",
    "microphone_id",
    "speaker_id",
)
# AMI timestamps come from the same annotation for both configs, so a pair that
# disagrees by more than a sample period is a join error, not a rounding one.
TIMESTAMP_TOLERANCE_SECONDS = 1e-3


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--utterances",
        type=int,
        default=800,
        help=(
            "Paired segments to keep, allocated as evenly as the meetings allow "
            "so no single meeting decides the result. Each one is written twice, "
            "once per microphone."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument(
        "--min-duration-seconds",
        type=float,
        default=2.0,
        help=(
            "Matches the filter used everywhere else in this repo. AMI is full "
            "of sub-second backchannels whose references are one or two words, "
            "and corpus WER over those is dominated by insertion behaviour "
            "rather than by whether the words were heard."
        ),
    )
    parser.add_argument("--max-duration-seconds", type=float, default=30.0)
    parser.add_argument(
        "--output", type=Path, default=Path("data/ami_paired.parquet")
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf_cache"))
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON audit written alongside the corpus.",
    )
    return parser.parse_args(argv)


def shard_paths(repo: str, microphone: str, split: str, cache_dir: Path) -> list[Path]:
    """Download the split's parquet shards; audio arrives as embedded WAV bytes."""
    prefix = f"{microphone}/{split}-"
    names = sorted(
        name for name in list_repo_files(repo, repo_type="dataset")
        if name.startswith(prefix) and name.endswith(".parquet")
    )
    if not names:
        raise ValueError(f"No {split} shards under {microphone}/ in {repo}")
    return [
        Path(
            hf_hub_download(
                repo, name, repo_type="dataset", cache_dir=str(cache_dir)
            )
        )
        for name in names
    ]


def read_metadata(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(pq.read_table(path, columns=list(METADATA_COLUMNS)).to_pylist())
    return rows


def segment_key(audio_id: str) -> tuple[str, str, str, str]:
    """Identity of the annotated segment, with the microphone field removed."""
    parts = audio_id.split("_")
    if len(parts) != 6 or parts[0] != "AMI":
        raise ValueError(f"Unexpected AMI audio_id: {audio_id!r}")
    _, meeting, _microphone, speaker, start, end = parts
    return meeting, speaker, start, end


def overlap_ratios(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Fraction of each segment covered by some other speaker's segment.

    Computed over every annotated segment in the meeting, including ones this
    corpus later drops, because a neighbouring backchannel still lands in the
    distant microphone whether or not it is scored.
    """
    by_meeting: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_meeting.setdefault(row["meeting_id"], []).append(row)

    ratios: dict[str, float] = {}
    for segments in by_meeting.values():
        ordered = sorted(segments, key=lambda row: float(row["begin_time"]))
        starts = np.array([float(row["begin_time"]) for row in ordered])
        ends = np.array([float(row["end_time"]) for row in ordered])
        speakers = np.array([row["speaker_id"] for row in ordered])
        for index, row in enumerate(ordered):
            begin, end = starts[index], ends[index]
            duration = end - begin
            if duration <= 0:
                ratios[row["audio_id"]] = 0.0
                continue
            others = (speakers != speakers[index]) & (starts < end) & (ends > begin)
            spans = np.stack(
                [np.maximum(starts[others], begin), np.minimum(ends[others], end)]
            ).T if others.any() else np.empty((0, 2))
            covered = 0.0
            cursor = begin
            for span_start, span_end in sorted(spans.tolist()):
                if span_end <= cursor:
                    continue
                covered += span_end - max(span_start, cursor)
                cursor = max(cursor, span_end)
            ratios[row["audio_id"]] = float(min(1.0, covered / duration))
    return ratios


def pair_segments(
    metadata: dict[str, list[dict[str, Any]]],
) -> tuple[dict[tuple[str, str, str, str], dict[str, dict[str, Any]]], dict[str, int]]:
    """Join the microphone configs on segment identity, re-verifying the fields."""
    indexed = {
        microphone: {segment_key(row["audio_id"]): row for row in rows}
        for microphone, rows in metadata.items()
    }
    for microphone, rows in metadata.items():
        if len(indexed[microphone]) != len(rows):
            raise ValueError(f"Duplicate segment keys in {microphone}")

    shared = set.intersection(*(set(index) for index in indexed.values()))
    paired: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = {}
    mismatched = 0
    for key in sorted(shared):
        rows = {microphone: indexed[microphone][key] for microphone in indexed}
        reference = rows[MICROPHONES[0]]
        agrees = all(
            row["meeting_id"] == reference["meeting_id"]
            and row["speaker_id"] == reference["speaker_id"]
            and row["text"] == reference["text"]
            and abs(float(row["begin_time"]) - float(reference["begin_time"]))
            <= TIMESTAMP_TOLERANCE_SECONDS
            and abs(float(row["end_time"]) - float(reference["end_time"]))
            <= TIMESTAMP_TOLERANCE_SECONDS
            for row in rows.values()
        )
        if not agrees:
            mismatched += 1
            continue
        paired[key] = rows

    counts = {
        "mismatched_pairs_dropped": mismatched,
        **{f"segments_{microphone}": len(rows) for microphone, rows in metadata.items()},
    }
    return paired, counts


def eligible(
    paired: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]],
    args: argparse.Namespace,
) -> tuple[list[tuple[str, str, str, str]], dict[str, int]]:
    kept: list[tuple[str, str, str, str]] = []
    dropped = {"empty_reference": 0, "too_short": 0, "too_long": 0}
    for key, rows in paired.items():
        reference = rows[MICROPHONES[0]]
        duration = float(reference["end_time"]) - float(reference["begin_time"])
        if not normalize_for_wer(reference["text"]):
            dropped["empty_reference"] += 1
        elif duration < args.min_duration_seconds:
            dropped["too_short"] += 1
        elif duration > args.max_duration_seconds:
            dropped["too_long"] += 1
        else:
            kept.append(key)
    return kept, dropped


def allocate(
    keys: Sequence[tuple[str, str, str, str]], wanted: int, seed: int
) -> list[tuple[str, str, str, str]]:
    """Spread the sample across meetings, taking the shortfall from the rest."""
    rng = np.random.default_rng(seed)
    by_meeting: dict[str, list[tuple[str, str, str, str]]] = {}
    for key in keys:
        by_meeting.setdefault(key[0], []).append(key)

    if wanted > len(keys):
        raise ValueError(f"Only {len(keys)} eligible pairs; requested {wanted}")

    shuffled = {
        meeting: [
            members[index]
            for index in rng.permutation(len(members))
        ]
        for meeting, members in sorted(by_meeting.items())
    }
    chosen: list[tuple[str, str, str, str]] = []
    taken = {meeting: 0 for meeting in shuffled}
    while len(chosen) < wanted:
        available = [m for m in shuffled if taken[m] < len(shuffled[m])]
        if not available:
            raise ValueError("Ran out of eligible pairs while allocating")
        for meeting in available:
            if len(chosen) == wanted:
                break
            chosen.append(shuffled[meeting][taken[meeting]])
            taken[meeting] += 1
    return chosen


def iter_audio_rows(
    paths: Sequence[Path], selected: set[tuple[str, str, str, str]]
) -> Iterator[dict[str, Any]]:
    """Stream only the wanted rows, one parquet row group at a time."""
    for path in paths:
        parquet = pq.ParquetFile(path)
        for group in range(parquet.metadata.num_row_groups):
            table = parquet.read_row_group(group, columns=["audio_id", "audio"])
            for row in table.to_pylist():
                if segment_key(row["audio_id"]) in selected:
                    yield row


def canonical_audio(encoded: bytes, sample_rate: int, audio_id: str) -> tuple[bytes, int]:
    audio, rate = sf.read(io.BytesIO(encoded), dtype="float64")
    if rate != sample_rate:
        raise ValueError(f"{audio_id} is {rate} Hz; no resampling is applied here")
    if audio.ndim != 1:
        raise ValueError(f"{audio_id} is not mono")
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError(f"{audio_id} is empty or non-finite")
    return wav_bytes(audio, sample_rate), audio.size


def build_rows(
    args: argparse.Namespace,
    shards: dict[str, list[Path]],
    paired: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]],
    ratios: dict[str, float],
    selected: Sequence[tuple[str, str, str, str]],
) -> Iterator[dict[str, Any]]:
    wanted = set(selected)
    order = {key: index for index, key in enumerate(selected)}
    for microphone in MICROPHONES:
        found = 0
        for row in iter_audio_rows(shards[microphone], wanted):
            key = segment_key(row["audio_id"])
            metadata = paired[key][microphone]
            encoded, samples = canonical_audio(
                row["audio"]["bytes"], args.sample_rate, row["audio_id"]
            )
            found += 1
            if found % 100 == 0:
                print(f"  {microphone}: {found}/{len(wanted)}", flush=True)
            yield {
                "id": f"{microphone}/{'_'.join(key)}",
                "pair_id": "_".join(key),
                "pair_index": order[key],
                "microphone": microphone,
                "audio_id": row["audio_id"],
                "audio": {"bytes": encoded, "path": f"{microphone}/{row['audio_id']}.wav"},
                "text": metadata["text"],
                "meeting_id": metadata["meeting_id"],
                "speaker_id": metadata["speaker_id"],
                "microphone_id": metadata["microphone_id"],
                "begin_time": float(metadata["begin_time"]),
                "end_time": float(metadata["end_time"]),
                "duration_seconds": samples / args.sample_rate,
                "overlap_ratio": ratios[row["audio_id"]],
                "sample_rate": args.sample_rate,
                "sha256": sha256(encoded),
            }
        if found != len(wanted):
            raise ValueError(
                f"Expected {len(wanted)} {microphone} rows, materialized {found}"
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)

    shards = {
        microphone: shard_paths(args.repo, microphone, args.split, args.cache_dir)
        for microphone in MICROPHONES
    }
    metadata = {
        microphone: read_metadata(paths) for microphone, paths in shards.items()
    }
    for microphone, rows in metadata.items():
        print(f"{microphone}: {len(rows)} annotated segments")

    paired, counts = pair_segments(metadata)
    print(f"paired on segment identity: {len(paired)} "
          f"({counts['mismatched_pairs_dropped']} rejected on field mismatch)")

    ratios = overlap_ratios(metadata[MICROPHONES[0]])
    ratios.update(overlap_ratios(metadata[MICROPHONES[1]]))

    keys, dropped = eligible(paired, args)
    print(f"eligible after filters: {len(keys)} "
          f"(dropped {dropped['empty_reference']} empty, {dropped['too_short']} short, "
          f"{dropped['too_long']} long)")

    selected = allocate(keys, args.utterances, args.seed)
    per_meeting: dict[str, int] = {}
    for key in selected:
        per_meeting[key[0]] = per_meeting.get(key[0], 0) + 1
    print(f"selected {len(selected)} pairs across {len(per_meeting)} meetings")

    written = write_parquet(
        args.output,
        build_rows(args, shards, paired, ratios, selected),
    )
    print(f"\nwrote {args.output}  rows={written}  "
          f"{args.output.stat().st_size / 1e6:.1f} MB")

    overlaps = np.array([ratios[paired[k][MICROPHONES[0]]["audio_id"]] for k in selected])
    durations = np.array(
        [
            float(paired[k][MICROPHONES[0]]["end_time"])
            - float(paired[k][MICROPHONES[0]]["begin_time"])
            for k in selected
        ]
    )
    report = {
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "microphones": list(MICROPHONES),
        "rows": written,
        "pairs": len(selected),
        "segment_counts": counts,
        "dropped": dropped,
        "eligible_pairs": len(keys),
        "pairs_per_meeting": per_meeting,
        "duration_seconds": {
            "total": float(durations.sum()),
            "mean": float(durations.mean()),
            "min": float(durations.min()),
            "max": float(durations.max()),
        },
        "overlap_ratio": {
            "mean": float(overlaps.mean()),
            "clean_fraction": float((overlaps == 0).mean()),
            "overlapped_fraction": float((overlaps > 0.2).mean()),
        },
    }
    print(f"audio: {durations.sum() / 3600:.2f} h per microphone, "
          f"mean {durations.mean():.2f}s")
    print(f"overlap: {100 * (overlaps == 0).mean():.0f}% none, "
          f"{100 * (overlaps > 0.2).mean():.0f}% above 0.2")
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

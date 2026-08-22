"""Convert Meta's AcousticRooms into the RIR parquet schema the pipeline reads.

Writing Treble10's schema means nothing in the training path changes -- point
`[datasets].rir_parquet` at the output and the sampler, `get_groups`, and
`render_scene_from_recipe` all work unmodified.

Three things about the source data drove the decisions here, each verified
against Treble10 rather than assumed:

RIRs are used AS SHIPPED. Each metadata file may carry an `IR_norm` scalar, but
regressing RIR energy on source-receiver distance shows the shipped waveforms
already preserve the physical falloff (corr -0.88, energy ~ r^-1.43 in an
auditorium) while both multiplying and dividing by `IR_norm` degrade it. Half the
rooms omit the field entirely, and it correlates with distance at only -0.16.

The join is on PARSED INTEGERS. Metadata names receivers `R0010` where the audio
names the same receiver `R010`, so string matching silently drops every receiver
above nine. Parsing `S(\\d+)_R(\\d+)` to ints matches 100% in spot checks.

Level matters because the mixer sums both noise sources and applies a SINGLE
gain to hit the target SNR, so their relative loudness comes entirely from RIR
energy. Peak-normalised RIRs would erase the near/far distinction.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from datasets import Dataset, Features, Value
from scipy.signal import resample_poly

from audio_utils.audio_mixing import direct_index

STEM = re.compile(r"S(\d+)_R(\d+)")
ROOM_SUFFIX = re.compile(r"_idx_\d+$")

FEATURES = Features(
    {
        # struct<bytes, path>, not Audio(): Audio() re-encodes through torchcodec,
        # which needs FFmpeg and gains nothing when we already hold WAV bytes.
        "audio": {"bytes": Value("binary"), "path": Value("string")},
        "Room": Value("string"),
        "Room Description": Value("string"),
        "Receiver Label": Value("string"),
        "Direct Path Length [m]": Value("float32"),
        # Absent upstream. Computed here so the parquet is self-describing and
        # hardness stratification does not need a second pass over the audio.
        "Avg C50": Value("float32"),
        "Avg T30": Value("float32"),
    }
)


def clarity_c50_db(rir: np.ndarray, direct: int, sample_rate: int) -> float:
    """Energy in the first 50 ms after the direct sound, against everything later."""
    split = direct + int(0.05 * sample_rate)
    early = float(np.sum(rir[direct:split] ** 2))
    late = float(np.sum(rir[split:] ** 2))
    if late <= 0.0 or early <= 0.0:
        return float("nan")
    return 10.0 * np.log10(early / late)


def reverberation_t30_s(rir: np.ndarray, direct: int, sample_rate: int) -> float:
    """Schroeder backward integration, fitted between -5 dB and -35 dB.

    Returns nan when the decay never reaches -35 dB, which happens on short
    RIRs from small rooms rather than being an error.
    """
    tail = rir[direct:]
    if tail.size == 0:
        return float("nan")
    energy = np.cumsum(tail[::-1] ** 2)[::-1]
    if energy[0] <= 0.0:
        return float("nan")
    curve = 10.0 * np.log10(np.maximum(energy / energy[0], 1e-20))
    below5 = np.flatnonzero(curve <= -5.0)
    below35 = np.flatnonzero(curve <= -35.0)
    if below5.size == 0 or below35.size == 0:
        return float("nan")
    return float(2.0 * (below35[0] - below5[0]) / sample_rate)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir-dir", type=Path, default=Path("single_channel_ir_1"))
    parser.add_argument("--metadata-dir", type=Path, default=Path("metadata"))
    parser.add_argument("--output", type=Path, default=Path("data/acoustic_rooms.parquet"))
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument(
        "--receivers-per-room",
        type=int,
        default=0,
        help=(
            "Keep at most this many receivers per room, chosen by a seeded shuffle. "
            "0 keeps all. Sources are always kept in full, since the sampler draws "
            "1 + number_of_noises of them from a single receiver."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1,
                        help="Split the output across this many parquet files.")
    parser.add_argument("--subtype", default="PCM_16", choices=("PCM_16", "FLOAT"),
                        help="Source is PCM_16 at peak ~0.03, so re-encoding at the "
                             "same scale costs about one LSB and halves the size.")
    return parser.parse_args(argv)


def load_room_metadata(metadata_dir: Path, room_type: str, room: str) -> dict[tuple[int, int], dict[str, Any]]:
    """Index one room's JSON sidecars by (source, receiver) as integers."""
    room_dir = metadata_dir / room_type / room
    if not room_dir.is_dir():
        return {}

    index: dict[tuple[int, int], dict[str, Any]] = {}
    for path in room_dir.glob("*.json"):
        match = STEM.match(path.stem)
        if match is None:
            continue
        with path.open(encoding="utf-8") as handle:
            index[(int(match.group(1)), int(match.group(2)))] = json.load(handle)
    return index


def to_wav_bytes(samples: np.ndarray, sample_rate: int, subtype: str) -> bytes:
    """Never rescale per file -- that is what erases the relative level."""
    buffer = io.BytesIO()
    sf.write(buffer, samples.astype(np.float32), sample_rate, format="WAV", subtype=subtype)
    return buffer.getvalue()


def iter_rows(args: argparse.Namespace) -> Iterator[dict[str, Any]]:
    rng = np.random.default_rng(args.seed)
    archives = sorted(args.ir_dir.glob("*.zip"))
    if not archives:
        raise FileNotFoundError(f"No .zip archives under {args.ir_dir}")

    for archive in archives:
        room_type = archive.stem
        with zipfile.ZipFile(archive) as bundle:
            members: dict[str, list[tuple[str, int, int]]] = {}
            for name in bundle.namelist():
                if not name.endswith("_hybrid_IR.wav"):
                    continue
                parts = name.split("/")
                match = STEM.match(parts[-1])
                if match is None:
                    continue
                members.setdefault(parts[-2], []).append(
                    (name, int(match.group(1)), int(match.group(2)))
                )

            for room, entries in sorted(members.items()):
                metadata = load_room_metadata(args.metadata_dir, room_type, room)
                if not metadata:
                    print(f"  {room}: no metadata, skipped")
                    continue

                keep = None
                if args.receivers_per_room > 0:
                    receivers = sorted({receiver for _, _, receiver in entries})
                    rng.shuffle(receivers)
                    keep = set(receivers[: args.receivers_per_room])

                written = 0
                for name, source, receiver in sorted(entries):
                    if keep is not None and receiver not in keep:
                        continue
                    record = metadata.get((source, receiver))
                    if record is None:
                        continue

                    with bundle.open(name) as handle:
                        samples, source_rate = sf.read(io.BytesIO(handle.read()))
                    if source_rate != args.sample_rate:
                        samples = resample_poly(samples, args.sample_rate, source_rate)

                    distance = float(
                        np.linalg.norm(
                            np.asarray(record["src_loc"], dtype=np.float64)
                            - np.asarray(record["rec_loc"], dtype=np.float64)
                        )
                    )
                    # direct_index searches a window around the arrival time implied
                    # by distance, so distance must be known first.
                    direct = direct_index(samples, distance, args.sample_rate)
                    yield {
                        "audio": {
                            "bytes": to_wav_bytes(samples, args.sample_rate, args.subtype),
                            "path": f"{room}/S{source}_R{receiver}.wav",
                        },
                        "Room": room,
                        "Room Description": ROOM_SUFFIX.sub("", room),
                        "Receiver Label": f"R{receiver}",
                        "Direct Path Length [m]": distance,
                        "Avg C50": clarity_c50_db(samples, direct, args.sample_rate),
                        "Avg T30": reverberation_t30_s(samples, direct, args.sample_rate),
                    }
                    written += 1
                print(f"  {room}: {written} RIRs", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    dataset = Dataset.from_generator(
        iter_rows,
        features=FEATURES,
        gen_kwargs={"args": args},
    )
    # to_parquet takes no max_shard_size -- that is a save_to_disk/push_to_hub
    # argument. Shard explicitly instead.
    if args.shards <= 1:
        dataset.to_parquet(args.output)
    else:
        for index in range(args.shards):
            part = args.output.with_name(
                f"{args.output.stem}-{index:05d}-of-{args.shards:05d}.parquet"
            )
            dataset.shard(num_shards=args.shards, index=index).to_parquet(part)

    rooms = sorted(set(dataset["Room"]))
    print(f"\nwrote {args.output}  rows={len(dataset)}  rooms={len(rooms)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

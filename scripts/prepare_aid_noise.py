"""Assemble extracted AID recordings into dry mono stems, outside the DSP layer.

No downloads, training, RIR processing or SNR scaling happen here. Use one
microphone so simultaneous recordings are not treated as independent events.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from scripts.noise_eval_io import sha256, wav_bytes, write_parquet

AID_URL = "https://zenodo.org/records/6974033"
# Continuous/transient grouping from Table 2 of the AID paper. This is an
# assembly policy, not a claim that every instant of these recordings is steady.
CONTINUOUS = frozenset(
    "breath cloth consonant hairdryer paper pen_case plastic_bag plastic_bottle "
    "plastic_wrap roll_stool rubber_band vacuum vacuum_cleaner zipper blender drill "
    "keys whisk".split()
)


@dataclass
class Clip:
    name: str
    category: str
    digest: str
    audio: np.ndarray


def load_clips(root: Path, microphone: str, sample_rate: int) -> dict[str, list[Clip]]:
    import io

    groups: dict[str, list[Clip]] = {}
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() != ".wav" or "__MACOSX" in path.parts:
            continue
        if not path.stem.upper().endswith(f"_{microphone.upper()}"):
            continue
        # The archive includes maker prefixes omitted in the paper's shorthand:
        # cutlery_11_RHODE_NT1.wav / tape_measure_03_SH_MKH800.wav.
        match = re.fullmatch(r"(.+)_(\d+)_(?:RHODE_|SH_)?(NT1|NT5|MKH800)", path.stem, re.IGNORECASE)
        if match is None:
            raise ValueError(f"Unexpected AID filename: {path.name}")
        category = match.group(1).lower()
        data = path.read_bytes()
        audio, rate = sf.read(io.BytesIO(data), dtype="float64")
        if audio.ndim != 1 or audio.size < 4 or not np.isfinite(audio).all():
            raise ValueError(f"Expected finite mono audio: {path}")
        if rate != sample_rate:
            audio = resample_poly(audio, sample_rate, rate)
        if not np.any(audio):
            raise ValueError(f"Silent recording: {path}")
        groups.setdefault(category, []).append(
            Clip(path.relative_to(root).as_posix(), category, sha256(data), audio)
        )
    if not groups:
        raise ValueError(f"No *_index_{microphone}.wav recordings found in {root}")
    return groups


def assemble_stem(
    clips: Sequence[Clip],
    length: int,
    sample_rate: int,
    rng: np.random.Generator,
    kind: str,
    gap_seconds: tuple[float, float] = (0.25, 1.5),
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Sequence variants with gaps for transients, short crossfades otherwise.

    Preserve within-category recording levels. The existing mixer subsequently
    normalizes the complete stem and applies its unchanged received-SNR rule.
    """
    if not clips or length < 4 or sample_rate <= 0:
        raise ValueError("Need clips, positive sample rate and at least four samples")
    if kind not in ("continuous", "transient"):
        raise ValueError(f"Unknown noise kind: {kind}")
    if not 0 <= gap_seconds[0] <= gap_seconds[1] or not np.isfinite(gap_seconds).all():
        raise ValueError("Require finite 0 <= min gap <= max gap")
    output = np.zeros(length, dtype=np.float64)
    events = []
    start = 0
    previous = -1
    previous_fade = 0
    while start < length:
        choices = [i for i in range(len(clips)) if i != previous] or [previous]
        index = int(rng.choice(choices))
        clip = clips[index]
        signal = clip.audio.copy()
        fade = min(round(sample_rate * 0.005), len(signal) // 4)
        if kind == "continuous" and events:
            start -= min(previous_fade, fade)
        end = min(start + len(signal), length)
        signal = signal[: end - start]
        fade = min(fade, len(signal) // 4)
        # Soft edges also avoid artificial clicks when the final event is cut.
        if fade:
            ramp = np.linspace(0.0, 1.0, fade)
            signal[:fade] *= ramp
            signal[-fade:] *= ramp[::-1]
        output[start:end] += signal
        events.append({
            "source": clip.name, "source_sha256": clip.digest,
            "start_sample": start, "end_sample": end, "fade_samples": fade,
        })
        if end == length:
            break
        start = end
        if kind == "transient":
            start += round(float(rng.uniform(*gap_seconds)) * sample_rate)
        previous = index
        previous_fade = fade
    return output, events


def iter_stems(args: argparse.Namespace) -> Iterator[dict[str, Any]]:
    groups = load_clips(args.input_dir, args.microphone, args.sample_rate)
    categories = sorted(groups) if args.categories is None else sorted(set(args.categories))
    missing = set(categories) - groups.keys()
    if missing:
        raise ValueError(f"Unknown categories: {sorted(missing)}; available: {sorted(groups)}")
    print(f"Found {sum(map(len, groups.values()))} recordings; using {len(categories)} categories", flush=True)
    length = round(args.duration_seconds * args.sample_rate)
    for index in range(args.count):
        category = categories[index % len(categories)]
        kind = "continuous" if category in CONTINUOUS else "transient"
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, index]))
        audio, events = assemble_stem(
            groups[category], length, args.sample_rate, rng, kind,
            (args.min_gap_seconds, args.max_gap_seconds),
        )
        encoded = wav_bytes(audio, args.sample_rate)
        stem_id = f"aid-{args.microphone.lower()}-{args.seed}-{index:05d}-{category}"
        yield {
            "id": stem_id, "audio": {"bytes": encoded, "path": f"{stem_id}.wav"},
            "category": category, "noise_kind": kind,
            "duration": length / args.sample_rate, "sample_rate": args.sample_rate,
            "sha256": sha256(encoded), "seed": args.seed, "microphone": args.microphone,
            "events_json": json.dumps(events, sort_keys=True),
            "assembly_json": json.dumps({
                "version": 1, "gap_seconds": [args.min_gap_seconds, args.max_gap_seconds],
                "edge_fade_ms": 5, "category_sampling": "round_robin",
            }, sort_keys=True),
            "source_url": AID_URL, "license": "CC-BY-NC-SA-4.0",
            "license_note": "Archive LICENSE is BY-NC-SA-4.0; Zenodo metadata says BY-4.0. Retain the restrictive archive terms pending clarification.",
            "attribution": "Philipp Götz, Cagdas Tuna, Andreas Walther, Emanuël A. P. Habets (2022), AID",
        }
        if (index + 1) % 16 == 0:
            print(f"Assembled {index + 1}/{args.count} stems", flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Extracted official AID.zip")
    parser.add_argument("--output", type=Path, default=Path("data/aid_noise.parquet"))
    parser.add_argument("--microphone", choices=("NT1", "NT5", "MKH800"), default="NT1")
    parser.add_argument("--categories", nargs="+", help="Optional filename categories to include")
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--duration-seconds", type=float, default=60.0)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-gap-seconds", type=float, default=0.25)
    parser.add_argument("--max-gap-seconds", type=float, default=1.5)
    args = parser.parse_args(argv)
    if args.count <= 0 or args.sample_rate <= 0 or args.seed < 0:
        parser.error("count/sample-rate must be positive and seed nonnegative")
    if not np.isfinite(args.duration_seconds) or args.duration_seconds * args.sample_rate < 4:
        parser.error("duration must be finite and at least four samples")
    if not np.isfinite([args.min_gap_seconds, args.max_gap_seconds]).all() or not 0 <= args.min_gap_seconds <= args.max_gap_seconds:
        parser.error("require finite 0 <= min-gap-seconds <= max-gap-seconds")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    count = write_parquet(args.output, iter_stems(args))
    print(f"Wrote {count} dry stems to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

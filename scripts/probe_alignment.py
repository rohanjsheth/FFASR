"""Does RIR convolution shift speech in time, or only add a tail?

Every frame-level loss (encoder-state matching, CTC teacher-student) needs the
clean teacher and the reverberant student to line up on the time axis. Our
`convolve` trims each RIR to 1 ms before the direct arrival, which *should* mean
dry and reverberant share an onset and differ only by the tail -- but that has
never been measured, and a silent misalignment would poison a whole run.

This aligns both versions of the same utterance with Qwen3-ForcedAligner and
compares word timestamps. A constant near-zero offset means frame-level losses
need nothing but a length crop. Drift with room or distance means alignment has
to be a pipeline step.
"""

from __future__ import annotations

import argparse
import io
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import soundfile as sf

ALIGNER = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
TREBLE_MONO_PARQUET = (
    "https://huggingface.co/datasets/treble-technologies/Treble10-RIR/"
    "resolve/main/data/rir_mono-00000-of-00001.parquet"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speech-parquet", required=True)
    parser.add_argument("--rir-parquet", default=TREBLE_MONO_PARQUET)
    parser.add_argument("--utterances", type=int, default=200)
    parser.add_argument("--language", default="English")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"),
                        help="'auto' picks cuda, then mps, then cpu. bf16 on cuda, fp32 elsewhere.")
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf_cache"))
    parser.add_argument("--output", type=Path, default=Path("results/alignment_probe.json"))
    return parser.parse_args(argv)


def align(processor, model, audio, transcript, language):
    """Word timestamps for one utterance, or None if the aligner declines it."""
    import torch

    inputs, word_lists = processor.prepare_forced_aligner_inputs(
        audio=audio, transcript=transcript, language=language,
    )
    inputs = inputs.to(model.device, model.dtype)
    with torch.inference_mode():
        outputs = model(**inputs)
    stamps = processor.decode_forced_alignment(
        logits=outputs.logits,
        input_ids=inputs["input_ids"],
        word_lists=word_lists,
        timestamp_token_id=model.config.timestamp_token_id,
    )[0]
    return stamps or None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    import torch
    from datasets import Audio, load_dataset
    from transformers import AutoModelForTokenClassification, AutoProcessor

    from audio_utils.audio_mixing import convolve
    from data_utils.data_utils import read_and_resample

    cache = str(args.cache_dir)
    speech = load_dataset("parquet", data_files=args.speech_parquet, split="train",
                          cache_dir=cache).cast_column("audio", Audio(decode=False))
    rirs = load_dataset("parquet", data_files=args.rir_parquet, split="train",
                        cache_dir=cache).cast_column("audio", Audio(decode=False))
    print(f"speech={len(speech)}  rirs={len(rirs)}")

    device = args.device
    if device == "auto":
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    # bf16 is a cuda convenience; mps and cpu are more reliable in fp32 here.
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"aligner on {device} ({dtype})")

    processor = AutoProcessor.from_pretrained(ALIGNER, cache_dir=cache)
    model = AutoModelForTokenClassification.from_pretrained(
        ALIGNER, dtype=dtype, cache_dir=cache
    ).to(device)
    model.eval()

    rng = np.random.default_rng(args.seed)
    indices = [int(i) for i in rng.choice(len(speech), args.utterances, replace=False)]

    rows = []
    skipped = 0
    for count, index in enumerate(indices, start=1):
        record = speech[index]
        text = str(record.get("text") or record.get("text_normalized") or "").strip()
        if not text:
            skipped += 1
            continue
        dry = read_and_resample(record["audio"], args.sample_rate)

        rir_index = int(rng.integers(0, len(rirs)))
        rir_record = rirs[rir_index]
        rir = read_and_resample(rir_record["audio"], args.sample_rate)
        distance = float(rir_record["Direct Path Length [m]"])
        wet = convolve(dry, rir, distance, args.sample_rate)

        dry_stamps = align(processor, model, dry, text, args.language)
        wet_stamps = align(processor, model, wet, text, args.language)
        if not dry_stamps or not wet_stamps or len(dry_stamps) != len(wet_stamps):
            skipped += 1
            continue

        offsets = [w["start_time"] - d["start_time"] for d, w in zip(dry_stamps, wet_stamps)]
        rows.append({
            "speech_index": index,
            "room": str(rir_record.get("Room")),
            "distance_m": distance,
            "words": len(offsets),
            "dry_seconds": dry.size / args.sample_rate,
            "wet_seconds": wet.size / args.sample_rate,
            "offset_mean_s": float(np.mean(offsets)),
            "offset_max_abs_s": float(np.max(np.abs(offsets))),
            "offset_first_word_s": float(offsets[0]),
        })
        if count % 25 == 0:
            print(f"  {count}/{len(indices)} ({skipped} skipped)", flush=True)

    if not rows:
        raise RuntimeError("No utterance aligned in both conditions")

    mean = np.array([r["offset_mean_s"] for r in rows])
    worst = np.array([r["offset_max_abs_s"] for r in rows])
    dist = np.array([r["distance_m"] for r in rows])

    print(f"\naligned {len(rows)} utterances, skipped {skipped}")
    print(f"per-utterance mean offset: {1000*mean.mean():+.1f} ms  "
          f"sd {1000*mean.std():.1f} ms  range [{1000*mean.min():+.1f}, {1000*mean.max():+.1f}]")
    print(f"worst within-utterance offset: median {1000*np.median(worst):.1f} ms  "
          f"max {1000*worst.max():.1f} ms")
    if dist.std() > 0:
        print(f"correlation of offset with direct-path distance: "
              f"{np.corrcoef(dist, mean)[0, 1]:+.3f}")

    half_frame_s = 0.5 * 160 / args.sample_rate  # half a 10 ms encoder hop, in SECONDS
    within = float(np.mean(np.abs(mean) < half_frame_s))
    print(f"\n{100 * within:.0f}% of utterances within half an encoder frame "
          f"({1000 * half_frame_s:.0f} ms) -> {'crop is enough' if within > 0.9 else 'alignment needed'}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump({"args": {k: str(v) for k, v in vars(args).items()}, "rows": rows},
                  handle, indent=2)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

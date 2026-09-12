"""Score a model on Treble10-Speech, split by room and source-receiver distance.

Reverberation only, no added noise: LibriSpeech test-clean/test-other convolved
with the Treble10 RIRs by Treble themselves. That makes it the one far-field
eval here whose audio our renderer did not produce while still using the
leaderboard's own simulator and rooms, so it isolates the reverberation axis the
way NOIZEUS isolates the additive-noise one.

Held out for our models: training used AcousticRooms; Treble10 appeared only in
validation, and the run took the final checkpoint rather than best-by-metric, so
nothing selected on it.
"""

from __future__ import annotations

import argparse
import io
import json
from collections.abc import Sequence
from pathlib import Path

import soundfile as sf
import torch
from datasets import Audio, load_dataset
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

from eval_utils.text_norm import edit_distance, normalize_for_wer
from eval_utils.transcribe import transcribe_with_scores

MONO_SHARDS = "hf://datasets/treble-technologies/Treble10-Speech/data/speech_mono-*.parquet"
# T30/C50 are per octave band; 1 kHz is index 4 of [63,125,250,500,1000,2000,4000,8000].
BAND_1K = 4


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--shards", default=MONO_SHARDS)
    parser.add_argument("--language", default="English")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="0 scores everything.")
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf_cache"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = load_dataset(
        "parquet", data_files=args.shards, split="train", cache_dir=str(args.cache_dir)
    ).cast_column("audio", Audio(decode=False))
    if args.limit:
        dataset = dataset.select(range(args.limit))
    print(f"scoring {len(dataset)} clips with {args.model_id}", flush=True)

    processor = AutoProcessor.from_pretrained(args.model_id, cache_dir=str(args.cache_dir))
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        args.model_id, cache_dir=str(args.cache_dir), dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).eval().cuda()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    errors = words = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for start in range(0, len(dataset), args.batch_size):
            batch = dataset[start : start + args.batch_size]
            scenes = []
            for record in batch["audio"]:
                audio, rate = sf.read(io.BytesIO(record["bytes"]), dtype="float32")
                if rate != args.sample_rate:
                    raise SystemExit(f"expected {args.sample_rate} Hz, got {rate}")
                scenes.append({"audio": audio})

            results = transcribe_with_scores(
                model=model, processor=processor, scenes=scenes, language=args.language,
                sample_rate=args.sample_rate, max_new_tokens=args.max_new_tokens,
                device="cuda", model_dtype=torch.bfloat16,
            )

            n_rows = len(scenes)
            for i, result in enumerate(results):
                reference = normalize_for_wer(batch["transcript"][i]).split()
                hypothesis = normalize_for_wer(result["hypothesis"]).split()
                if not reference:
                    continue
                n = edit_distance(reference, hypothesis)
                errors += n
                words += len(reference)
                handle.write(json.dumps({
                    "filename": batch["filename"][i],
                    "room": batch["room"][i],
                    "room_description": batch["room_description"][i],
                    "room_volume": float(batch["room_volume"][i]),
                    "source": batch["source"][i],
                    "receiver": batch["receiver"][i],
                    "distance_m": float(batch["direct_path_length_m"][i]),
                    "t30_1k": float(batch["T30"][i][BAND_1K]),
                    "c50_1k": float(batch["C50"][i][BAND_1K]),
                    "librispeech_split": batch["librispeech_split"][i],
                    "librispeech_file": batch["librispeech_file"][i],
                    "reference": batch["transcript"][i],
                    "hypothesis": result["hypothesis"],
                    "word_errors": n,
                    "reference_words": len(reference),
                    "hypothesis_words": len(hypothesis),
                    "mean_entropy": result["mean_entropy"],
                    "avg_logprob": result["avg_logprob"],
                }) + "\n")

            done = start + n_rows
            if done % 400 < args.batch_size:
                print(f"{done}/{len(dataset)}  running WER={100 * errors / max(1, words):.2f}%", flush=True)

    print(f"{args.model_id}  WER={100 * errors / max(1, words):.2f}%  ({words} words)")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

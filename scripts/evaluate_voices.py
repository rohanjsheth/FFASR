"""Score a model on the VOiCES devkit test set, split by distractor, room and mic."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

from eval_utils.text_norm import edit_distance, normalize_for_wer
from eval_utils.transcribe import transcribe_with_scores


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--voices-root", type=Path, default=Path("/root/voices/VOiCES_devkit"))
    parser.add_argument("--language", default="English")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="0 scores the whole test set.")
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf_cache"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    index = args.voices_root / "references" / "test_index.csv"
    rows = list(csv.DictReader(index.open()))
    if args.limit:
        rows = rows[: args.limit]
    print(f"scoring {len(rows)} clips with {args.model_id}", flush=True)

    processor = AutoProcessor.from_pretrained(args.model_id, cache_dir=str(args.cache_dir))
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        args.model_id, cache_dir=str(args.cache_dir), dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).eval().cuda()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    errors = words = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            scenes = []
            for row in batch:
                audio, _ = sf.read(args.voices_root / row["filename"], dtype="float32")
                scenes.append({"audio": audio})

            results = transcribe_with_scores(
                model=model, processor=processor, scenes=scenes, language=args.language,
                sample_rate=args.sample_rate, max_new_tokens=args.max_new_tokens,
                device="cuda", model_dtype=torch.bfloat16,
            )

            for row, result in zip(batch, results, strict=True):
                reference = normalize_for_wer(row["transcript"]).split()
                hypothesis = normalize_for_wer(result["hypothesis"]).split()
                if not reference:
                    continue
                n = edit_distance(reference, hypothesis)
                errors += n
                words += len(reference)
                handle.write(json.dumps({
                    "query_name": row["query_name"],
                    "distractor": row["distractor"],
                    "room": row["room"],
                    "mic": int(row["mic"]),
                    "degrees": int(row["degrees"]),
                    "speaker": row["speaker"],
                    "source": row["source"],
                    "duration_s": float(row["noisy_time"]),
                    "reference": row["transcript"],
                    "hypothesis": result["hypothesis"],
                    "word_errors": n,
                    "reference_words": len(reference),
                    "hypothesis_words": len(hypothesis),
                    "mean_entropy": result["mean_entropy"],
                    "avg_logprob": result["avg_logprob"],
                }) + "\n")

            done = start + len(batch)
            if done % 400 < args.batch_size:
                print(f"{done}/{len(rows)}  running WER={100 * errors / max(1, words):.2f}%", flush=True)

    print(f"{args.model_id}  WER={100 * errors / max(1, words):.2f}%  ({words} words)")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Score a model on Voices-in-the-Wild-Bench, split by source, condition and language.

The benchmark released with Mega-ASR (arXiv 2605.19833, MIT licence). 5,000 clips
across seven single perturbations plus mixed, half Chinese and half English, and
split between their simulation pipeline (`sim-`) and real recordings (`real-`).
The `subset` field carries all three axes, e.g. `sim-en-far_field`.

English only by default: our models are English-only, and the benchmark scores
Chinese with CER rather than WER.
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
from scipy.signal import resample_poly
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

from eval_utils.text_norm import edit_distance, normalize_for_wer
from eval_utils.transcribe import transcribe_with_scores

DATASET = "zhifeixie/Voices-in-the-Wild-Bench"
CONDITIONS = ("noise", "far_field", "obstructed", "distortion", "recording", "echo",
              "dropout", "mixed")
SPLITS = tuple(f"{source}_{condition}" for source in ("real", "syn") for condition in CONDITIONS)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--splits", nargs="*", default=list(SPLITS))
    parser.add_argument("--language", default="en", help="Filter on the subset field; 'all' keeps both.")
    parser.add_argument("--prompt-language", default="English")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf_cache"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    processor = AutoProcessor.from_pretrained(args.model_id, cache_dir=str(args.cache_dir))
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        args.model_id, cache_dir=str(args.cache_dir), dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).eval().cuda()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    errors = words = kept = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for split in args.splits:
            dataset = load_dataset(DATASET, split=split, cache_dir=str(args.cache_dir))
            dataset = dataset.cast_column("audio", Audio(decode=False))
            if args.language != "all":
                dataset = dataset.filter(lambda r: f"-{args.language}-" in r["subset"])
            if not len(dataset):
                print(f"{split}: no {args.language} rows, skipping", flush=True)
                continue
            print(f"{split}: {len(dataset)} rows", flush=True)

            for start in range(0, len(dataset), args.batch_size):
                batch = dataset[start : start + args.batch_size]
                scenes = []
                for record in batch["audio"]:
                    audio, rate = sf.read(io.BytesIO(record["bytes"]), dtype="float32")
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)
                    if rate != args.sample_rate:
                        audio = resample_poly(audio, args.sample_rate, rate).astype("float32")
                    scenes.append({"audio": audio})

                results = transcribe_with_scores(
                    model=model, processor=processor, scenes=scenes,
                    language=args.prompt_language, sample_rate=args.sample_rate,
                    max_new_tokens=args.max_new_tokens, device="cuda",
                    model_dtype=torch.bfloat16,
                )

                for i, result in enumerate(results):
                    reference = normalize_for_wer(batch["answer"][i]).split()
                    hypothesis = normalize_for_wer(result["hypothesis"]).split()
                    if not reference:
                        continue
                    subset = batch["subset"][i]
                    source, language, condition = subset.split("-", 2)
                    n = edit_distance(reference, hypothesis)
                    errors += n
                    words += len(reference)
                    kept += 1
                    handle.write(json.dumps({
                        "name": batch["name"][i],
                        "split": split,
                        "subset": subset,
                        "source": source,
                        "language": language,
                        "condition": condition,
                        "reference": batch["answer"][i],
                        "hypothesis": result["hypothesis"],
                        "word_errors": n,
                        "reference_words": len(reference),
                        "hypothesis_words": len(hypothesis),
                        "mean_entropy": result["mean_entropy"],
                        "avg_logprob": result["avg_logprob"],
                    }) + "\n")

            print(f"  running WER={100 * errors / max(1, words):.2f}%  ({kept} clips)", flush=True)

    print(f"{args.model_id}  WER={100 * errors / max(1, words):.2f}%  ({words} words, {kept} clips)")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

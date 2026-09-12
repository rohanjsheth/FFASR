"""Score a model on the CHiME-4 isolated 1-channel track, split by arm and environment.

Read WSJ prompts in four real noisy environments -- bus, cafe, pedestrian area,
street -- captured on a 6-mic tablet, plus a simulated arm built by mixing clean
WSJ0 with the same recorded backgrounds. Single talker plus real noise, which is
the closest public benchmark to FFASR's scene composition, and the third column
of Mega-ASR's Table 2 (arXiv 2605.19833).

Licensing: the `_real` sets are CC BY-NC-SA 2.0 and citable for non-commercial
research. The `_simu` sets derive from WSJ0, which is LDC-licensed -- score them
only if you hold that licence.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

from eval_utils.text_norm import edit_distance, normalize_for_wer
from eval_utils.transcribe import transcribe_with_scores

ENVIRONMENTS = ("bus", "caf", "ped", "str")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--root", type=Path, default=Path("/root/chime4/CHiME4/data"))
    parser.add_argument("--set", default="et05", choices=("et05", "dt05"))
    parser.add_argument("--arms", nargs="*", default=["real", "simu"])
    parser.add_argument("--language", default="English")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf_cache"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def collect(root: Path, split: str, arm: str, environment: str) -> list[dict[str, str]]:
    audio_dir = root / "audio" / "16kHz" / "isolated_1ch_track" / f"{split}_{environment}_{arm}"
    text_dir = root / "transcriptions" / f"{split}_{environment}_{arm}"
    items = []
    for wav in sorted(audio_dir.glob("*.wav")):
        transcript = text_dir / f"{wav.stem}.trn"
        if not transcript.exists():
            raise SystemExit(f"no .trn for {wav}")
        # "<uttid> WORDS WORDS WORDS"
        _, _, text = transcript.read_text(encoding="utf-8").strip().partition(" ")
        items.append({"utterance_id": wav.stem, "wav": str(wav), "arm": arm,
                      "environment": environment, "reference": text})
    return items


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows: list[dict[str, str]] = []
    for arm in args.arms:
        for environment in ENVIRONMENTS:
            rows += collect(args.root, args.set, arm, environment)
    print(f"scoring {len(rows)} clips ({args.set}, arms={args.arms}) with {args.model_id}", flush=True)

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
                audio, rate = sf.read(row["wav"], dtype="float32")
                if rate != args.sample_rate:
                    raise SystemExit(f"{row['utterance_id']} is {rate} Hz")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                scenes.append({"audio": audio})

            results = transcribe_with_scores(
                model=model, processor=processor, scenes=scenes, language=args.language,
                sample_rate=args.sample_rate, max_new_tokens=args.max_new_tokens,
                device="cuda", model_dtype=torch.bfloat16,
            )

            for row, result in zip(batch, results, strict=True):
                reference = normalize_for_wer(row["reference"]).split()
                hypothesis = normalize_for_wer(result["hypothesis"]).split()
                if not reference:
                    continue
                n = edit_distance(reference, hypothesis)
                errors += n
                words += len(reference)
                handle.write(json.dumps({
                    "utterance_id": row["utterance_id"],
                    "arm": row["arm"],
                    "environment": row["environment"],
                    "reference": row["reference"],
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

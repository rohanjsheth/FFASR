"""Score a model on NOIZEUS, split by noise type and SNR.

30 IEEE sentences x 8 recorded noise types x 4 SNRs = 960 clips. Additive noise
on a clean near-field channel, so this isolates noise robustness from the
reverberation our renderer also trains. Mega-ASR (arXiv 2605.19833, Table 2)
reports Qwen3-ASR at 23.97 / 8.47 / 3.41 / 1.96 for 0/5/10/15 dB.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path

import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

from eval_utils.text_norm import edit_distance, normalize_for_wer
from eval_utils.transcribe import transcribe_with_scores

# sp01_airport_sn0 -> sentence sp01, noise airport, snr 0
INDEX_RE = re.compile(r"^(?P<sentence>sp\d+)_(?P<noise>.+)_sn(?P<snr>\d+)$")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--manifest", type=Path, default=Path("/root/noizeus/noizeus.jsonl"))
    parser.add_argument("--audio-root", type=Path, default=Path("/root/noizeus"))
    parser.add_argument("--language", default="English")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf_cache"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def index_audio(root: Path) -> dict[str, Path]:
    """The manifest carries the authors' absolute paths, so match on basename."""
    return {p.name: p for p in root.rglob("*.wav")}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = [json.loads(line) for line in args.manifest.open()]
    by_name = index_audio(args.audio_root)
    print(f"manifest rows={len(rows)}  wav files found={len(by_name)}", flush=True)

    missing = [r for r in rows if Path(r["audio_path"]).name not in by_name]
    if missing:
        raise SystemExit(f"{len(missing)} manifest rows have no wav, e.g. {missing[0]['audio_path']}")

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
                audio, rate = sf.read(by_name[Path(row["audio_path"]).name], dtype="float32")
                # NOIZEUS ships at 8 kHz; the processor wants 16 kHz.
                if rate != args.sample_rate:
                    audio = resample_poly(audio, args.sample_rate, rate).astype("float32")
                scenes.append({"audio": audio})

            results = transcribe_with_scores(
                model=model, processor=processor, scenes=scenes, language=args.language,
                sample_rate=args.sample_rate, max_new_tokens=args.max_new_tokens,
                device="cuda", model_dtype=torch.bfloat16,
            )

            for row, result in zip(batch, results, strict=True):
                reference = normalize_for_wer(row["answer"]).split()
                hypothesis = normalize_for_wer(result["hypothesis"]).split()
                if not reference:
                    continue
                match = INDEX_RE.match(row["index"])
                if match is None:
                    raise SystemExit(f"unparsable index {row['index']!r}")
                n = edit_distance(reference, hypothesis)
                errors += n
                words += len(reference)
                handle.write(json.dumps({
                    "index": row["index"],
                    "sentence": match["sentence"],
                    "noise": match["noise"],
                    "snr_db": int(match["snr"]),
                    "reference": row["answer"],
                    "hypothesis": result["hypothesis"],
                    "word_errors": n,
                    "reference_words": len(reference),
                    "hypothesis_words": len(hypothesis),
                    "mean_entropy": result["mean_entropy"],
                    "avg_logprob": result["avg_logprob"],
                }) + "\n")

            done = start + len(batch)
            if done % 240 < args.batch_size:
                print(f"{done}/{len(rows)}  running WER={100 * errors / max(1, words):.2f}%", flush=True)

    print(f"{args.model_id}  WER={100 * errors / max(1, words):.2f}%  ({words} words)")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

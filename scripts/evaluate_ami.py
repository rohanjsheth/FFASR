"""Score one model on the frozen paired AMI headset/distant corpus.

Straight pass-through: the frozen WAV bytes go to the processor unchanged. No
RIR is convolved, no noise is mixed and no SNR is targeted, so whatever
separates the two microphone conditions is the real room rather than our
simulator. Run this once per model over the same parquet; the per-clip SHA-256
is re-checked here so the two runs are known to have scored identical samples.

Results are broken out by microphone, by meeting and by cross-talk bucket.
The last one matters: overlapping speech is the degradation a headset does not
share with a distant microphone, so a distant-microphone regression confined to
the overlapped subset is a different claim from a uniform one.
"""

from __future__ import annotations

import argparse
import io
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval_utils.text_norm import edit_distance, normalize_for_wer
from eval_utils.transcribe import transcribe_with_scores

DEFAULT_MODEL_ID = "Qwen/Qwen3-ASR-1.7B-hf"
REQUIRED_COLUMNS = frozenset(
    {
        "id", "pair_id", "pair_index", "microphone", "audio_id", "audio", "text",
        "meeting_id", "speaker_id", "begin_time", "end_time", "duration_seconds",
        "overlap_ratio", "sample_rate", "sha256",
    }
)
# Boundaries chosen so the first bucket is genuinely single-speaker audio and the
# last is speech the distant microphone cannot have captured cleanly.
OVERLAP_BUCKETS = ((0.0, "none"), (0.2, "slight"), (0.5, "moderate"), (1.01, "heavy"))


@dataclass(slots=True)
class WERAccumulator:
    examples: int = 0
    word_errors: int = 0
    reference_words: int = 0
    audio_seconds: float = 0.0
    catastrophic: int = 0
    with_punctuation: int = 0
    hypothesis_words: int = 0
    entropy_sum: float = 0.0

    @property
    def wer(self) -> float:
        if self.reference_words == 0:
            raise ValueError("Cannot compute WER without reference words")
        return self.word_errors / self.reference_words

    def add(
        self, reference: str, hypothesis: str, seconds: float, mean_entropy: float = 0.0
    ) -> tuple[int, int, str, str]:
        normalized_reference = normalize_for_wer(reference)
        normalized_hypothesis = normalize_for_wer(hypothesis)
        reference_tokens = normalized_reference.split()
        if not reference_tokens:
            raise ValueError(f"Reference became empty after normalization: {reference!r}")
        hypothesis_tokens = normalized_hypothesis.split()
        errors = edit_distance(reference_tokens, hypothesis_tokens)
        self.examples += 1
        self.word_errors += errors
        self.reference_words += len(reference_tokens)
        self.audio_seconds += seconds
        self.catastrophic += errors > len(reference_tokens)
        self.with_punctuation += any(c in hypothesis for c in ".,?!;:")
        self.hypothesis_words += len(hypothesis_tokens)
        self.entropy_sum += mean_entropy
        return errors, len(reference_tokens), normalized_reference, normalized_hypothesis


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=None, help="Pin the checkpoint revision.")
    parser.add_argument(
        "--ami-parquet",
        required=True,
        help="Frozen corpus from scripts.prepare_ami_eval.",
    )
    parser.add_argument("--language", default="English")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/ami"))
    return parser.parse_args(argv)


def overlap_bucket(ratio: float) -> str:
    for upper, name in OVERLAP_BUCKETS:
        if ratio <= upper:
            return name
    raise ValueError(f"Overlap ratio outside [0, 1]: {ratio}")


def load_model(args: argparse.Namespace) -> tuple[Any, Any, Any, Any]:
    import torch
    from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("AMI evaluation requires a CUDA GPU")
    print(f"Loading processor and model {args.model_id!r}...")
    common = {"cache_dir": str(args.cache_dir)}
    if args.revision is not None:
        common["revision"] = args.revision
    processor = AutoProcessor.from_pretrained(args.model_id, **common)
    if int(processor.feature_extractor.sampling_rate) != args.sample_rate:
        raise ValueError("Corpus rate does not match processor rate")
    device = torch.device("cuda")
    model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        args.model_id, dtype=model_dtype, low_cpu_mem_usage=True, **common
    ).to(device)
    model.eval()
    print(f"CUDA device: {torch.cuda.get_device_name(device)}, dtype={model_dtype}, "
          f"batch_size={args.batch_size}")
    return model, processor, device, model_dtype


def batch_scenes(rows: Sequence[dict[str, Any]], sample_rate: int) -> list[dict[str, Any]]:
    import numpy as np
    import soundfile as sf

    from scripts.noise_eval_io import sha256

    scenes = []
    for row in rows:
        encoded = row["audio"]["bytes"]
        if sha256(encoded) != row["sha256"]:
            raise ValueError(f"Audio checksum mismatch for {row['id']}")
        audio, rate = sf.read(io.BytesIO(encoded), dtype="float64")
        if rate != sample_rate or row["sample_rate"] != sample_rate:
            raise ValueError(f"Rate mismatch for {row['id']}; no resampling allowed")
        if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
            raise ValueError(f"Invalid audio for {row['id']}")
        scenes.append({"audio": audio, "text": row["text"], "metadata": {}})
    return scenes


def report_table(title: str, scores: dict[str, WERAccumulator], keys: Sequence[str]) -> None:
    print(f"\n{title}")
    print("group                examples   errors/words        WER")
    for key in keys:
        score = scores.get(key)
        if score is None or not score.reference_words:
            continue
        print(f"{key:<20} {score.examples:>8}   "
              f"{score.word_errors:>6}/{score.reference_words:<8}  {100 * score.wer:>6.2f}%")


def main(argv: Sequence[str] | None = None) -> int:
    from datasets import load_dataset

    args = parse_args(argv)
    predictions_path = args.output_dir / "predictions.jsonl"
    summary_path = args.output_dir / "summary.json"
    if predictions_path.exists() or summary_path.exists():
        raise FileExistsError(f"Choose a fresh --output-dir: {args.output_dir}")

    dataset = load_dataset(
        "parquet", data_files=args.ami_parquet, split="train",
        cache_dir=str(args.cache_dir),
    )
    missing = REQUIRED_COLUMNS - set(dataset.column_names)
    if missing or not len(dataset):
        raise ValueError(f"Expected output from scripts.prepare_ami_eval; missing {missing}")
    if len(set(dataset["id"])) != len(dataset):
        raise ValueError("Duplicate corpus IDs")

    # Longest-first inside each microphone keeps padded batches cheap and makes
    # any out-of-memory failure happen in the first batch rather than halfway in.
    order = sorted(
        range(len(dataset)),
        key=lambda i: (dataset[i]["microphone"], -dataset[i]["duration_seconds"], dataset[i]["id"]),
    )
    print(f"Scoring {len(dataset)} clips "
          f"({len(set(dataset['pair_id']))} pairs x {len(set(dataset['microphone']))} microphones)")

    model, processor, device, model_dtype = load_model(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    overall: dict[str, WERAccumulator] = {}
    by_meeting: dict[str, WERAccumulator] = {}
    by_overlap: dict[str, WERAccumulator] = {}
    with predictions_path.open("x", encoding="utf-8") as output:
        for start in range(0, len(order), args.batch_size):
            rows = [dataset[i] for i in order[start:start + args.batch_size]]
            scored = transcribe_with_scores(
                model=model, processor=processor,
                scenes=batch_scenes(rows, args.sample_rate),
                language=args.language, sample_rate=args.sample_rate,
                max_new_tokens=args.max_new_tokens, device=device, model_dtype=model_dtype,
            )
            for row, result in zip(rows, scored, strict=True):
                hypothesis = result["hypothesis"]
                microphone = row["microphone"]
                bucket = overlap_bucket(float(row["overlap_ratio"]))
                seconds = float(row["duration_seconds"])
                entropy = result["mean_entropy"]
                errors, words, reference, normalized = overall.setdefault(
                    microphone, WERAccumulator()
                ).add(row["text"], hypothesis, seconds, entropy)
                by_meeting.setdefault(f"{microphone}/{row['meeting_id']}", WERAccumulator()).add(
                    row["text"], hypothesis, seconds, entropy
                )
                by_overlap.setdefault(f"{microphone}/{bucket}", WERAccumulator()).add(
                    row["text"], hypothesis, seconds, entropy
                )
                output.write(json.dumps({
                    "id": row["id"], "pair_id": row["pair_id"],
                    "microphone": microphone, "audio_id": row["audio_id"],
                    "audio_sha256": row["sha256"], "meeting_id": row["meeting_id"],
                    "speaker_id": row["speaker_id"], "overlap_ratio": row["overlap_ratio"],
                    "overlap_bucket": bucket, "duration_seconds": seconds,
                    "reference": row["text"], "hypothesis": hypothesis,
                    "normalized_reference": reference, "normalized_hypothesis": normalized,
                    "word_errors": errors, "reference_words": words,
                    **{k: result[k] for k in
                       ("tokens", "avg_logprob", "min_logprob", "mean_entropy", "max_entropy")},
                }, ensure_ascii=False) + "\n")
            output.flush()
            print(f"Scored {min(start + len(rows), len(order))}/{len(order)}", flush=True)

    report_table("WER by microphone", overall, sorted(overall))
    report_table("WER by cross-talk", by_overlap, sorted(by_overlap))
    report_table("WER by meeting", by_meeting, sorted(by_meeting))
    print("\nDrift (invisible to WER)")
    print("mic     catastrophic    punct   len/ref   entropy")
    for key in sorted(overall):
        s = overall[key]
        print(f"{key:<8}{s.catastrophic:>6}/{s.examples:<6}"
              f"{100 * s.with_punctuation / s.examples:>6.0f}%"
              f"{s.hypothesis_words / s.reference_words:>9.3f}"
              f"{s.entropy_sum / s.examples:>10.3f}")
    if len(overall) == 2:
        near, far = sorted(overall)
        print(f"\n{far} minus {near}: "
              f"{100 * (overall[far].wer - overall[near].wer):+.2f} pp")

    def serialize(scores: dict[str, WERAccumulator]) -> dict[str, Any]:
        return {
            key: {
                "examples": score.examples, "word_errors": score.word_errors,
                "reference_words": score.reference_words, "wer": score.wer,
                "audio_seconds": score.audio_seconds,
                "catastrophic": score.catastrophic,
                "catastrophic_rate": score.catastrophic / score.examples,
                "punctuation_rate": score.with_punctuation / score.examples,
                "length_ratio": score.hypothesis_words / score.reference_words,
                "mean_entropy": score.entropy_sum / score.examples,
            }
            for key, score in sorted(scores.items())
        }

    summary = {
        "model_id": args.model_id, "revision": args.revision,
        "ami_parquet": args.ami_parquet, "dataset_fingerprint": dataset._fingerprint,
        "clips": len(dataset), "pairs": len(set(dataset["pair_id"])),
        "sample_rate": args.sample_rate, "language": args.language,
        "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens,
        "by_microphone": serialize(overall), "by_overlap": serialize(by_overlap),
        "by_meeting": serialize(by_meeting),
    }
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"\nwrote {predictions_path} and {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

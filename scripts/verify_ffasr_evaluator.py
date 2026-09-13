"""Run the FFASR submission module the way the leaderboard will, and diff it.

The leaderboard calls `evaluate(file)` once per clip; every WER in this campaign
came from batched `transcribe_with_scores`. Padding and bf16 reduction order
differ between the two, so the numbers we would quote on a submission are only
predictive if the per-file path reproduces them. This imports the submission
module unmodified, replays already-scored audio through it one file at a time,
and reports exact-string agreement plus the WER computed both ways.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections.abc import Sequence
from pathlib import Path

from eval_utils.text_norm import edit_distance, normalize_for_wer


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator", type=Path, default=Path("ffasr_submission/evaluate.py"))
    parser.add_argument("--recorded", type=Path, required=True,
                        help="The jsonl this campaign already scored.")
    parser.add_argument("--voices-root", type=Path, default=Path("/root/voices/VOiCES_devkit"))
    parser.add_argument("--limit", type=int, default=400, help="0 replays everything.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def load_evaluator(path: Path):
    spec = importlib.util.spec_from_file_location("ffasr_submission_evaluate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    recorded = [json.loads(line) for line in args.recorded.open()]
    index = {row["query_name"]: row["filename"]
             for row in csv.DictReader((args.voices_root / "references" / "test_index.csv").open())}

    # Stride rather than head, so a subset still spans every distractor/room/mic.
    if args.limit and args.limit < len(recorded):
        stride = len(recorded) // args.limit
        recorded = recorded[::stride][: args.limit]
    print(f"replaying {len(recorded)} clips through {args.evaluator}", flush=True)

    module = load_evaluator(args.evaluator)
    print(f"loaded {module.MODEL_ID} @ {module.REVISION}", flush=True)
    print(f"parameters: {sum(p.numel() for p in module.model.parameters()):,}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    identical = new_errors = old_errors = words = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for i, row in enumerate(recorded):
            hypothesis = module.evaluate(args.voices_root / index[row["query_name"]])
            reference = normalize_for_wer(row["reference"]).split()
            new = normalize_for_wer(hypothesis).split()
            n = edit_distance(reference, new)
            identical += hypothesis == row["hypothesis"]
            new_errors += n
            old_errors += row["word_errors"]
            words += len(reference)
            handle.write(json.dumps({
                "query_name": row["query_name"],
                "distractor": row["distractor"],
                "room": row["room"],
                "mic": row["mic"],
                "reference": row["reference"],
                "recorded_hypothesis": row["hypothesis"],
                "evaluator_hypothesis": hypothesis,
                "identical": hypothesis == row["hypothesis"],
                "recorded_word_errors": row["word_errors"],
                "evaluator_word_errors": n,
                "reference_words": len(reference),
            }) + "\n")

            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(recorded)}  identical={100 * identical / (i + 1):.1f}%  "
                      f"evaluator={100 * new_errors / max(1, words):.2f}%  "
                      f"recorded={100 * old_errors / max(1, words):.2f}%", flush=True)

    print(f"identical strings: {identical}/{len(recorded)} ({100 * identical / len(recorded):.2f}%)")
    print(f"evaluator WER={100 * new_errors / max(1, words):.4f}%  "
          f"recorded WER={100 * old_errors / max(1, words):.4f}%  "
          f"delta={100 * (new_errors - old_errors) / max(1, words):+.4f} pp  ({words} words)")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

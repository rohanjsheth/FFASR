"""Is the model confident when it confabulates?

The low-SNR failure is fluent, confident, wrong text -- correct for a few words,
then drifting into plausible filler. Two questions follow from that, and one
measurement answers both.

  Separability: does average token logprob distinguish the catastrophic
      utterances from the rest? If it does, a Whisper-style threshold plus
      fallback fixes most of the damage at inference, with no retraining.
  Exploration: how peaked is the token distribution on those utterances? A
      GRPO-family method computes its advantage across sampled rollouts, so a
      near-deterministic policy returns identical rollouts, zero variance and
      zero gradient -- precisely on the utterances that matter most. Mean
      per-token entropy says whether sampling could surface a hedged candidate.

Renders only the low band, live, using the same recipe sampler and retry loop as
the offline eval so the scenes match what `evaluate_snr_wer` would have built.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from eval_utils.text_norm import edit_distance, normalize_for_wer
from eval_utils.transcribe import transcribe_with_scores
from scripts.evaluate_snr_wer import (
    TREBLE_MONO_PARQUET,
    load_model,
    render_in_band,
)

DEFAULT_MODEL_ID = "Qwen/Qwen3-ASR-1.7B-hf"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--speech-parquet", required=True)
    parser.add_argument("--noise-parquet", default=None)
    parser.add_argument("--rir-parquet", default=TREBLE_MONO_PARQUET)
    parser.add_argument("--band", default="low", choices=("high", "mid", "low"))
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--number-of-noises", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--language", default="English")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-render-attempts", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf_cache"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)

    import numpy as np
    from datasets import Audio, load_dataset

    from data_utils.data_utils import get_groups, sample_scene_recipe

    cache_dir = str(args.cache_dir)
    speech_ds = load_dataset("parquet", data_files=args.speech_parquet, split="train",
                             cache_dir=cache_dir).cast_column("audio", Audio(decode=False))
    noise_ds = (
        load_dataset("parquet", data_files=args.noise_parquet, split="train", cache_dir=cache_dir)
        if args.noise_parquet else
        load_dataset("bilguun/musan-noise", split="train", cache_dir=cache_dir)
    ).cast_column("audio", Audio(decode=False))
    rir_ds = load_dataset("parquet", data_files=args.rir_parquet, split="train",
                          cache_dir=cache_dir).cast_column("audio", Audio(decode=False))
    print(f"speech={len(speech_ds)} noise={len(noise_ds)} rir={len(rir_ds)}")

    groups = get_groups(rir_ds)
    selection_rng = np.random.default_rng(args.seed)
    indices = [int(i) for i in selection_rng.choice(len(speech_ds), args.samples, replace=False)]

    model, processor, device, model_dtype = load_model(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    errors = words = 0
    with args.output.open("x", encoding="utf-8") as handle:
        for start in range(0, len(indices), args.batch_size):
            batch = indices[start:start + args.batch_size]
            scenes, recipes = [], []
            for speech_index in batch:
                recipe_rng = np.random.default_rng(
                    np.random.SeedSequence([args.seed, speech_index, 0])
                )
                base = sample_scene_recipe(
                    speech_index=speech_index, noise_ds=noise_ds, groups=groups,
                    rng=recipe_rng, number_of_noises=args.number_of_noises,
                )
                target_rng = np.random.default_rng(
                    np.random.SeedSequence([args.seed, speech_index, 1])
                )
                recipe, scene, _ = render_in_band(
                    base_recipe=base, requested_band=args.band, target_rng=target_rng,
                    speech_ds=speech_ds, noise_ds=noise_ds, rir_ds=rir_ds,
                    sample_rate=args.sample_rate, max_attempts=args.max_render_attempts,
                )
                scenes.append(scene)
                recipes.append(recipe)

            results = transcribe_with_scores(
                model=model, processor=processor, scenes=scenes, language=args.language,
                sample_rate=args.sample_rate, max_new_tokens=args.max_new_tokens,
                device=device, model_dtype=model_dtype,
            )
            for speech_index, scene, recipe, result in zip(batch, scenes, recipes, results, strict=True):
                reference = normalize_for_wer(scene["text"])
                hypothesis = normalize_for_wer(result["hypothesis"])
                reference_tokens = reference.split()
                if not reference_tokens:
                    continue
                utterance_errors = edit_distance(reference_tokens, hypothesis.split())
                errors += utterance_errors
                words += len(reference_tokens)
                handle.write(json.dumps({
                    "speech_index": speech_index,
                    "speech_id": str(speech_ds[speech_index].get("id", speech_index)),
                    "band": args.band,
                    "final_snr_db": float(scene["metadata"]["final_snr_db"]),
                    "room": recipe.room,
                    "reference": scene["text"],
                    "hypothesis": result["hypothesis"],
                    "normalized_reference": reference,
                    "normalized_hypothesis": hypothesis,
                    "word_errors": utterance_errors,
                    "reference_words": len(reference_tokens),
                    "recipe": asdict(recipe),
                    **{k: result[k] for k in
                       ("tokens", "avg_logprob", "min_logprob", "mean_entropy", "max_entropy")},
                }, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{min(start + len(batch), len(indices))}/{len(indices)}  "
                  f"running WER={100 * errors / max(1, words):.2f}%", flush=True)

    print(f"\n{args.model_id}  {args.band} band  WER={100 * errors / words:.2f}%")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

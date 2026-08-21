"""Measure WER across a fine grid of SNR bins to locate the intelligibility floor.

evaluate_snr_wer scores three wide bands, which is the right shape for reporting
but too coarse to answer where audio stops carrying recoverable content. This
sweeps contiguous narrow bins so the curve itself is the output: run it on the
base model to choose a training SNR floor from evidence rather than assumption.

Scenes are seeded from (seed, speech_index, bin_index) alone, so two models swept
with the same seed see byte-identical audio and their per-utterance predictions
can be compared as a paired sample.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from data_utils.room_folds import FOLDS
from eval_utils.transcribe import transcribe_batch
from scripts.evaluate_snr_wer import (
    DEFAULT_MODEL_ID,
    LIBRISPEECH_TEST_CLEAN_PARQUET,
    TREBLE_MONO_PARQUET,
    WERAccumulator,
)

if TYPE_CHECKING:
    import numpy as np
    from datasets import Dataset

    from data_utils.data_utils import RenderedScene, SceneRecipe


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--language", default="English")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--min-snr", type=float, default=-8.0)
    parser.add_argument("--max-snr", type=float, default=6.0)
    parser.add_argument("--bin-width", type=float, default=2.0)
    parser.add_argument("--samples-per-bin", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--number-of-noises", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument(
        "--fold",
        type=int,
        choices=tuple(FOLDS),
        default=None,
        help=(
            "Restrict rooms to the validation half of a training fold. Omit to "
            "sweep all ten rooms, which is what a pretrained baseline wants."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/snr_sweep"))
    parser.add_argument(
        "--max-render-attempts",
        type=int,
        default=20,
        help="Target-SNR draws allowed before giving up on landing inside a bin.",
    )
    return parser.parse_args(argv)


def build_bins(minimum: float, maximum: float, width: float) -> list[tuple[float, float]]:
    """Contiguous [lower, upper) bins covering the requested range.

    A trailing partial bin is kept rather than dropped -- a truncated bin still
    reports a valid WER, and silently discarding the hardest slice of the range
    would defeat the purpose of the sweep.
    """
    if width <= 0:
        raise ValueError("--bin-width must be positive")
    if maximum <= minimum:
        raise ValueError("--max-snr must exceed --min-snr")

    bins: list[tuple[float, float]] = []
    lower = minimum
    while lower < maximum - 1e-9:
        bins.append((lower, min(lower + width, maximum)))
        lower += width
    return bins


def render_in_bin(
    base_recipe: SceneRecipe,
    bounds: tuple[float, float],
    target_rng: np.random.Generator,
    speech_ds: Dataset,
    noise_ds: Dataset,
    rir_ds: Dataset,
    sample_rate: int,
    max_attempts: int,
) -> tuple[SceneRecipe, RenderedScene, int]:
    """Render until the achieved SNR lands inside the bin, or give up."""
    from data_utils.data_utils import render_scene_from_recipe

    lower, upper = bounds
    rejected = 0

    for _ in range(max_attempts):
        recipe = replace(
            base_recipe,
            target_snr_db=float(target_rng.uniform(lower, upper)),
        )
        scene = render_scene_from_recipe(
            recipe=recipe,
            speech_ds=speech_ds,
            noise_ds=noise_ds,
            rir_ds=rir_ds,
            sr=sample_rate,
        )
        realized = float(scene["metadata"]["final_snr_db"])
        if lower <= realized < upper:
            return recipe, scene, rejected
        rejected += 1

    raise RuntimeError(
        f"Could not land a scene in [{lower}, {upper}) after {max_attempts} "
        "attempts; widen --bin-width or raise --max-render-attempts"
    )


def print_curve(
    bins: Sequence[tuple[float, float]],
    scores: dict[int, WERAccumulator],
) -> None:
    print("\nWER by SNR bin:")
    print("  bin (dB)       n   realized SNR       errors/words      WER")
    for index, (lower, upper) in enumerate(bins):
        score = scores[index]
        if score.examples == 0:
            continue
        mean_snr = sum(score.snr_values) / len(score.snr_values)
        print(
            f"  [{lower:>5.1f},{upper:>5.1f})  {score.examples:>5}  "
            f"{mean_snr:>6.2f}  "
            f"{score.word_errors:>7}/{score.reference_words:<7}  "
            f"{100 * score.wer:>6.2f}%"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.samples_per_bin <= 0:
        raise ValueError("--samples-per-bin must be positive")
    if args.max_render_attempts <= 0:
        raise ValueError("--max-render-attempts must be positive")

    import numpy as np
    import torch
    from datasets import Audio, load_dataset
    from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

    from data_utils.data_utils import get_groups, sample_scene_recipe

    if not torch.cuda.is_available():
        raise RuntimeError("SNR sweeps currently require a CUDA GPU")

    bins = build_bins(args.min_snr, args.max_snr, args.bin_width)
    cache_dir = str(args.cache_dir)

    print("Loading datasets...")
    speech_ds = load_dataset(
        "parquet",
        data_files=LIBRISPEECH_TEST_CLEAN_PARQUET,
        split="train",
        cache_dir=cache_dir,
    ).cast_column("audio", Audio(decode=False))
    noise_ds = load_dataset(
        "bilguun/musan-noise",
        split="train",
        cache_dir=cache_dir,
    ).cast_column("audio", Audio(decode=False))
    rir_ds = load_dataset(
        "parquet",
        data_files=TREBLE_MONO_PARQUET,
        split="train",
        cache_dir=cache_dir,
    ).cast_column("audio", Audio(decode=False))

    if args.fold is not None:
        _, validation_rooms = FOLDS[args.fold]
        # input_columns keeps the filter from reading every RIR waveform off disk.
        rir_ds = rir_ds.filter(
            lambda room: room in validation_rooms,
            input_columns="Room",
        )
        print(f"Fold {args.fold}: sweeping only held-out rooms {sorted(validation_rooms)}")

    if args.samples_per_bin > len(speech_ds):
        raise ValueError(
            f"Requested {args.samples_per_bin} speech records from {len(speech_ds)}"
        )

    print(
        f"Dataset rows: speech={len(speech_ds)}, noise={len(noise_ds)}, "
        f"RIR={len(rir_ds)}"
    )
    groups = get_groups(rir_ds)
    print(f"Indexed {len(groups)} rooms into {len(bins)} bins of {args.bin_width} dB.")

    selection_rng = np.random.default_rng(args.seed)
    speech_indices = [
        int(index)
        for index in selection_rng.choice(
            len(speech_ds),
            size=args.samples_per_bin,
            replace=False,
        )
    ]

    print(f"Loading processor and model {args.model_id!r}...")
    processor = AutoProcessor.from_pretrained(args.model_id, cache_dir=cache_dir)
    processor_sample_rate = int(processor.feature_extractor.sampling_rate)
    if processor_sample_rate != args.sample_rate:
        raise ValueError(
            f"Scene rate {args.sample_rate} does not match processor rate "
            f"{processor_sample_rate}"
        )

    device = torch.device("cuda")
    model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        args.model_id,
        cache_dir=cache_dir,
        dtype=model_dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    print(f"CUDA device: {torch.cuda.get_device_name(device)}, dtype={model_dtype}")

    scores = {index: WERAccumulator() for index in range(len(bins))}
    rejected_by_bin = {index: 0 for index in range(len(bins))}
    output_records: list[dict[str, Any]] = []

    for batch_start in range(0, len(speech_indices), args.batch_size):
        batch_indices = speech_indices[batch_start : batch_start + args.batch_size]
        base_recipes = [
            sample_scene_recipe(
                speech_index=speech_index,
                noise_ds=noise_ds,
                groups=groups,
                rng=np.random.default_rng(
                    np.random.SeedSequence([args.seed, speech_index, 0])
                ),
                number_of_noises=args.number_of_noises,
            )
            for speech_index in batch_indices
        ]

        for bin_index, bounds in enumerate(bins):
            recipes = []
            scenes = []
            for speech_index, base_recipe in zip(
                batch_indices, base_recipes, strict=True
            ):
                # Offset by one so bin 0 never reuses the base recipe's stream.
                target_rng = np.random.default_rng(
                    np.random.SeedSequence([args.seed, speech_index, bin_index + 1])
                )
                recipe, scene, rejected = render_in_bin(
                    base_recipe=base_recipe,
                    bounds=bounds,
                    target_rng=target_rng,
                    speech_ds=speech_ds,
                    noise_ds=noise_ds,
                    rir_ds=rir_ds,
                    sample_rate=args.sample_rate,
                    max_attempts=args.max_render_attempts,
                )
                if scene["text"] != speech_ds[speech_index]["text"]:
                    raise ValueError("Simulated transcript no longer matches clean speech")
                rejected_by_bin[bin_index] += rejected
                recipes.append(recipe)
                scenes.append(scene)

            hypotheses = transcribe_batch(
                model=model,
                processor=processor,
                scenes=scenes,
                language=args.language,
                sample_rate=args.sample_rate,
                max_new_tokens=args.max_new_tokens,
                device=device,
                model_dtype=model_dtype,
            )

            for speech_index, scene, hypothesis, recipe in zip(
                batch_indices, scenes, hypotheses, recipes, strict=True
            ):
                realized = float(scene["metadata"]["final_snr_db"])
                errors, words, normalized_reference, normalized_hypothesis = scores[
                    bin_index
                ].add(scene["text"], hypothesis, realized)
                speech_record = speech_ds[speech_index]
                output_records.append(
                    {
                        "bin_index": bin_index,
                        "bin_lower": bounds[0],
                        "bin_upper": bounds[1],
                        "speech_index": speech_index,
                        "speech_id": str(speech_record.get("id", speech_index)),
                        "final_snr_db": realized,
                        "room": recipe.room,
                        "receiver": recipe.receiver,
                        "reference": scene["text"],
                        "hypothesis": hypothesis,
                        "normalized_reference": normalized_reference,
                        "normalized_hypothesis": normalized_hypothesis,
                        "word_errors": errors,
                        "reference_words": words,
                        "recipe": asdict(recipe),
                    }
                )

            score = scores[bin_index]
            print(
                f"  [{bounds[0]:>5.1f},{bounds[1]:>5.1f}) "
                f"{score.examples}/{args.samples_per_bin}, "
                f"cumulative WER={100 * score.wer:.2f}%",
                flush=True,
            )

    print_curve(bins, scores)
    print(f"Discarded out-of-bin renders: {rejected_by_bin}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as output_file:
        for record in output_records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "model_id": args.model_id,
        "seed": args.seed,
        "samples_per_bin": args.samples_per_bin,
        "number_of_noises": args.number_of_noises,
        "fold": args.fold,
        "rooms": sorted(groups),
        "normalization": "whisper EnglishTextNormalizer (with english spelling map)",
        "discarded_out_of_bin_renders": rejected_by_bin,
        "bins": [
            {
                "lower": lower,
                "upper": upper,
                "examples": scores[index].examples,
                "word_errors": scores[index].word_errors,
                "reference_words": scores[index].reference_words,
                "wer": scores[index].wer if scores[index].examples else None,
                "mean_realized_snr_db": (
                    sum(scores[index].snr_values) / len(scores[index].snr_values)
                    if scores[index].snr_values
                    else None
                ),
            }
            for index, (lower, upper) in enumerate(bins)
        ],
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote predictions to {predictions_path}")
    print(f"Wrote summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

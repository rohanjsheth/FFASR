"""Evaluate Qwen3-ASR on paired clean and simulated speech by SNR band."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from data_utils.room_folds import FOLDS
from eval_utils.text_norm import edit_distance, normalize_for_wer
from eval_utils.transcribe import transcribe_batch, transcribe_with_scores

if TYPE_CHECKING:
    import numpy as np
    from datasets import Dataset

    from data_utils.data_utils import RenderedScene, SceneRecipe


DEFAULT_MODEL_ID = "Qwen/Qwen3-ASR-1.7B-hf"
LIBRISPEECH_TEST_CLEAN_PARQUET = (
    "https://huggingface.co/datasets/openslr/librispeech_asr/resolve/main/"
    "clean/test/0000.parquet"
)
TREBLE_MONO_PARQUET = (
    "https://huggingface.co/datasets/treble-technologies/Treble10-RIR/"
    "resolve/main/data/rir_mono-00000-of-00001.parquet"
)
BAND_ORDER = ("high", "mid", "low")

# Target values retain the scene sampler's original -8 to 24 dB range. Results
# are assigned using the realized SNR after the pink-noise bed is added.
TARGET_SNR_RANGES = {
    "high": (14.0, 24.0),
    "mid": (8.0, 12.0),
    "low": (-4.0, 6.0),
}


@dataclass(slots=True)
class WERAccumulator:
    examples: int = 0
    word_errors: int = 0
    reference_words: int = 0
    snr_values: list[float] = field(default_factory=list)
    # Drift the Whisper normalizer hides from WER.
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
        self,
        reference: str,
        hypothesis: str,
        snr_db: float | None,
        mean_entropy: float = 0.0,
    ) -> tuple[int, int, str, str]:
        normalized_reference = normalize_for_wer(reference)
        normalized_hypothesis = normalize_for_wer(hypothesis)
        reference_tokens = normalized_reference.split()
        hypothesis_tokens = normalized_hypothesis.split()
        if not reference_tokens:
            raise ValueError(f"Reference became empty after normalization: {reference!r}")

        errors = edit_distance(reference_tokens, hypothesis_tokens)
        self.examples += 1
        self.word_errors += errors
        self.reference_words += len(reference_tokens)
        if snr_db is not None:
            self.snr_values.append(snr_db)
        self.catastrophic += errors > len(reference_tokens)
        self.with_punctuation += any(character in ".,?!;:" for character in hypothesis)
        self.hypothesis_words += len(hypothesis_tokens)
        self.entropy_sum += mean_entropy

        return errors, len(reference_tokens), normalized_reference, normalized_hypothesis


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure baseline WER on paired clean speech and high-, mid-, and "
            "low-SNR far-field simulations."
        )
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--rendered-parquet",
        help=("Score frozen audio from scripts.prepare_noise_eval without resampling "
              "or rendering. Speech/RIR/seed/sample-count options do not apply."),
    )
    parser.add_argument(
        "--noise-parquet",
        help="Optional dry-noise parquet for live rendering; defaults to MUSAN.",
    )
    parser.add_argument("--language", default="English")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--samples-per-band", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--number-of-noises", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument(
        "--fold",
        type=int,
        choices=tuple(FOLDS),
        default=None,
        help=(
            "Restrict rooms to the validation half of a training fold, so a "
            "fine-tuned checkpoint is scored only on rooms it never saw. Omit to "
            "use all ten rooms, which is what a pretrained baseline wants."
        ),
    )
    parser.add_argument(
        "--rir-parquet",
        default=TREBLE_MONO_PARQUET,
        help=(
            "RIR source. Needs `audio`, `Room`, `Receiver Label` and "
            "`Direct Path Length [m]`; scripts/ingest_mit_rir.py writes those "
            "for real measured responses."
        ),
    )
    parser.add_argument(
        "--speech-parquet",
        default=LIBRISPEECH_TEST_CLEAN_PARQUET,
        help=(
            "Speech source. Needs `audio`, `id` and `text` columns; "
            "scripts/ingest_voxpopuli.py writes exactly those."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/snr_wer"))
    parser.add_argument(
        "--max-render-attempts",
        type=int,
        default=20,
        help=(
            "Maximum target-SNR draws used to obtain a realized SNR inside the "
            "requested band."
        ),
    )
    return parser.parse_args(argv)


def classify_snr(snr_db: float) -> str | None:
    """Map realized SNR to the requested bands, leaving both gaps unassigned."""
    if snr_db > 14.0:
        return "high"
    if 8.0 <= snr_db <= 12.0:
        return "mid"
    if snr_db < 6.0:
        return "low"
    return None


def render_in_band(
    base_recipe: SceneRecipe,
    requested_band: str,
    target_rng: np.random.Generator,
    speech_ds: Dataset,
    noise_ds: Dataset,
    rir_ds: Dataset,
    sample_rate: int,
    max_attempts: int,
) -> tuple[SceneRecipe, RenderedScene, int]:
    """Render until achieved SNR lands inside the requested evaluation band."""
    from data_utils.data_utils import render_scene_from_recipe

    lower, upper = TARGET_SNR_RANGES[requested_band]
    rejected = 0

    for _ in range(max_attempts):
        target_snr_db = float(target_rng.uniform(lower, upper))
        recipe = replace(base_recipe, target_snr_db=target_snr_db)
        scene = render_scene_from_recipe(
            recipe=recipe,
            speech_ds=speech_ds,
            noise_ds=noise_ds,
            rir_ds=rir_ds,
            sr=sample_rate,
        )
        realized_snr_db = float(scene["metadata"]["final_snr_db"])
        if classify_snr(realized_snr_db) == requested_band:
            return recipe, scene, rejected
        rejected += 1

    raise RuntimeError(
        f"Could not render a {requested_band}-SNR scene after {max_attempts} "
        "attempts; inspect target and realized SNR values"
    )


def record_predictions(
    condition: str,
    speech_indices: Sequence[int],
    speech_ds: Dataset,
    scenes: Sequence[RenderedScene],
    hypotheses: Sequence[str],
    accumulator: WERAccumulator,
    recipes: Sequence[SceneRecipe | None],
    output_records: list[dict[str, Any]],
) -> None:
    """Accumulate corpus WER and retain auditable per-utterance predictions."""
    for speech_index, scene, hypothesis, recipe in zip(
        speech_indices,
        scenes,
        hypotheses,
        recipes,
        strict=True,
    ):
        reference = scene["text"]
        realized_snr_db = (
            None
            if condition == "clean"
            else float(scene["metadata"]["final_snr_db"])
        )
        errors, reference_words, normalized_reference, normalized_hypothesis = (
            accumulator.add(reference, hypothesis, realized_snr_db)
        )
        speech_record = speech_ds[speech_index]

        output_records.append(
            {
                "speech_index": speech_index,
                "speech_id": str(speech_record.get("id", speech_index)),
                "condition": condition,
                "target_snr_db": None if recipe is None else recipe.target_snr_db,
                "final_snr_db": realized_snr_db,
                "room": None if recipe is None else recipe.room,
                "receiver": None if recipe is None else recipe.receiver,
                "reference": reference,
                "hypothesis": hypothesis,
                "normalized_reference": normalized_reference,
                "normalized_hypothesis": normalized_hypothesis,
                "word_errors": errors,
                "reference_words": reference_words,
                "recipe": None if recipe is None else asdict(recipe),
            }
        )


def print_progress(condition: str, score: WERAccumulator, total: int) -> None:
    print(
        f"  {condition}: {score.examples}/{total}, "
        f"cumulative WER={100 * score.wer:.2f}%"
    )


def score_summary(score: WERAccumulator) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "examples": score.examples,
        "word_errors": score.word_errors,
        "reference_words": score.reference_words,
        "wer": score.wer,
    }
    summary["catastrophic"] = score.catastrophic
    summary["catastrophic_rate"] = score.catastrophic / score.examples
    summary["punctuation_rate"] = score.with_punctuation / score.examples
    summary["length_ratio"] = score.hypothesis_words / score.reference_words
    summary["mean_entropy"] = score.entropy_sum / score.examples
    if score.snr_values:
        summary["realized_snr_db"] = {
            "mean": sum(score.snr_values) / len(score.snr_values),
            "min": min(score.snr_values),
            "max": max(score.snr_values),
        }
    return summary


def print_report(scores: dict[str, WERAccumulator]) -> None:
    clean_wer = scores["clean"].wer
    print("\nWER by condition (corpus-level, case/punctuation ignored):")
    print("condition  examples  realized SNR dB       errors/words      WER   delta")
    for condition in ("clean", *BAND_ORDER):
        score = scores[condition]
        if score.snr_values:
            mean_snr = sum(score.snr_values) / len(score.snr_values)
            snr_text = (
                f"{mean_snr:5.2f} "
                f"[{min(score.snr_values):5.2f}, {max(score.snr_values):5.2f}]"
            )
        else:
            snr_text = "          n/a         "
        delta = 100 * (score.wer - clean_wer)
        print(
            f"{condition:<9}  {score.examples:>8}  {snr_text:<21}  "
            f"{score.word_errors:>6}/{score.reference_words:<6}  "
            f"{100 * score.wer:>6.2f}%  {delta:>+6.2f} pp"
        )


def load_model(args: argparse.Namespace) -> tuple[Any, Any, Any, Any]:
    import torch
    from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("SNR-band WER evaluation currently requires a CUDA GPU")
    print(f"Loading processor and model {args.model_id!r}...")
    processor = AutoProcessor.from_pretrained(args.model_id, cache_dir=str(args.cache_dir))
    if int(processor.feature_extractor.sampling_rate) != args.sample_rate:
        raise ValueError("Scene rate does not match processor rate")
    device = torch.device("cuda")
    model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        args.model_id, cache_dir=str(args.cache_dir), dtype=model_dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    print(f"CUDA device: {torch.cuda.get_device_name(device)}, dtype={model_dtype}, "
          f"batch_size={args.batch_size}")
    return model, processor, device, model_dtype


def frozen_batch_scenes(rows: Sequence[dict[str, Any]], sample_rate: int) -> list[RenderedScene]:
    import io

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
            raise ValueError(f"Frozen audio rate mismatch for {row['id']}; no resampling allowed")
        if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
            raise ValueError(f"Invalid frozen audio for {row['id']}")
        scenes.append({"audio": audio, "text": row["text"],
                       "metadata": json.loads(row["metadata_json"])})
    return scenes


def score_frozen_batch(
    rows: Sequence[dict[str, Any]], scored: Sequence[dict[str, Any]],
    scores: dict[str, WERAccumulator],
) -> list[dict[str, Any]]:
    records = []
    for row, result in zip(rows, scored, strict=True):
        hypothesis = result["hypothesis"]
        condition = row["condition"]
        key = "clean" if condition == "clean" else f"{row['noise_source']}/{condition}"
        metadata = json.loads(row["metadata_json"])
        snr = None if condition == "clean" else float(metadata["final_snr_db"])
        accumulator = scores.setdefault(key, WERAccumulator())
        errors, words, reference, normalized = accumulator.add(
            row["text"], hypothesis, snr, result["mean_entropy"]
        )
        records.append({
            "id": row["id"], "speech_id": row["speech_id"],
            "speech_index": row["speech_index"], "condition": key,
            "audio_sha256": row["sha256"], "reference": row["text"],
            "hypothesis": hypothesis, "normalized_reference": reference,
            "normalized_hypothesis": normalized, "word_errors": errors,
            "reference_words": words, "final_snr_db": snr,
            "recipe": json.loads(row["recipe_json"]), "metadata": metadata,
            "noise_records": json.loads(row["noise_records_json"]),
            **{key: result[key] for key in
               ("tokens", "avg_logprob", "min_logprob", "mean_entropy", "max_entropy")},
        })
    return records


def evaluate_frozen(args: argparse.Namespace) -> int:
    from datasets import load_dataset

    if args.noise_parquet is not None:
        raise ValueError("--noise-parquet cannot be combined with --rendered-parquet")
    predictions_path = args.output_dir / "predictions.jsonl"
    summary_path = args.output_dir / "summary.json"
    if predictions_path.exists() or summary_path.exists():
        raise FileExistsError(f"Choose a fresh --output-dir: {args.output_dir}")
    dataset = load_dataset("parquet", data_files=args.rendered_parquet, split="train",
                           cache_dir=str(args.cache_dir))
    required = {"id", "speech_id", "speech_index", "audio", "text", "condition", "noise_source",
                "sha256", "sample_rate", "metadata_json", "recipe_json", "noise_records_json",
                "experiment_json"}
    if required - set(dataset.column_names) or not len(dataset):
        raise ValueError("Expected nonempty output from scripts.prepare_noise_eval")
    if set(dataset["sample_rate"]) != {args.sample_rate}:
        raise ValueError("Frozen corpus sample rate does not match --sample-rate")
    if not set(dataset["condition"]) <= {"clean", *BAND_ORDER}:
        raise ValueError("Unknown frozen condition")
    if len(set(dataset["id"])) != len(dataset):
        raise ValueError("Duplicate frozen scene IDs")
    print(f"Scoring {len(dataset)} frozen scenes; live-rendering options are ignored")
    model, processor, device, model_dtype = load_model(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores: dict[str, WERAccumulator] = {}
    with predictions_path.open("x", encoding="utf-8") as output:
        for start in range(0, len(dataset), args.batch_size):
            rows = [dataset[i] for i in range(start, min(start + args.batch_size, len(dataset)))]
            scenes = frozen_batch_scenes(rows, args.sample_rate)
            scored = transcribe_with_scores(
                model=model, processor=processor, scenes=scenes, language=args.language,
                sample_rate=args.sample_rate, max_new_tokens=args.max_new_tokens,
                device=device, model_dtype=model_dtype,
            )
            for record in score_frozen_batch(rows, scored, scores):
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            print(f"Scored {start + len(rows)}/{len(dataset)} frozen scenes", flush=True)
    summary = {
        "model_id": args.model_id, "rendered_parquet": args.rendered_parquet,
        "dataset_fingerprint": dataset._fingerprint, "sample_rate": args.sample_rate,
        "language": args.language, "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "experiment": json.loads(dataset[0]["experiment_json"]),
        "normalization": "whisper EnglishTextNormalizer (with english spelling map)",
        "conditions": {name: score_summary(score) for name, score in sorted(scores.items())},
    }
    with summary_path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    for name, score in sorted(scores.items()):
        print(f"{name}: {score.examples} examples, WER={100 * score.wer:.2f}%, "
              f"catastrophic={score.catastrophic}, "
              f"punct={100 * score.with_punctuation / score.examples:.0f}%, "
              f"len/ref={score.hypothesis_words / score.reference_words:.3f}")
    print(f"Wrote {predictions_path} and {summary_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.samples_per_band <= 0:
        raise ValueError("--samples-per-band must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.max_render_attempts <= 0:
        raise ValueError("--max-render-attempts must be positive")
    if args.rendered_parquet is not None:
        return evaluate_frozen(args)

    import numpy as np
    import torch
    from datasets import Audio, load_dataset

    from data_utils.data_utils import (
        get_groups,
        render_clean_scene,
        sample_scene_recipe,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("SNR-band WER evaluation currently requires a CUDA GPU")

    cache_dir = str(args.cache_dir)
    print("Loading full map-style evaluation datasets...")
    speech_ds = load_dataset(
        "parquet",
        data_files=args.speech_parquet,
        split="train",
        cache_dir=cache_dir,
    ).cast_column("audio", Audio(decode=False))
    noise_ds = (
        load_dataset("parquet", data_files=args.noise_parquet, split="train", cache_dir=cache_dir)
        if args.noise_parquet else
        load_dataset("bilguun/musan-noise", split="train", cache_dir=cache_dir)
    ).cast_column("audio", Audio(decode=False))
    rir_ds = load_dataset(
        "parquet",
        data_files=args.rir_parquet,
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
        print(
            f"Fold {args.fold}: scoring only held-out rooms "
            f"{sorted(validation_rooms)}"
        )

    if args.samples_per_band > len(speech_ds):
        raise ValueError(
            f"Requested {args.samples_per_band} speech records from {len(speech_ds)}"
        )

    print(
        f"Dataset rows: speech={len(speech_ds)}, noise={len(noise_ds)}, "
        f"RIR={len(rir_ds)}"
    )
    groups = get_groups(rir_ds)
    print(f"Indexed {len(groups)} rooms.")

    selection_rng = np.random.default_rng(args.seed)
    speech_indices = [
        int(index)
        for index in selection_rng.choice(
            len(speech_ds),
            size=args.samples_per_band,
            replace=False,
        )
    ]

    model, processor, device, model_dtype = load_model(args)

    scores = {
        condition: WERAccumulator()
        for condition in ("clean", *BAND_ORDER)
    }
    output_records: list[dict[str, Any]] = []
    rejected_by_band = {band: 0 for band in BAND_ORDER}

    for batch_start in range(0, len(speech_indices), args.batch_size):
        batch_indices = speech_indices[batch_start : batch_start + args.batch_size]
        base_recipes = []
        for speech_index in batch_indices:
            recipe_rng = np.random.default_rng(
                np.random.SeedSequence([args.seed, speech_index, 0])
            )
            base_recipes.append(
                sample_scene_recipe(
                    speech_index=speech_index,
                    noise_ds=noise_ds,
                    groups=groups,
                    rng=recipe_rng,
                    number_of_noises=args.number_of_noises,
                )
            )

        clean_scenes = [
            render_clean_scene(speech_index, speech_ds, args.sample_rate)
            for speech_index in batch_indices
        ]
        clean_hypotheses = transcribe_batch(
            model=model,
            processor=processor,
            scenes=clean_scenes,
            language=args.language,
            sample_rate=args.sample_rate,
            max_new_tokens=args.max_new_tokens,
            device=device,
            model_dtype=model_dtype,
        )
        record_predictions(
            condition="clean",
            speech_indices=batch_indices,
            speech_ds=speech_ds,
            scenes=clean_scenes,
            hypotheses=clean_hypotheses,
            accumulator=scores["clean"],
            recipes=[None] * len(batch_indices),
            output_records=output_records,
        )
        print_progress("clean", scores["clean"], args.samples_per_band)

        for band_index, band in enumerate(BAND_ORDER, start=1):
            band_recipes = []
            band_scenes = []
            for speech_index, base_recipe in zip(
                batch_indices,
                base_recipes,
                strict=True,
            ):
                target_rng = np.random.default_rng(
                    np.random.SeedSequence([args.seed, speech_index, band_index])
                )
                recipe, scene, rejected = render_in_band(
                    base_recipe=base_recipe,
                    requested_band=band,
                    target_rng=target_rng,
                    speech_ds=speech_ds,
                    noise_ds=noise_ds,
                    rir_ds=rir_ds,
                    sample_rate=args.sample_rate,
                    max_attempts=args.max_render_attempts,
                )
                if scene["text"] != speech_ds[speech_index]["text"]:
                    raise ValueError("Simulated transcript no longer matches clean speech")
                rejected_by_band[band] += rejected
                band_recipes.append(recipe)
                band_scenes.append(scene)

            band_hypotheses = transcribe_batch(
                model=model,
                processor=processor,
                scenes=band_scenes,
                language=args.language,
                sample_rate=args.sample_rate,
                max_new_tokens=args.max_new_tokens,
                device=device,
                model_dtype=model_dtype,
            )
            record_predictions(
                condition=band,
                speech_indices=batch_indices,
                speech_ds=speech_ds,
                scenes=band_scenes,
                hypotheses=band_hypotheses,
                accumulator=scores[band],
                recipes=band_recipes,
                output_records=output_records,
            )
            print_progress(band, scores[band], args.samples_per_band)

    print_report(scores)
    print(f"Discarded out-of-band renders: {rejected_by_band}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as output_file:
        for record in output_records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "model_id": args.model_id,
        "speech_parquet": args.speech_parquet,
        "rir_parquet": args.rir_parquet,
        "noise_source": args.noise_parquet or "bilguun/musan-noise",
        "seed": args.seed,
        "samples_per_band": args.samples_per_band,
        "number_of_noises": args.number_of_noises,
        "fold": args.fold,
        "rooms": sorted(groups),
        "band_definitions": {
            "high": "final_snr_db > 14",
            "mid": "8 <= final_snr_db <= 12",
            "low": "final_snr_db < 6",
        },
        "normalization": "whisper EnglishTextNormalizer (with english spelling map)",
        "discarded_out_of_band_renders": rejected_by_band,
        "conditions": {
            condition: score_summary(scores[condition])
            for condition in ("clean", *BAND_ORDER)
        },
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

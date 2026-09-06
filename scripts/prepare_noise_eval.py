"""Freeze a paired MUSAN/AID evaluation corpus on CPU, using the unchanged DSP.

All non-noise recipe fields are shared across noise datasets. Both models then
read these exact WAV bytes instead of independently sampling/rendering scenes.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterator, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Audio, Dataset, load_dataset

from data_utils.data_utils import (
    SceneRecipe, audio_duration_seconds, get_groups, render_clean_scene,
    render_scene_from_recipe, sample_scene_recipe,
)
from data_utils.room_folds import FOLDS
from scripts.evaluate_snr_wer import (
    BAND_ORDER, LIBRISPEECH_TEST_CLEAN_PARQUET, TARGET_SNR_RANGES,
    TREBLE_MONO_PARQUET, classify_snr,
)
from scripts.noise_eval_io import sha256, wav_bytes, write_parquet


def stratified_indices(
    rng: np.random.Generator, kinds: Sequence[str], number_of_noises: int,
) -> tuple[int, ...]:
    """Guarantee one continuous and one transient source, as FFASR scenes do.

    The official methodology states every scene carries both a transient and a
    continuous interferer. A uniform draw only lands on one of each about half
    the time, which leaves a third of scenes with no steady masker at all. The
    remaining slots stay uniform, and the result is permuted so that kind is not
    correlated with the pre-roll offset assigned to each slot.
    """
    labels = np.asarray(kinds)
    pools = {kind: np.flatnonzero(labels == kind) for kind in ("continuous", "transient")}
    for kind, pool in pools.items():
        if not pool.size:
            raise ValueError(f"Stratified pairing needs at least one {kind} stem")
    chosen = [int(rng.choice(pool)) for pool in pools.values()]
    if number_of_noises > 2:
        remaining = np.setdiff1d(np.arange(labels.size), chosen)
        chosen += [int(i) for i in rng.choice(remaining, number_of_noises - 2, replace=False)]
    return tuple(int(i) for i in rng.permutation(chosen))


def paired_recipes(
    speech_index: int, groups: Any, noise_sizes: dict[str, int], seed: int,
    number_of_noises: int, noise_kinds: dict[str, Sequence[str]] | None = None,
) -> dict[str, SceneRecipe]:
    # A fixed-size index-only pool keeps the original sampler's RNG consumption
    # independent of the real noise corpus. Do not change the training sampler.
    base = sample_scene_recipe(
        speech_index, range(number_of_noises), groups,
        np.random.default_rng(np.random.SeedSequence([seed, speech_index, 0])),
        number_of_noises,
    )
    recipes = {}
    for name, size in noise_sizes.items():
        if size < number_of_noises:
            raise ValueError(f"{name} has fewer than {number_of_noises} noise records")
        noise_rng = np.random.default_rng(np.random.SeedSequence([seed, speech_index, 1]))
        kinds = None if noise_kinds is None else noise_kinds.get(name)
        if kinds is None:
            indices = tuple(int(i) for i in noise_rng.choice(size, number_of_noises, replace=False))
        else:
            indices = stratified_indices(noise_rng, kinds, number_of_noises)
        recipes[name] = replace(base, noise_indices=indices)
    return recipes


def require_unlooped_stems(
    recipe: SceneRecipe, speech_ds: Dataset, noise_ds: Dataset, rir_ds: Dataset, sr: int,
) -> None:
    """Conservative raw-RIR bound on the mixer's offset + pre-roll + speech tail."""
    def length(audio: Any) -> int:
        return math.ceil(audio_duration_seconds(audio) * sr)

    speech_length = length(speech_ds[recipe.speech_index]["audio"])
    speech_tail = length(rir_ds[recipe.speech_rir_index]["audio"]) - 1
    for index, rir_index, offset in zip(
        recipe.noise_indices, recipe.noise_rir_indices, recipe.offsets_ms, strict=True,
    ):
        required = speech_length + speech_tail + length(rir_ds[rir_index]["audio"]) - 1
        required += round(offset * sr / 1000)
        available = length(noise_ds[index]["audio"])
        if available < required:
            raise ValueError(
                f"AID stem {index} needs >= {required / sr:.3f}s for speech "
                f"{recipe.speech_index}, but has {available / sr:.3f}s. "
                "Reassemble with a longer --duration-seconds; DSP looping is unchanged."
            )


def render_pair_in_band(
    recipes: dict[str, SceneRecipe], band: str, rng: np.random.Generator,
    speech_ds: Dataset, noises: dict[str, Dataset], rir_ds: Dataset,
    sample_rate: int, max_attempts: int, snr_tolerance_db: float,
) -> tuple[dict[str, tuple[SceneRecipe, dict[str, Any]]], int]:
    realized = []
    for rejected in range(max_attempts):
        target = float(rng.uniform(*TARGET_SNR_RANGES[band]))
        paired = {}
        for name, base in recipes.items():
            recipe = replace(base, target_snr_db=target)
            scene = render_scene_from_recipe(recipe, speech_ds, noises[name], rir_ds, sample_rate)
            paired[name] = (recipe, scene)
        realized = [float(scene["metadata"]["final_snr_db"]) for _, scene in paired.values()]
        if (
            all(classify_snr(value) == band for value in realized)
            and max(realized) - min(realized) <= snr_tolerance_db
        ):
            return paired, rejected
    raise RuntimeError(
        f"Could not jointly render {band} for speech {next(iter(recipes.values())).speech_index}; "
        f"realized SNRs={realized}. Inspect stem activity and SNR settings."
    )


def iter_scenes(
    speech_ds: Dataset, noises: dict[str, Dataset], rir_ds: Dataset,
    args: argparse.Namespace, provenance: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    groups = get_groups(rir_ds)
    # Only corpora that label their stems can be stratified; MUSAN carries no
    # transient/continuous annotation and keeps its uniform draw either way.
    noise_kinds = None
    if args.stratify_noise_kinds:
        noise_kinds = {name: ds["noise_kind"] for name, ds in noises.items()
                       if "noise_kind" in ds.column_names}
        if not noise_kinds:
            raise ValueError("--stratify-noise-kinds needs a corpus with a noise_kind column")
    selection = np.random.default_rng(args.seed).choice(
        len(speech_ds), args.samples_per_band, replace=False,
    )
    experiment_json = json.dumps(provenance, sort_keys=True)

    def record(index: int, source: str, band: str, scene: dict[str, Any],
               recipe: SceneRecipe | None = None, rejected: int = 0) -> dict[str, Any]:
        encoded = wav_bytes(scene["audio"], args.sample_rate)
        references = []
        if recipe is not None:
            for noise_index in recipe.noise_indices:
                row = noises[source][noise_index]
                references.append({
                    "index": noise_index, "id": str(row.get("id", noise_index)),
                    "category": row.get("category"), "noise_kind": row.get("noise_kind"),
                    "sha256": row.get("sha256"),
                })
        return {
            "id": f"{index}-{source}-{band}", "speech_index": index,
            "speech_id": str(speech_ds[index].get("id", index)),
            "text": scene["text"], "condition": band, "noise_source": source,
            "audio": {"bytes": encoded, "path": f"{index}-{source}-{band}.wav"},
            "sample_rate": args.sample_rate, "sha256": sha256(encoded),
            "recipe_json": json.dumps(None if recipe is None else asdict(recipe), sort_keys=True),
            "metadata_json": json.dumps(scene["metadata"], sort_keys=True),
            "noise_records_json": json.dumps(references, sort_keys=True),
            "rejected_pairs": rejected, "experiment_json": experiment_json,
        }

    for position, raw_index in enumerate(selection, start=1):
        index = int(raw_index)
        recipes = paired_recipes(index, groups, {k: len(v) for k, v in noises.items()},
                                 args.seed, args.number_of_noises, noise_kinds)
        require_unlooped_stems(recipes["aid"], speech_ds, noises["aid"], rir_ds, args.sample_rate)
        yield record(index, "none", "clean", render_clean_scene(index, speech_ds, args.sample_rate))
        for band_index, band in enumerate(BAND_ORDER, start=1):
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, index, 2, band_index]))
            paired, rejected = render_pair_in_band(
                recipes, band, rng, speech_ds, noises, rir_ds, args.sample_rate,
                args.max_render_attempts, args.snr_tolerance_db,
            )
            for source, (recipe, scene) in paired.items():
                yield record(index, source, band, scene, recipe, rejected)
        print(f"Rendered paired speech {position}/{len(selection)}", flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aid-noise-parquet", required=True)
    parser.add_argument("--musan-noise-parquet", help="Optional local MUSAN copy; defaults to bilguun/musan-noise")
    parser.add_argument("--speech-parquet", default=LIBRISPEECH_TEST_CLEAN_PARQUET)
    parser.add_argument("--rir-parquet", default=TREBLE_MONO_PARQUET)
    parser.add_argument("--fold", type=int, choices=tuple(FOLDS))
    parser.add_argument("--samples-per-band", type=int, default=500)
    parser.add_argument("--number-of-noises", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument(
        "--stratify-noise-kinds", action="store_true",
        help="Force one continuous and one transient source per scene in every "
             "labelled corpus, matching FFASR's stated scene composition.",
    )
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-render-attempts", type=int, default=20)
    parser.add_argument(
        "--snr-tolerance-db", type=float, default=0.25,
        help="Maximum paired realized-SNR gap after the unchanged pink-noise bed is added.",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf_cache"))
    parser.add_argument("--output", type=Path, default=Path("data/noise_eval.parquet"))
    args = parser.parse_args(argv)
    if min(args.samples_per_band, args.sample_rate, args.max_render_attempts) <= 0 or args.seed < 0:
        parser.error("counts/sample-rate must be positive and seed nonnegative")
    if not np.isfinite(args.snr_tolerance_db) or args.snr_tolerance_db < 0:
        parser.error("snr-tolerance-db must be finite and nonnegative")
    if args.stratify_noise_kinds and args.number_of_noises < 2:
        parser.error("stratify-noise-kinds needs at least two noise sources per scene")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)

    def parquet(path: str) -> Dataset:
        return load_dataset("parquet", data_files=path, split="train",
                            cache_dir=str(args.cache_dir)).cast_column("audio", Audio(decode=False))

    speech = parquet(args.speech_parquet)
    rirs = parquet(args.rir_parquet)
    if args.fold is not None:
        _, validation_rooms = FOLDS[args.fold]
        rirs = rirs.filter(lambda room: room in validation_rooms, input_columns="Room")
    if args.samples_per_band > len(speech):
        raise ValueError(f"Requested {args.samples_per_band} speech rows from {len(speech)}")
    noises = {
        "musan": parquet(args.musan_noise_parquet) if args.musan_noise_parquet else load_dataset(
            "bilguun/musan-noise", split="train", cache_dir=str(args.cache_dir),
        ).cast_column("audio", Audio(decode=False)),
        "aid": parquet(args.aid_noise_parquet),
    }
    provenance = {
        "version": 1, "args": {key: str(value) if isinstance(value, Path) else value
                               for key, value in vars(args).items()},
        "dataset_fingerprints": {key: ds._fingerprint for key, ds in
                                 {"speech": speech, "rir": rirs, **noises}.items()},
        "pairing": "fixed-size geometry sampler; separate noise RNG; joint target-SNR retries",
        "audio_storage": "mono float32 WAV; sha256 per row",
    }
    count = write_parquet(args.output, iter_scenes(speech, noises, rirs, args, provenance))
    print(f"Wrote {count} frozen scenes to {args.output}; no model loaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

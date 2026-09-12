"""Measure how far a fine-tune moved the projector output from base on identical audio."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from datasets import Audio, load_dataset
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

from data_utils.data_utils import get_groups, sample_scene_recipe
from scripts.evaluate_snr_wer import TREBLE_MONO_PARQUET, render_in_band


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen3-ASR-1.7B-hf")
    parser.add_argument("--tuned-model", required=True)
    parser.add_argument("--speech-parquet", required=True)
    parser.add_argument("--rir-parquet", default=TREBLE_MONO_PARQUET)
    parser.add_argument("--band", default="low", choices=("high", "mid", "low", "clean"))
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--number-of-noises", type=int, default=2)
    parser.add_argument("--language", default="English")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf_cache"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def load(model_id: str, cache_dir: str) -> Qwen3ASRForConditionalGeneration:
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        model_id, cache_dir=cache_dir, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    return model.eval().cuda()


def projector_output(model, inputs) -> torch.Tensor:
    captured = {}

    def hook(_module, _args, output):
        captured["value"] = output[0] if isinstance(output, tuple) else output

    handle = model.model.multi_modal_projector.register_forward_hook(hook)
    with torch.inference_mode():
        model(**inputs)
    handle.remove()
    return captured["value"].float()


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Rotation-invariant representational similarity; 1.0 means identical geometry."""
    x = x.reshape(-1, x.shape[-1]) - x.reshape(-1, x.shape[-1]).mean(0)
    y = y.reshape(-1, y.shape[-1]) - y.reshape(-1, y.shape[-1]).mean(0)
    cross = (y.T @ x).norm() ** 2
    return float(cross / ((x.T @ x).norm() * (y.T @ y).norm()))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cache_dir = str(args.cache_dir)

    speech_ds = load_dataset("parquet", data_files=args.speech_parquet, split="train",
                             cache_dir=cache_dir).cast_column("audio", Audio(decode=False))
    noise_ds = load_dataset("bilguun/musan-noise", split="train",
                            cache_dir=cache_dir).cast_column("audio", Audio(decode=False))
    rir_ds = load_dataset("parquet", data_files=args.rir_parquet, split="train",
                          cache_dir=cache_dir).cast_column("audio", Audio(decode=False))
    groups = get_groups(rir_ds)

    processor = AutoProcessor.from_pretrained(args.base_model, cache_dir=cache_dir)
    base = load(args.base_model, cache_dir)
    tuned = load(args.tuned_model, cache_dir)

    rng = np.random.default_rng(args.seed)
    indices = [int(i) for i in rng.choice(len(speech_ds), args.samples, replace=False)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for n, speech_index in enumerate(indices, 1):
            recipe_rng = np.random.default_rng(np.random.SeedSequence([args.seed, speech_index, 0]))
            recipe = sample_scene_recipe(
                speech_index=speech_index, noise_ds=noise_ds, groups=groups,
                rng=recipe_rng, number_of_noises=args.number_of_noises,
            )
            target_rng = np.random.default_rng(np.random.SeedSequence([args.seed, speech_index, 1]))
            recipe, scene, _ = render_in_band(
                base_recipe=recipe, requested_band=args.band, target_rng=target_rng,
                speech_ds=speech_ds, noise_ds=noise_ds, rir_ds=rir_ds,
                sample_rate=args.sample_rate, max_attempts=20,
            )

            inputs = processor.apply_transcription_request(
                audio=[scene["audio"]], language=args.language,
                processor_kwargs={"sampling_rate": args.sample_rate},
            ).to("cuda", torch.bfloat16)

            b = projector_output(base, inputs)
            t = projector_output(tuned, inputs)
            diff = t - b
            record = {
                "speech_index": speech_index,
                "final_snr_db": float(recipe.target_snr_db),
                "frames": int(b.shape[-2]),
                "rel_l2": float(diff.norm() / b.norm()),
                "cosine": float(
                    torch.nn.functional.cosine_similarity(
                        b.reshape(-1, b.shape[-1]), t.reshape(-1, t.shape[-1]), dim=-1
                    ).mean()
                ),
                "cka": linear_cka(b, t),
                "base_norm": float(b.norm()),
                "tuned_norm": float(t.norm()),
            }
            handle.write(json.dumps(record) + "\n")
            if n % 25 == 0:
                print(f"{n}/{len(indices)}", flush=True)

    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

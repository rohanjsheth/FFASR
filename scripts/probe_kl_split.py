"""How much distillation signal lived at the positions v2 masked out?

v2 trained with `mask_punctuation = true`, which sets every pure-punctuation
label to -100 and so drops it from the KL average -- about 12% of positions.
Whether that mattered depends on how much teacher and student actually diverge
there. Under teacher forcing both see the same text prefix, and punctuation is
largely prefix-determined, so the KL at those positions may be near zero.

This runs the training loss on unmasked labels and reports the per-position KL
split by whether the position would have been masked.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch
from datasets import Audio, load_dataset
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

from data_utils.data_collator import Qwen3ASRDataCollator, punctuation_token_ids
from data_utils.SceneDataset import SceneDataset


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-model", default="Qwen/Qwen3-ASR-1.7B-hf")
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--speech-parquet", required=True)
    parser.add_argument("--text-column", default="text_normalized")
    parser.add_argument("--rir-parquet", default="hf://datasets/rohansheth/AcousticRooms/acoustic_rooms_r20.parquet")
    parser.add_argument("--noise-id", default="bilguun/musan-noise")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--number-of-noises", type=int, default=2)
    parser.add_argument("--clean-probability", type=float, default=0.0)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--language", default="English")
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf_cache"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def load(model_id: str, cache_dir: str) -> Qwen3ASRForConditionalGeneration:
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        model_id, cache_dir=cache_dir, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.config.use_cache = False
    return model.eval().cuda()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cache_dir = str(args.cache_dir)

    speech_ds = load_dataset("parquet", data_files=args.speech_parquet, split="train",
                             cache_dir=cache_dir).cast_column("audio", Audio(decode=False))
    if args.text_column != "text":
        speech_ds = speech_ds.rename_column(args.text_column, "text")
    noise_ds = load_dataset(args.noise_id, split="train",
                            cache_dir=cache_dir).cast_column("audio", Audio(decode=False))
    rir_ds = load_dataset("parquet", data_files=args.rir_parquet, split="train",
                          cache_dir=cache_dir).cast_column("audio", Audio(decode=False))
    print(f"speech={len(speech_ds)} noise={len(noise_ds)} rir={len(rir_ds)}", flush=True)

    processor = AutoProcessor.from_pretrained(args.teacher_model, cache_dir=cache_dir)
    punct_ids = punctuation_token_ids(processor.tokenizer).cuda()

    dataset = SceneDataset(
        speech_ds=speech_ds, noise_ds=noise_ds, rir_ds=rir_ds,
        base_seed=args.base_seed, sample_rate=args.sample_rate,
        number_of_noises=args.number_of_noises,
        clean_probability=args.clean_probability, teacher=True,
    )
    # False so every position survives; the split is applied afterwards.
    collator = Qwen3ASRDataCollator(
        processor=processor, sample_rate=args.sample_rate,
        language=args.language, mask_punctuation=False,
    )

    teacher = load(args.teacher_model, cache_dir)
    student = load(args.student_model, cache_dir)

    totals = {k: 0.0 for k in ("kl_masked", "kl_kept", "sce_masked", "sce_kept",
                               "tent_masked", "tent_kept")}
    counts = {"masked": 0, "kept": 0}
    records = []

    for start in range(0, args.samples, args.batch_size):
        scenes = [dataset[i] for i in range(start, min(start + args.batch_size, args.samples))]
        batch = collator(scenes).to("cuda")
        student_inputs = {k: v for k, v in batch.items() if not k.startswith("teacher_")}
        teacher_inputs = {k[len("teacher_"):]: v for k, v in batch.items() if k.startswith("teacher_")}

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            t_logits = teacher(**teacher_inputs).logits[:, :-1]
            s_logits = student(**student_inputs).logits[:, :-1]

        labels = student_inputs["labels"][:, 1:]
        valid = labels != -100

        t_logp = t_logits.float().log_softmax(-1)
        t_prob = t_logp.exp()
        s_logp = s_logits.float().log_softmax(-1)

        sce = -(t_prob * s_logp).sum(-1)          # the training loss term
        tent = -(t_prob * t_logp).sum(-1)         # teacher entropy, the floor
        kl = sce - tent                           # what the gradient actually chases

        is_punct = torch.isin(labels, punct_ids) & valid
        is_kept = valid & ~is_punct

        for name, sel in (("masked", is_punct), ("kept", is_kept)):
            counts[name] += int(sel.sum())
            totals[f"kl_{name}"] += float((kl * sel).sum())
            totals[f"sce_{name}"] += float((sce * sel).sum())
            totals[f"tent_{name}"] += float((tent * sel).sum())

        done = min(start + args.batch_size, args.samples)
        if done % 64 == 0:
            m, k = counts["masked"], counts["kept"]
            print(f"{done}/{args.samples}  KL masked={totals['kl_masked']/max(1,m):.4f} "
                  f"kept={totals['kl_kept']/max(1,k):.4f}", flush=True)

    result = {
        "teacher_model": args.teacher_model,
        "student_model": args.student_model,
        "samples": args.samples,
        "positions_masked": counts["masked"],
        "positions_kept": counts["kept"],
        "masked_share": counts["masked"] / max(1, counts["masked"] + counts["kept"]),
    }
    for name in ("masked", "kept"):
        n = max(1, counts[name])
        result[f"mean_kl_{name}"] = totals[f"kl_{name}"] / n
        result[f"mean_sce_{name}"] = totals[f"sce_{name}"] / n
        result[f"mean_teacher_entropy_{name}"] = totals[f"tent_{name}"] / n

    total_kl = totals["kl_masked"] + totals["kl_kept"]
    result["kl_share_lost"] = totals["kl_masked"] / max(1e-9, total_kl)
    result["kl_ratio_masked_to_kept"] = result["mean_kl_masked"] / max(1e-9, result["mean_kl_kept"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

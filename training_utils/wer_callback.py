import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

# Shared with the offline eval so both score identically.
from eval_utils.text_norm import edit_distance, normalize_for_wer
from eval_utils.transcribe import transcribe_batch

# Same definitions the offline eval uses, so the two are directly comparable.
# Validation SNR is drawn uniform(-8, 24), so the 6-8 and 12-14 gaps land in "other".
BANDS: dict[str, tuple[float, float]] = {
    "low": (-float("inf"), 6.0),
    "mid": (8.0, 12.0),
    "high": (14.0, float("inf")),
}


def band_for(snr_db: float | None) -> str:
    if snr_db is None:
        return "clean"
    for name, (low, high) in BANDS.items():
        if low <= snr_db <= high:
            return name
    return "other"


class BandWERCallback(TrainerCallback):
    """Log per-SNR-band WER on the validation scenes at every evaluation step."""

    def __init__(
        self,
        dataset: Any,
        processor: Any,
        language: str,
        sample_rate: int,
        num_examples: int,
        batch_size: int,
        max_new_tokens: int,
        scene_cache: Path | None = None,
    ) -> None:
        self._dataset = dataset
        self._processor = processor
        self._language = language
        self._sample_rate = sample_rate
        self._num_examples = min(num_examples, len(dataset))
        self._batch_size = batch_size
        self._max_new_tokens = max_new_tokens
        # Scene construction is deterministic, so the same audio is scored every
        # step -- and across runs, which is why it is worth caching to disk.
        self._scenes = _load_or_render(dataset, self._num_examples, scene_cache)
        self._bands = [
            band_for(scene["metadata"].get("final_snr_db")) for scene in self._scenes
        ]

    def _transcribe(self, model: Any) -> list[str]:
        was_training = model.training
        checkpointing = getattr(model, "is_gradient_checkpointing", False)
        if checkpointing:
            model.gradient_checkpointing_disable()
        model.eval()
        try:
            hypotheses: list[str] = []
            for start in range(0, len(self._scenes), self._batch_size):
                hypotheses.extend(
                    transcribe_batch(
                        model=model,
                        processor=self._processor,
                        scenes=self._scenes[start : start + self._batch_size],
                        language=self._language,
                        sample_rate=self._sample_rate,
                        max_new_tokens=self._max_new_tokens,
                        device=model.device,
                        model_dtype=model.dtype,
                    )
                )
        finally:
            if checkpointing:
                model.gradient_checkpointing_enable()
            if was_training:
                model.train()
        return hypotheses

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Score the untouched model so the trajectory has a zero point.

        Without this the first data point is whatever the first eval_steps
        interval lands on, and a run that only ever recovers toward baseline
        reads as one that improves.
        """
        self._report(kwargs["model"], step=0, output_dir=args.output_dir)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        self._report(
            kwargs["model"],
            step=state.global_step,
            metrics=metrics,
            output_dir=args.output_dir,
        )

    def _report(
        self,
        model: Any,
        step: int,
        metrics: dict[str, float] | None = None,
        output_dir: str | None = None,
    ) -> None:
        with torch.inference_mode():
            hypotheses = self._transcribe(model)

        errors: dict[str, int] = {}
        words: dict[str, int] = {}
        for band, scene, hypothesis in zip(self._bands, self._scenes, hypotheses, strict=True):
            reference = normalize_for_wer(scene["text"]).split()
            if not reference:
                continue
            distance = edit_distance(reference, normalize_for_wer(hypothesis).split())
            for key in (band, "all"):
                errors[key] = errors.get(key, 0) + distance
                words[key] = words.get(key, 0) + len(reference)

        if output_dir is not None:
            self._dump(output_dir, step, hypotheses)

        print(f"\nband WER at step {step} ({self._num_examples} scenes):")
        for key in ("all", "low", "mid", "high", "other", "clean"):
            if words.get(key):
                wer = 100.0 * errors[key] / words[key]
                print(f"  {key:<6} n={words[key]:>6} words  WER={wer:6.2f}%")
                if metrics is not None:
                    metrics[f"eval_wer_{key}"] = wer

    def _dump(self, output_dir: str, step: int, hypotheses: list[str]) -> None:
        """Keep every transcript, so a WER change can be traced to what changed.

        The number alone cannot distinguish a model that mishears from one that
        starts repeating, truncating, or writing in a different register.
        """
        directory = Path(output_dir) / "band_wer"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"step-{step:05d}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for band, scene, hypothesis in zip(
                self._bands, self._scenes, hypotheses, strict=True
            ):
                handle.write(
                    json.dumps(
                        {
                            "band": band,
                            "final_snr_db": scene["metadata"].get("final_snr_db"),
                            "reference": scene["text"],
                            "hypothesis": hypothesis,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def _load_or_render(
    dataset: Any,
    num_examples: int,
    cache_path: Path | None,
) -> list[Any]:
    """Render the validation scenes, reusing a cached copy when one exists.

    Rendering is deterministic given the dataset's seed, so a cache hit returns
    byte-identical audio. The caller owns invalidation by putting the seed, fold
    and example count in the filename.
    """
    if cache_path is not None and cache_path.exists():
        blob = np.load(cache_path, allow_pickle=False)
        records = json.loads(str(blob["meta"]))
        print(f"Reusing {len(records)} cached validation scenes from {cache_path}")
        return [
            {
                "audio": blob[f"audio_{index}"],
                "text": record["text"],
                "metadata": record["metadata"],
            }
            for index, record in enumerate(records)
        ]

    scenes = [dataset[index] for index in range(num_examples)]

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            f"audio_{index}": np.asarray(scene["audio"], dtype=np.float32)
            for index, scene in enumerate(scenes)
        }
        arrays["meta"] = np.array(
            json.dumps(
                [
                    {"text": scene["text"], "metadata": scene["metadata"]}
                    for scene in scenes
                ]
            )
        )
        # Write then rename so an interrupted run never leaves a half file behind.
        # np.savez appends .npz unless handed an open file, which would defeat
        # the rename, so pass the handle rather than the path.
        temporary = cache_path.with_name(cache_path.name + ".tmp")
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)
        temporary.replace(cache_path)
        print(f"Cached {len(scenes)} validation scenes to {cache_path}")

    return scenes

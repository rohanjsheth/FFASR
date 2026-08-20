from typing import Any

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
    ) -> None:
        self._dataset = dataset
        self._processor = processor
        self._language = language
        self._sample_rate = sample_rate
        self._num_examples = min(num_examples, len(dataset))
        self._batch_size = batch_size
        self._max_new_tokens = max_new_tokens
        # Scene construction is deterministic, so the same audio is scored every step.
        self._scenes = [dataset[index] for index in range(self._num_examples)]
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

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        model = kwargs["model"]
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

        print(f"\nband WER at step {state.global_step} ({self._num_examples} scenes):")
        for key in ("all", "low", "mid", "high", "other", "clean"):
            if words.get(key):
                wer = 100.0 * errors[key] / words[key]
                print(f"  {key:<6} n={words[key]:>6} words  WER={wer:6.2f}%")
                if metrics is not None:
                    metrics[f"eval_wer_{key}"] = wer

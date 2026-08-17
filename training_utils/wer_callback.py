"""Autoregressive WER reporting during Trainer evaluation."""

from typing import Any

from torch.utils.data import Dataset
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from data_utils.data_utils import RenderedScene
from scripts.evaluate_snr_wer import WERAccumulator, transcribe_batch


class ValidationWERCallback(TrainerCallback):
    """Print autoregressive WER for the same fixed scenes at every evaluation."""

    def __init__(
        self,
        dataset: Dataset[RenderedScene],
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
        self._num_examples = num_examples
        self._batch_size = batch_size
        self._max_new_tokens = max_new_tokens

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Any | None = None,
        **kwargs: Any,
    ) -> TrainerControl:
        if not state.is_world_process_zero or model is None:
            return control

        floating_parameter = next(
            parameter
            for parameter in model.parameters()
            if parameter.is_floating_point()
        )
        score = WERAccumulator()
        was_training = model.training
        model.eval()

        try:
            for batch_start in range(0, self._num_examples, self._batch_size):
                batch_end = min(
                    batch_start + self._batch_size,
                    self._num_examples,
                )
                scenes = [
                    self._dataset[index]
                    for index in range(batch_start, batch_end)
                ]
                hypotheses = transcribe_batch(
                    model=model,
                    processor=self._processor,
                    scenes=scenes,
                    language=self._language,
                    sample_rate=self._sample_rate,
                    max_new_tokens=self._max_new_tokens,
                    device=floating_parameter.device,
                    model_dtype=floating_parameter.dtype,
                )
                for scene, hypothesis in zip(scenes, hypotheses, strict=True):
                    score.add(scene["text"], hypothesis, snr_db=None)
        finally:
            model.train(was_training)

        wer = score.wer
        print(
            f"Validation generation WER at step {state.global_step}: "
            f"{100 * wer:.2f}% "
            f"({score.word_errors}/{score.reference_words} words)"
        )

        metrics = kwargs.get("metrics")
        if isinstance(metrics, dict):
            metrics["eval_wer"] = wer
        if state.log_history:
            state.log_history[-1]["eval_wer"] = wer
        else:
            state.log_history.append({"eval_wer": wer, "step": state.global_step})
        return control

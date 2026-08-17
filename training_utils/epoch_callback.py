"""Advance the scene RNG epoch so repeated passes render different scenes."""

from typing import Any, Protocol

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


class SupportsSetEpoch(Protocol):
    def set_epoch(self, epoch: int) -> None: ...


class SceneEpochCallback(TrainerCallback):
    def __init__(self, dataset: SupportsSetEpoch) -> None:
        self._dataset = dataset

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self._dataset.set_epoch(int(state.epoch))

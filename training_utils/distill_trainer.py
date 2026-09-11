from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from transformers import Trainer

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module

TEACHER_PREFIX = "teacher_"


def split_teacher_inputs(inputs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    student = {k: v for k, v in inputs.items() if not k.startswith(TEACHER_PREFIX)}
    teacher = {
        k[len(TEACHER_PREFIX):]: v for k, v in inputs.items() if k.startswith(TEACHER_PREFIX)
    }
    return student, teacher


class DistillTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        teacher_model: Module,
        temperature: float = 1.0,
        ce_weight: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._teacher_model = teacher_model
        self._temperature = temperature
        self._ce_weight = ce_weight

    def compute_loss(
        self,
        model: Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> Tensor | tuple[Tensor, Any]:
        student_inputs, teacher_inputs = split_teacher_inputs(inputs)
        if not teacher_inputs:
            return super().compute_loss(model, inputs, return_outputs, **kwargs)

        student_logits = model(**student_inputs).logits[:, :-1]

        # accelerate autocasts only the model it prepared; the teacher is not one.
        with torch.no_grad(), torch.autocast(self.args.device.type, dtype=torch.bfloat16):
            teacher_logits = self._teacher_model(**teacher_inputs).logits[:, :-1]

        mask = student_inputs["labels"][:, 1:] != -100

        teacher_probs = (teacher_logits / self._temperature).softmax(-1)
        student_logp = (student_logits / self._temperature).log_softmax(-1)
        loss = -(teacher_probs * student_logp).sum(-1)
        loss = (loss * mask).sum() / mask.sum() * self._temperature**2

        return loss

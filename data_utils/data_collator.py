from __future__ import annotations

import unicodedata
from typing import TYPE_CHECKING

import torch

from data_utils.data_utils import RenderedScene

if TYPE_CHECKING:
    from audio_utils.audio_types import FloatArray
    from torch import Tensor
    from transformers import BatchFeature, Qwen3ASRProcessor
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def punctuation_token_ids(tokenizer: PreTrainedTokenizerBase) -> Tensor:
    special = set(tokenizer.all_special_ids)
    ids = [
        token_id
        for token, token_id in tokenizer.get_vocab().items()
        if token_id not in special
        and (text := tokenizer.convert_tokens_to_string([token]).strip())
        and all(unicodedata.category(char)[0] in ("P", "S") for char in text)
    ]
    return torch.tensor(sorted(ids), dtype=torch.long)


class Qwen3ASRDataCollator:
    def __init__(
        self,
        processor: Qwen3ASRProcessor,
        sample_rate: int,
        language: str,
        mask_punctuation: bool = True,
    ) -> None:
        self._processor = processor
        self._sample_rate = sample_rate
        self._language = language
        self._asr_text_id = processor.tokenizer.convert_tokens_to_ids("<asr_text>")
        self._mask_punctuation = mask_punctuation
        self._punctuation_ids = punctuation_token_ids(processor.tokenizer)

    def __call__(self, features: list[RenderedScene]) -> BatchFeature:
        if not features:
            raise ValueError("Cannot collate an empty batch")

        texts = [render["text"] for render in features]
        batch = self._encode([render["audio"] for render in features], texts)

        if "teacher_audio" in features[0]:
            teacher = self._encode([render["teacher_audio"] for render in features], texts)
            for key, value in teacher.items():
                batch[f"teacher_{key}"] = value

        return batch

    def _encode(self, audios: list[FloatArray], texts: list[str]) -> BatchFeature:
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio",
                            "audio": audio,
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": f"language {self._language}<asr_text>{text}",
                        }
                    ],
                },
            ]
            for audio, text in zip(audios, texts, strict=True)
        ]

        batch = self._processor.apply_chat_template(
            conversations,
            tokenize=True,
            return_dict=True,
            processor_kwargs={
                "output_labels": True,
                "sampling_rate": self._sample_rate,
            },
        )

        for input_ids, labels in zip(batch["input_ids"], batch["labels"], strict=True):
            boundary = (input_ids == self._asr_text_id).nonzero()[-1].item()
            labels[: boundary + 1] = -100

        if self._mask_punctuation:
            labels = batch["labels"]
            labels[torch.isin(labels, self._punctuation_ids)] = -100

        return batch

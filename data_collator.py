from __future__ import annotations

from typing import TYPE_CHECKING

from data_utils import RenderedScene

if TYPE_CHECKING:
    from transformers import BatchFeature, Qwen3ASRProcessor


class Qwen3ASRDataCollator:
    def __init__(
        self,
        processor: Qwen3ASRProcessor,
        sample_rate: int,
        language: str,
    ) -> None:
        self._processor = processor
        self._sample_rate = sample_rate
        self._language = language

    def __call__(self, features: list[RenderedScene]) -> BatchFeature:
        if not features:
            raise ValueError("Cannot collate an empty batch")

        audios = [render["audio"] for render in features]
        texts = [render["text"] for render in features]

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

        return self._processor.apply_chat_template(
            conversations,
            tokenize=True,
            return_dict=True,
            processor_kwargs={
                "output_labels": True,
                "sampling_rate": self._sample_rate,
            },
        )

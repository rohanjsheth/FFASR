"""The collator must supervise only what the WER metric can see.

Whisper's normalizer deletes punctuation, so a punctuation token is a position
the model can be wrong at for free -- and the fine-tune drifted toward LibriTTS
quoting conventions because of it. Apostrophes are the exception that makes this
delicate: `don't` normalizes to `do not`, but `dont` stays `dont`.
"""

from __future__ import annotations

import numpy as np
import pytest
from transformers import AutoProcessor

from data_utils.data_collator import Qwen3ASRDataCollator

MODEL_ID = "Qwen/Qwen3-ASR-1.7B-hf"
SAMPLE_RATE = 16000


@pytest.fixture(scope="module")
def collator() -> Qwen3ASRDataCollator:
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=".hf_cache")
    return Qwen3ASRDataCollator(processor, SAMPLE_RATE, "English")


def supervised_text(collator: Qwen3ASRDataCollator, text: str) -> str:
    feature = {
        "audio": np.zeros(SAMPLE_RATE, dtype=np.float32),
        "text": text,
        "metadata": {},
    }
    batch = collator([feature])
    tokenizer = collator._processor.tokenizer
    labels = batch["labels"][0]
    kept = labels[labels != -100]
    return tokenizer.decode(kept, skip_special_tokens=True).strip()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('"Concord returned to its place."', "Concord returned to its place"),
        ("I've got it, sir.", "I've got it sir"),
        ("Don't you think?", "Don't you think"),
        ("well-known cases", "well-known cases"),
    ],
)
def test_punctuation_is_dropped_and_apostrophes_survive(
    collator: Qwen3ASRDataCollator, text: str, expected: str
) -> None:
    assert supervised_text(collator, text) == expected


def test_stop_token_stays_supervised(collator: Qwen3ASRDataCollator) -> None:
    """<|im_end|> is a special token, so it must escape the punctuation sweep."""
    feature = {
        "audio": np.zeros(SAMPLE_RATE, dtype=np.float32),
        "text": "hello world.",
        "metadata": {},
    }
    batch = collator([feature])
    tokenizer = collator._processor.tokenizer
    labels = batch["labels"][0]
    assert tokenizer.eos_token_id in labels[labels != -100].tolist()


def test_every_row_keeps_a_supervised_token(collator: Qwen3ASRDataCollator) -> None:
    """An all-masked row would make the batch loss nan."""
    features = [
        {"audio": np.zeros(SAMPLE_RATE, dtype=np.float32), "text": text, "metadata": {}}
        for text in ("...", '"', "a real sentence.")
    ]
    batch = collator(features)
    for labels in batch["labels"]:
        assert (labels != -100).any()

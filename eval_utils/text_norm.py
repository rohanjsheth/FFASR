import json
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

SPELLING_PATH = Path(__file__).resolve().parent / "english_spelling.json"


@lru_cache(maxsize=1)
def get_normalizer() -> EnglishTextNormalizer:
    with SPELLING_PATH.open(encoding="utf-8") as spelling_file:
        return EnglishTextNormalizer(json.load(spelling_file))


def normalize_for_wer(text: str) -> str:
    return " ".join(get_normalizer()(text).split())


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Word-level Levenshtein distance using linear working memory."""
    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_word in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_word in enumerate(hypothesis, start=1):
            substitution_cost = int(reference_word != hypothesis_word)
            current.append(
                min(
                    previous[hypothesis_index] + 1,
                    current[hypothesis_index - 1] + 1,
                    previous[hypothesis_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]

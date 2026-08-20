"""Our WER normalizer must match openai/whisper's reference implementation exactly.

The reference lives in scripts/english.py, vendored verbatim from openai/whisper.
transformers ships the same algorithm but requires the spelling map to be passed
in, so this pins that we pass the real one -- an empty dict silently disables the
British-to-American step and still scores.
"""

import json
from pathlib import Path

import pytest

from eval_utils.text_norm import normalize_for_wer
from scripts.english import EnglishTextNormalizer as ReferenceNormalizer

REFERENCE = ReferenceNormalizer()
PREDICTIONS = sorted(Path("results/rebalanced").glob("*/predictions.jsonl"))

CASES = [
    "MM hmm uh um mhm mmm",
    "mm",
    "He realised the COLOUR of the theatre programme.",
    "Mr. Smith and Dr. Jones met St. Paul's",
    "It cost $20 million and 5 percent",
    "twenty three thousand four hundred and fifty six",
    "one oh one",
    "double oh seven",
    "three point one four",
    "The 1960s, the 274th, the 32nd",
    "I'd been there, she's gone, they'd done it",
    "won't can't let's ain't y'all wanna gotta gonna",
    "[laughter] some words (aside) more",
    "naïve café résumé",
    "1,234,567",
    "two and a half million",
    "minus five degrees",
    "twenty pounds and fifty cents",
    "ninety nine percent",
    "",
    "   ",
    "...",
    "A",
    "THE QUICK BROWN FOX JUMPED OVER THE LAZY DOG",
    # whisper_normalizer 0.1.15 rewrites this one to "because"; the reference does not.
    "The cause seems superfluous on first sight.",
    "kinda sorta dunno",
]


def reference(text: str) -> str:
    return " ".join(REFERENCE(text).split())


@pytest.mark.parametrize("text", CASES)
def test_matches_reference(text: str) -> None:
    assert normalize_for_wer(text) == reference(text)


def test_spelling_map_matches_upstream() -> None:
    ours = json.loads((Path("eval_utils") / "english_spelling.json").read_text())
    assert ours == REFERENCE.standardize_spellings.mapping


def test_spelling_map_is_actually_applied() -> None:
    assert normalize_for_wer("colour") == "color"


@pytest.mark.skipif(not PREDICTIONS, reason="no evaluation predictions on disk")
def test_matches_reference_on_recorded_predictions() -> None:
    for path in PREDICTIONS:
        for line in path.open(encoding="utf-8"):
            record = json.loads(line)
            for text in (record["reference"], record["hypothesis"]):
                assert normalize_for_wer(text) == reference(text), (path, text)

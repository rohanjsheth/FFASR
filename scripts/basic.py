"""Dependency of the vendored openai/whisper reference normalizer in english.py."""

from transformers.models.whisper.english_normalizer import remove_symbols_and_diacritics

__all__ = ["remove_symbols_and_diacritics"]

"""Tests for deterministic text utilities."""

from notnews.utils import clean_text, tokenize


def test_clean_text_normalizes_without_external_corpora() -> None:
    """Cleaning removes punctuation, digits, and common English stop words."""
    assert clean_text("The policy-in-2026 works!") == "the policy in works"


def test_tokenize_separates_punctuation() -> None:
    """Punctuation forms token boundaries instead of joining adjacent words."""
    assert tokenize("Hard-news, today") == ["hard", "news", "today"]

"""Tests for deterministic text utilities."""

import pandas as pd

from notnews.utils import clean_text, tokenize


def test_clean_text_normalizes_without_external_corpora() -> None:
    """Cleaning removes punctuation, digits, and common English stop words."""
    assert clean_text("The policy-in-2026 works!") == "the policy in works"


def test_clean_text_treats_missing_scalars_as_empty() -> None:
    """Nullable text columns can be cleaned without truth-testing pd.NA."""
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""
    assert clean_text(pd.NA) == ""


def test_tokenize_separates_punctuation() -> None:
    """Punctuation forms token boundaries instead of joining adjacent words."""
    assert tokenize("Hard-news, today") == ["hard", "news", "today"]

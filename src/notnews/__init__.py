#!/usr/bin/env python

"""
notnews: News classification library.

A simple, unified library for classifying news articles as hard/soft news
using URL patterns, machine learning models, and Large Language Models.
"""

from importlib.metadata import PackageNotFoundError, version

from .classifiers import (
    classify_by_url,
    predict_news_category,
    predict_soft_news,
)
from .llm import DEFAULT_CATEGORIES, classify_with_llm
from .utils import clean_text, fetch_web_content

try:
    # The distribution metadata is the single source; a literal here drifts
    # from pyproject.toml and from the tag, and nothing notices which is right.
    __version__ = version("notnews")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0.dev0"

__all__ = [
    "classify_by_url",
    "predict_soft_news",
    "predict_news_category",
    "classify_with_llm",
    "DEFAULT_CATEGORIES",
    "clean_text",
    "fetch_web_content",
]

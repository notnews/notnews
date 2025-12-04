#!/usr/bin/env python

"""
notnews: News classification library.

A simple, unified library for classifying news articles as hard/soft news
using URL patterns, machine learning models, and Large Language Models.
"""

# Main classification functions - new unified API
from .classifiers import (
    classify_by_url,
    pred_soft_news_uk,
    pred_soft_news_us,
    pred_what_news_us,
    predict_news_category,
    predict_soft_news,
    soft_news_url_cat_uk,
    # Legacy functions for backward compatibility
    soft_news_url_cat_us,
)
from .llm import DEFAULT_CATEGORIES
from .llm import classify_news as classify_with_llm
from .utils import clean_text, fetch_web_content

__version__ = "0.3.0"

__all__ = [
    # Modern unified API
    "classify_by_url",
    "predict_soft_news",
    "predict_news_category",
    "classify_with_llm",
    "DEFAULT_CATEGORIES",
    # Utility functions
    "clean_text",
    "fetch_web_content",
    # Legacy functions (deprecated)
    "soft_news_url_cat_us",
    "soft_news_url_cat_uk",
    "pred_soft_news_us",
    "pred_soft_news_uk",
    "pred_what_news_us",
]

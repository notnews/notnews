#!/usr/bin/env python

"""
News classification functionality for notnews.

Consolidated module providing URL pattern classification and ML model prediction
for both US and UK regions with a unified interface.
"""

import logging
import re

import pandas as pd

from ._portable_model import PortableTextClassifier, load_model

logger = logging.getLogger(__name__)

# URL patterns for different regions
URL_PATTERNS = {
    "us": {
        "hard_news": re.compile(
            r"(politi|usnews|world|national|state|elect|vote|govern|campaign|war|polic|econ|unemploy|racis|energy|abortion|educa|healthcare|immigration)"
        ),
        "soft_news": re.compile(
            r"(sport|entertainment|arts|fashion|style|lifestyle|leisure|celeb|movie|music|gossip|food|travel|horoscope|weather|gadget)"
        ),
    },
    "uk": {
        "hard_news": re.compile(
            r"(politi|world|national|uk-news|scottish-news|news-eu|state|local|elect|vote|govern|campaign|war|polic|econ|unemploy|energy|educa|healthcare|immigration)"
        ),
        "soft_news": re.compile(
            r"(sport|football|entertainment|culture|arts|fashion|style|lifestyle|life-style|leisure|celeb|movie|music|gossip|food|travel|horoscope|weather|gadget)"
        ),
    },
}

MODEL_NAMES = {
    "us": "us_soft",
    "uk": "uk_soft",
}

SOFT_NEWS_CATEGORIES = [
    "Arts",
    "Books",
    "Classifieds",
    "Dining",
    "Leisure",
    "Obits",
    "Other",
    "Real Estate",
    "Style",
    "Travel",
]


def _load_model(region: str) -> PortableTextClassifier:
    """Load the portable model for a supported region."""
    if region not in MODEL_NAMES:
        raise ValueError(f"Unsupported region: {region}. Use 'us' or 'uk'.")

    try:
        return load_model(MODEL_NAMES[region])
    except Exception as e:
        raise RuntimeError(f"Failed to load model for {region}: {e}") from e


def classify_by_url(
    df: pd.DataFrame, url_col: str = "url", region: str = "us"
) -> pd.DataFrame:
    """Classify news articles as hard/soft based on URL patterns.

    Args:
        df: DataFrame containing URLs to classify.
        url_col: Column name containing URLs. Defaults to "url".
        region: Region-specific patterns to use ("us" or "uk"). Defaults to "us".

    Returns:
        DataFrame with original columns plus:
            - hard_news: 1 if URL matches hard news patterns, None otherwise
            - soft_news: 1 if URL matches soft news patterns, None otherwise

    Raises:
        ValueError: If url_col not found in DataFrame or region not supported.

    Example:
        >>> import pandas as pd
        >>> import notnews
        >>> df = pd.DataFrame({"url": ["cnn.com/politics/election", "espn.com/sports/football"]})
        >>> result = notnews.classify_by_url(df, region="us")
        >>> print(result[["url", "hard_news", "soft_news"]])
    """
    if url_col not in df.columns:
        raise ValueError(f"Column '{url_col}' not found in DataFrame")

    if region not in URL_PATTERNS:
        raise ValueError(f"Unsupported region: {region}. Use 'us' or 'uk'.")

    patterns = URL_PATTERNS[region]
    result_df = df.copy()

    def classify_url(url):
        if pd.isna(url):
            return None, None

        hard_match = patterns["hard_news"].search(str(url))
        soft_match = patterns["soft_news"].search(str(url))

        return (1 if hard_match else None), (1 if soft_match else None)

    # Apply classification
    result_df[["hard_news", "soft_news"]] = result_df[url_col].apply(
        lambda x: pd.Series(classify_url(x))
    )

    return result_df


def predict_soft_news(
    df: pd.DataFrame, text_col: str = "text", region: str = "us"
) -> pd.DataFrame:
    """Predict soft news probability using trained ML models.

    Args:
        df: DataFrame containing article text to classify.
        text_col: Column name containing article text. Defaults to "text".
        region: Region-specific model to use ("us" or "uk"). Defaults to "us".

    Returns:
        DataFrame with original columns plus:
            - prob_soft_news_{region}: Predicted probability of soft news (0-1)

    Raises:
        ValueError: If text_col not found in DataFrame or region not supported.

    Example:
        >>> import pandas as pd
        >>> import notnews
        >>> df = pd.DataFrame({"text": ["Election coverage from Washington", "Celebrity wedding photos"]})
        >>> result = notnews.predict_soft_news(df, region="us")
        >>> print(result[["text", "prob_soft_news_us"]])
    """
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in DataFrame")

    if region not in MODEL_NAMES:
        raise ValueError(f"Unsupported region: {region}. Use 'us' or 'uk'.")

    result_df = df.copy()
    output_col = f"prob_soft_news_{region}"
    result_df[output_col] = pd.Series(pd.NA, index=result_df.index, dtype="Float64")

    # Filter to non-null text rows
    valid_rows = df[text_col].notnull()
    if not bool(valid_rows.any()):
        logger.warning("No valid text rows found")
        return result_df

    model = _load_model(region)
    text_data = result_df.loc[valid_rows, text_col].astype(str)

    try:
        y_prob = model.predict_proba(text_data)

        probabilities = pd.Series(
            y_prob[:, 1], index=result_df.index[valid_rows], dtype="Float64"
        )
        result_df.loc[valid_rows, output_col] = probabilities

    except Exception as e:
        logger.error(f"Prediction failed for {region} model: {e}")
        result_df[output_col] = pd.NA

    return result_df


def predict_news_category(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Predict detailed news categories using US model.

    Args:
        df: DataFrame containing text
        text_col: Column name containing text

    Returns:
        DataFrame with additional columns:
        - pred_category: Predicted category
        - prob_soft_news: Probability of soft news categories

    Raises:
        ValueError: If the requested text column is absent.
        RuntimeError: If the classifier or vectorizer cannot be loaded.
    """
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in DataFrame")

    result_df = df.copy()
    result_df["pred_category"] = pd.Series(pd.NA, index=result_df.index, dtype="string")
    result_df["prob_soft_news"] = pd.Series(
        pd.NA, index=result_df.index, dtype="Float64"
    )

    valid_rows = df[text_col].notnull()
    if not bool(valid_rows.any()):
        logger.warning("No valid text rows found")
        return result_df

    try:
        model = load_model("us_category")
    except Exception as e:
        raise RuntimeError(f"Failed to load US category model: {e}") from e

    text_data = result_df.loc[valid_rows, text_col].astype(str)

    try:
        y_pred = model.predict(text_data)
        y_prob = model.predict_proba(text_data)

        # Add predictions
        valid_index = result_df.index[valid_rows]
        result_df.loc[valid_rows, "pred_category"] = pd.Series(
            y_pred, index=valid_index, dtype="string"
        )

        prob_df = pd.DataFrame(
            y_prob, index=valid_index, columns=pd.Index(model.classes, dtype="string")
        )
        available_soft_categories = [
            category for category in SOFT_NEWS_CATEGORIES if category in model.classes
        ]
        result_df.loc[valid_rows, "prob_soft_news"] = prob_df[
            available_soft_categories
        ].sum(axis=1)

    except Exception as e:
        logger.error(f"Prediction failed for US category model: {e}")
        result_df.loc[valid_rows, "pred_category"] = "Other"
        result_df.loc[valid_rows, "prob_soft_news"] = 0.5

    return result_df

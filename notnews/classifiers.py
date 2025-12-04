#!/usr/bin/env python

"""
News classification functionality for notnews.

Consolidated module providing URL pattern classification and ML model prediction
for both US and UK regions with a unified interface.
"""

import logging
import re
import sys

import joblib
import pandas as pd

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

from .utils import clean_text

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

# Model configurations
MODEL_CONFIGS = {
    "us": {
        "model_file": "data/us_model/nyt_us_soft_news_classifier.joblib",
        "vectorizer_file": "data/us_model/nyt_us_soft_news_vectorizer.joblib",
        "soft_news_categories": [
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
        ],
    },
    "uk": {
        "model_file": "data/uk_model/url_uk_classifier.joblib",
        "vectorizer_file": "data/uk_model/url_uk_vectorizer.joblib",
        "soft_news_categories": None,  # UK model is binary
    },
}

# Cache for loaded models
_model_cache = {}


def _download_model_if_needed(file_path: str) -> bool:
    """Download model file if it doesn't exist locally."""
    try:
        full_path = str(files("notnews").joinpath(file_path))
        import os

        from .utils import REPO_BASE_URL, download_file

        if not os.path.exists(full_path):
            logger.info(f"Downloading {file_path}...")
            return download_file(REPO_BASE_URL + file_path, full_path)
        return True
    except Exception as e:
        logger.error(f"Failed to download {file_path}: {e}")
        return False


def _load_model(region: str) -> tuple:
    """Load model and vectorizer for given region."""
    if region in _model_cache:
        return _model_cache[region]

    if region not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported region: {region}. Use 'us' or 'uk'.")

    config = MODEL_CONFIGS[region]

    # For both US and UK models, ensure custom tokenizer is available before loading
    def custom_tokenizer(doc):
        doc = re.sub(r"\d+", "[NUM]", doc)
        return doc.split()

    # Make custom tokenizer available in __main__ for model loading
    main = sys.modules["__main__"]
    main.custom_tokenizer = custom_tokenizer

    # Download model files if needed
    if not _download_model_if_needed(config["model_file"]):
        raise RuntimeError(f"Could not download model for {region}")
    if not _download_model_if_needed(config["vectorizer_file"]):
        raise RuntimeError(f"Could not download vectorizer for {region}")

    # Load model and vectorizer
    try:
        model_path = str(files("notnews").joinpath(config["model_file"]))
        vectorizer_path = str(files("notnews").joinpath(config["vectorizer_file"]))

        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)

        _model_cache[region] = (model, vectorizer)
        return model, vectorizer

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
        RuntimeError: If model files cannot be downloaded or loaded.

    Example:
        >>> import pandas as pd
        >>> import notnews
        >>> df = pd.DataFrame({"text": ["Election coverage from Washington", "Celebrity wedding photos"]})
        >>> result = notnews.predict_soft_news(df, region="us")
        >>> print(result[["text", "prob_soft_news_us"]])
    """
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in DataFrame")

    if region not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported region: {region}. Use 'us' or 'uk'.")

    # Filter to non-null text rows
    valid_rows = df[text_col].notnull()
    if not valid_rows.any():
        logger.warning("No valid text rows found")
        result_df = df.copy()
        result_df[f"prob_soft_news_{region}"] = None
        return result_df

    # Load model
    model, vectorizer = _load_model(region)

    # Prepare text
    result_df = df.copy()
    text_data = result_df[text_col].apply(
        lambda x: clean_text(x) if pd.notnull(x) else ""
    )

    # Handle custom tokenizer for US model
    if region == "us":

        def custom_tokenizer(doc):
            doc = re.sub(r"\d+", "[NUM]", doc)
            return doc.split()

        # Monkey patch for compatibility with old models
        main = sys.modules["__main__"]
        if not hasattr(main, "custom_tokenizer"):
            main.custom_tokenizer = custom_tokenizer

    # Vectorize text
    X = vectorizer.transform(text_data.astype(str))

    # Predict
    try:
        y_prob = model.predict_proba(X)

        if region == "us":
            # US model: binary classifier (soft news probability)
            result_df[f"prob_soft_news_{region}"] = y_prob[:, 1]
        elif region == "uk":
            # UK model: binary classifier (soft news probability)
            result_df[f"prob_soft_news_{region}"] = y_prob[:, 1]

    except Exception as e:
        logger.error(f"Prediction failed for {region} model: {e}")
        result_df[f"prob_soft_news_{region}"] = None

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
    """
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in DataFrame")

    # Load US detailed model
    model_file = "data/us_model/nyt_us_classifier.joblib"
    vectorizer_file = "data/us_model/nyt_us_vectorizer.joblib"

    # Set up custom tokenizer before loading models
    def custom_tokenizer(doc):
        doc = re.sub(r"\d+", "[NUM]", doc)
        return doc.split()

    # Make custom tokenizer available in __main__ for model loading
    main = sys.modules["__main__"]
    main.custom_tokenizer = custom_tokenizer

    if not _download_model_if_needed(model_file):
        raise RuntimeError("Could not download US category model")
    if not _download_model_if_needed(vectorizer_file):
        raise RuntimeError("Could not download US category vectorizer")

    try:
        model_path = str(files("notnews").joinpath(model_file))
        vectorizer_path = str(files("notnews").joinpath(vectorizer_file))

        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)

    except Exception as e:
        raise RuntimeError(f"Failed to load US category model: {e}") from e

    # Filter valid rows
    valid_rows = df[text_col].notnull()
    if not valid_rows.any():
        logger.warning("No valid text rows found")
        result_df = df.copy()
        result_df["pred_category"] = None
        result_df["prob_soft_news"] = None
        return result_df

    # Prepare text (simple preprocessing for category model)
    result_df = df.copy()
    text_data = result_df[text_col].str.strip().str.lower().fillna("")

    # Custom tokenizer for US model
    def custom_tokenizer(doc):
        doc = re.sub(r"\d+", "[NUM]", doc)
        return doc.split()

    main = sys.modules["__main__"]
    if not hasattr(main, "custom_tokenizer"):
        main.custom_tokenizer = custom_tokenizer

    # Vectorize and predict
    try:
        X = vectorizer.transform(text_data.astype(str))
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)

        # Add predictions
        result_df["pred_category"] = y_pred

        # Calculate soft news probability
        soft_categories = MODEL_CONFIGS["us"]["soft_news_categories"]
        prob_df = pd.DataFrame(y_prob, columns=model.classes_)
        result_df["prob_soft_news"] = prob_df[soft_categories].sum(axis=1)

    except Exception as e:
        logger.error(f"Prediction failed for US category model: {e}")
        # Fallback values when model fails
        result_df["pred_category"] = "Other"
        result_df["prob_soft_news"] = 0.5

    return result_df


# Convenience functions for backward compatibility
def soft_news_url_cat_us(df: pd.DataFrame, url_col: str = "url") -> pd.DataFrame:
    """US URL classification (deprecated - use classify_by_url with region='us')."""
    return classify_by_url(df, url_col, region="us")


def soft_news_url_cat_uk(df: pd.DataFrame, url_col: str = "url") -> pd.DataFrame:
    """UK URL classification (deprecated - use classify_by_url with region='uk')."""
    return classify_by_url(df, url_col, region="uk")


def pred_soft_news_us(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """US soft news prediction (deprecated - use predict_soft_news with region='us')."""
    return predict_soft_news(df, text_col, region="us")


def pred_soft_news_uk(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """UK soft news prediction (deprecated - use predict_soft_news with region='uk')."""
    return predict_soft_news(df, text_col, region="uk")


def pred_what_news_us(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """US category prediction (deprecated - use predict_news_category)."""
    return predict_news_category(df, text_col)

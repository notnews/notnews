#!/usr/bin/env python

"""
LLM-based news classification for notnews.

Consolidated module providing news classification using Large Language Models
(Claude and OpenAI) with a simple, unified interface.
"""

import json
import logging
import os
import time
from typing import Any

import pandas as pd

# Optional imports with graceful fallbacks
try:
    from anthropic import Anthropic

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from openai import OpenAI

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

logger = logging.getLogger(__name__)

# Default news categories
DEFAULT_CATEGORIES = {
    "hard_news": {
        "description": (
            "Serious journalism covering politics, economics, international "
            "affairs, war, policy, and important societal issues"
        ),
        "examples": [
            "Election coverage",
            "Economic policy",
            "International relations",
            "War reporting",
            "Government decisions",
        ],
    },
    "soft_news": {
        "description": (
            "Entertainment, lifestyle, sports, celebrity coverage, "
            "and human interest stories"
        ),
        "examples": [
            "Celebrity gossip",
            "Sports scores",
            "Fashion trends",
            "Food reviews",
            "Travel destinations",
        ],
    },
    "opinion": {
        "description": "Editorial content, opinion pieces, analysis, and commentary",
        "examples": [
            "Editorial opinions",
            "Op-ed columns",
            "Political commentary",
            "Cultural criticism",
        ],
    },
    "feature": {
        "description": (
            "In-depth reporting, investigative journalism, and long-form articles"
        ),
        "examples": [
            "Investigative reports",
            "Long-form journalism",
            "Profile pieces",
            "Data journalism",
        ],
    },
}

DEFAULT_PROMPT_TEMPLATE = """You are a news categorization expert. Classify this article into one of the categories.

Categories:
{categories}

Article:
{content}

Respond with valid JSON:
{{
    "category": "category_name",
    "confidence": 0.95,
    "reasoning": "Brief explanation"
}}"""


def _format_categories(categories: dict) -> str:
    """Format categories for prompt."""
    formatted = []
    for name, info in categories.items():
        desc = info.get("description", "")
        examples = info.get("examples", [])
        formatted.append(f"- {name}: {desc}")
        if examples:
            formatted.append(f"  Examples: {', '.join(examples[:3])}")
    return "\n".join(formatted)


def _truncate_text(text: str, max_tokens: int = 3000) -> str:
    """Truncate text to fit token limits (rough estimation: 1 token ≈ 4 chars)."""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    # Try to truncate at sentence boundary
    last_period = truncated.rfind(".")
    if last_period > max_chars * 0.8:
        truncated = truncated[: last_period + 1]

    return truncated + "..."


def _parse_llm_response(response_text: str, categories: dict) -> dict[str, Any]:
    """Parse and validate LLM response."""
    try:
        # Clean up common formatting issues
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        result = json.loads(response_text.strip())

        # Validate structure
        if not all(k in result for k in ["category", "confidence", "reasoning"]):
            raise ValueError("Missing required fields")

        # Validate category
        if result["category"] not in categories:
            logger.warning(
                f"Invalid category: {result['category']}, using first available"
            )
            result["category"] = list(categories.keys())[0]

        # Validate confidence
        result["confidence"] = float(result["confidence"])
        if not 0 <= result["confidence"] <= 1:
            result["confidence"] = min(max(result["confidence"], 0), 1)

        return result

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.error(f"Failed to parse LLM response: {e}")

        # Fallback: try to extract category from text
        category = None
        for cat in categories.keys():
            if cat.lower() in response_text.lower():
                category = cat
                break

        return {
            "category": category or list(categories.keys())[0],
            "confidence": 0.5,
            "reasoning": "Failed to parse structured response",
        }


def _classify_with_claude(
    text: str, categories: dict, api_key: str, model: str = "claude-3-haiku-20240307"
) -> dict[str, Any]:
    """Classify text using Claude."""
    if not HAS_ANTHROPIC:
        raise ImportError(
            "anthropic package required for Claude. Install with: pip install anthropic"
        )

    client = Anthropic(api_key=api_key)
    categories_str = _format_categories(categories)
    text = _truncate_text(text)
    prompt = DEFAULT_PROMPT_TEMPLATE.format(categories=categories_str, content=text)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=500,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
            system=(
                "You are a news categorization expert. Always respond with valid JSON."
            ),
        )

        response_text = response.content[0].text
        return _parse_llm_response(response_text, categories)

    except Exception as e:
        if "rate_limit" in str(e).lower():
            logger.info("Rate limited, waiting 1 second...")
            time.sleep(1)
            # Retry once
            return _classify_with_claude(text, categories, api_key, model)
        raise


def _classify_with_openai(
    text: str, categories: dict, api_key: str, model: str = "gpt-3.5-turbo"
) -> dict[str, Any]:
    """Classify text using OpenAI."""
    if not HAS_OPENAI:
        raise ImportError(
            "openai package required for OpenAI. Install with: pip install openai"
        )

    client = OpenAI(api_key=api_key)
    categories_str = _format_categories(categories)
    text = _truncate_text(text)
    prompt = DEFAULT_PROMPT_TEMPLATE.format(categories=categories_str, content=text)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a news categorization expert. "
                        "Always respond with valid JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=500,
            response_format={"type": "json_object"}
            if "gpt-4" in model or "3.5-turbo-1106" in model
            else None,
        )

        response_text = response.choices[0].message.content
        return _parse_llm_response(response_text, categories)

    except Exception as e:
        if "rate_limit" in str(e).lower() or "429" in str(e):
            logger.info("Rate limited, waiting 1 second...")
            time.sleep(1)
            # Retry once
            return _classify_with_openai(text, categories, api_key, model)
        raise


def classify_news(
    df: pd.DataFrame,
    text_col: str = "text",
    provider: str = "claude",
    categories: dict = None,
    api_key: str = None,
    model: str = None,
) -> pd.DataFrame:
    """Classify news articles using Large Language Models.

    Args:
        df: DataFrame containing articles to classify.
        text_col: Column name containing text to classify. Defaults to "text".
        provider: LLM provider to use ("claude" or "openai"). Defaults to "claude".
        categories: Custom categories dictionary with descriptions and examples.
            Uses DEFAULT_CATEGORIES if None.
        api_key: API key for the LLM provider. Uses environment variable if None.
        model: Model name to use. Uses provider defaults if None
            (claude-3-haiku-20240307 for Claude, gpt-3.5-turbo for OpenAI).

    Returns:
        DataFrame with original columns plus:
            - llm_category: Predicted category name
            - llm_confidence: Confidence score (0-1)
            - llm_reasoning: Brief explanation of classification

    Raises:
        ValueError: If text_col not found in DataFrame or provider not supported.
        ImportError: If required LLM package not installed.

    Example:
        >>> import pandas as pd
        >>> import notnews
        >>> df = pd.DataFrame({
        ...     "text": ["Election results announced", "Celebrity wedding"]
        ... })
        >>> result = notnews.classify_news(df, provider="claude")
        >>> print(result[["text", "llm_category"]].head())
    """
    # Validate inputs
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in DataFrame")

    # Validate provider first
    if provider not in ["claude", "openai"]:
        raise ValueError(f"Unsupported provider: {provider}")

    categories = categories or DEFAULT_CATEGORIES

    # Get API key
    if not api_key:
        if provider == "claude":
            api_key = os.getenv("ANTHROPIC_API_KEY")
        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            f"No API key provided for {provider}. Set environment variable "
            f"or pass api_key parameter."
        )

    # Set default model
    if not model:
        model = "claude-3-haiku-20240307" if provider == "claude" else "gpt-3.5-turbo"

    # Choose classifier function
    if provider == "claude":
        classify_func = _classify_with_claude
    elif provider == "openai":
        classify_func = _classify_with_openai

    # Initialize result columns
    result_df = df.copy()
    result_df["llm_category"] = None
    result_df["llm_confidence"] = None
    result_df["llm_reasoning"] = None

    # Process non-null rows
    valid_rows = result_df[text_col].notnull()
    total_rows = valid_rows.sum()

    logger.info(f"Classifying {total_rows} articles using {provider} ({model})...")

    for i, idx in enumerate(result_df[valid_rows].index, 1):
        text = result_df.at[idx, text_col]

        try:
            result = classify_func(text, categories, api_key, model)
            result_df.at[idx, "llm_category"] = result["category"]
            result_df.at[idx, "llm_confidence"] = result["confidence"]
            result_df.at[idx, "llm_reasoning"] = result["reasoning"]

            if i % 10 == 0:
                logger.info(f"Processed {i}/{total_rows} articles")

        except Exception as e:
            logger.error(f"Error classifying row {idx}: {e}")
            continue

    classified_count = result_df["llm_category"].notnull().sum()
    logger.info(f"Successfully classified {classified_count}/{total_rows} articles")

    return result_df

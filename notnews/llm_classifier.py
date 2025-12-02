#!/usr/bin/env python

"""
LLM-based news classification module for notnews.

This module provides news classification using Large Language Models (LLMs)
like Claude and OpenAI, with support for custom categories and web content fetching.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

# Default news categories with descriptions and examples
DEFAULT_CATEGORIES = {
    "hard_news": {
        "description": (
            "Serious journalism covering politics, economics, international affairs, "
            "war, policy, and important societal issues"
        ),
        "examples": [
            "Election coverage and political campaigns",
            "Economic policy and financial markets",
            "International relations and diplomacy",
            "War reporting and conflict coverage",
            "Government decisions and legislation",
            "Healthcare policy and reform",
            "Climate change and environmental policy",
        ],
    },
    "soft_news": {
        "description": "Entertainment, lifestyle, sports, celebrity coverage, and human interest stories",
        "examples": [
            "Celebrity gossip and entertainment news",
            "Sports scores and athlete profiles",
            "Lifestyle tips and fashion trends",
            "Food and restaurant reviews",
            "Travel destinations and experiences",
            "Weather updates",
            "Viral videos and social media trends",
        ],
    },
    "opinion": {
        "description": "Editorial content, opinion pieces, analysis, and commentary",
        "examples": [
            "Editorial board opinions",
            "Op-ed columns",
            "Political commentary",
            "Cultural criticism",
            "Personal essays",
        ],
    },
    "feature": {
        "description": "In-depth reporting, investigative journalism, and long-form articles",
        "examples": [
            "Investigative reports",
            "Long-form journalism",
            "Profile pieces",
            "Documentary-style reporting",
            "Data journalism",
        ],
    },
}

# Default base prompt template
DEFAULT_PROMPT_TEMPLATE = (
    """You are a news categorization expert. Your task is to classify news articles """
    """into specific categories based on their content.

Categories and their definitions:
{categories}

Article to classify:
{content}

Please analyze this article and:
1. Determine which category best fits
2. Provide a confidence score from 0 to 1
3. Give a brief explanation (1-2 sentences) for your classification

Respond in the following JSON format:
{{
    "category": "category_name",
    "confidence": 0.95,
    "reasoning": "Brief explanation of why this category was chosen"
}}"""
)


class LLMNewsClassifier(ABC):
    """
    Abstract base class for LLM-based news classification.

    This follows the same pattern as existing classifiers in the notnews package,
    providing a consistent interface for different LLM providers.
    """

    def __init__(self, api_key: str | None = None):
        """Initialize the classifier with optional API key."""
        self.api_key = api_key or self._get_api_key_from_env()
        self.categories = DEFAULT_CATEGORIES.copy()
        self.prompt_template = DEFAULT_PROMPT_TEMPLATE
        self.provider = "base"

    @abstractmethod
    def _get_api_key_from_env(self) -> str | None:
        """Get API key from environment variables."""
        pass

    @abstractmethod
    def _classify_text(self, text: str, categories: dict[str, dict]) -> dict[str, Any]:
        """
        Classify a single text using the LLM.

        Args:
            text: Text to classify
            categories: Dictionary of categories with descriptions

        Returns:
            Dictionary with category, confidence, and reasoning
        """
        pass

    def validate_api_key(self) -> bool:
        """Validate that API key is present and valid."""
        if not self.api_key:
            logging.error(
                f"No API key provided for {self.provider} provider. "
                f"Please set the environment variable or pass api_key parameter."
            )
            return False
        return True

    def set_categories(self, categories: dict[str, dict]):
        """Set custom categories for classification."""
        self.categories = categories

    def set_prompt_template(self, template: str):
        """Set custom prompt template."""
        self.prompt_template = template

    def format_categories_for_prompt(self, categories: dict[str, dict]) -> str:
        """Format categories dictionary for inclusion in prompt."""
        formatted = []
        for name, info in categories.items():
            desc = info.get("description", "")
            examples = info.get("examples", [])

            formatted.append(f"- {name}: {desc}")
            if examples:
                formatted.append(f"  Examples: {', '.join(examples[:3])}")

        return "\n".join(formatted)

    def classify_dataframe(
        self,
        df: pd.DataFrame,
        col: str = "text",
        categories: dict[str, dict] | None = None,
        fetch_content: bool = False,
        url_col: str = "url",
    ) -> pd.DataFrame:
        """
        Classify news articles in a DataFrame using LLM.

        Args:
            df: DataFrame containing articles to classify
            col: Column name containing text to classify
            categories: Optional custom categories (uses defaults if None)
            fetch_content: Whether to fetch full content from URLs
            url_col: Column name containing URLs (if fetch_content=True)

        Returns:
            Original DataFrame with additional classification columns
        """
        # Validate column exists
        if col not in df.columns:
            raise Exception(f"The column {col} doesn't exist in the dataframe.")

        # Validate API key
        if not self.validate_api_key():
            # Return DataFrame unchanged if no API key
            logging.warning("Returning DataFrame unchanged due to missing API key.")
            return df

        # Use custom categories if provided
        categories_to_use = categories or self.categories

        # Initialize new columns
        df[f"llm_category_{self.provider}"] = None
        df[f"llm_confidence_{self.provider}"] = None
        df[f"llm_reasoning_{self.provider}"] = None

        if fetch_content:
            df["llm_fetched_content"] = False
            # Web content fetching will be implemented in llm_utils.py

        # Process each non-null row
        nn = df[col].notnull()
        for idx in df[nn].index:
            text = df.at[idx, col]

            # Fetch web content if requested
            if fetch_content and url_col in df.columns:
                # This will be implemented in llm_utils.py
                # For now, use the existing text
                content = text
            else:
                content = text

            try:
                # Classify the text
                result = self._classify_text(content, categories_to_use)

                # Update DataFrame
                df.at[idx, f"llm_category_{self.provider}"] = result.get("category")
                df.at[idx, f"llm_confidence_{self.provider}"] = result.get("confidence")
                df.at[idx, f"llm_reasoning_{self.provider}"] = result.get("reasoning")

            except Exception as e:
                logging.error(f"Error classifying row {idx}: {e}")
                # Leave as None/NaN for failed classifications
                continue

        return df

    def handle_rate_limit(self, retry_after: int | None = None):
        """Handle rate limiting with exponential backoff."""
        wait_time = retry_after or 1
        logging.info(f"Rate limited. Waiting {wait_time} seconds...")
        time.sleep(wait_time)


def llm_classify_news(
    df: pd.DataFrame,
    col: str = "text",
    provider: str = "claude",
    categories: dict[str, dict] | None = None,
    api_key: str | None = None,
    fetch_content: bool = False,
    url_col: str = "url",
) -> pd.DataFrame:
    """
    Classify news articles using Large Language Models.

    This function provides a unified interface for classifying news articles
    using different LLM providers (Claude, OpenAI, etc.) while maintaining
    consistency with the existing notnews API.

    Args:
        df: DataFrame containing articles to classify
        col: Column name containing text to classify (default: "text")
        provider: LLM provider to use ("claude" or "openai", default: "claude")
        categories: Optional custom categories dictionary with descriptions and examples
        api_key: Optional API key (uses environment variable if not provided)
        fetch_content: Whether to fetch full content from URLs (default: False)
        url_col: Column name containing URLs if fetch_content=True (default: "url")

    Returns:
        Original DataFrame with additional columns:
            - llm_category_{provider}: Predicted category name
            - llm_confidence_{provider}: Confidence score (0-1)
            - llm_reasoning_{provider}: Brief explanation
            - llm_fetched_content: Boolean if content was fetched (when fetch_content=True)

    Raises:
        Exception: If specified column doesn't exist in DataFrame
        ValueError: If unsupported provider is specified

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'text': ['Election results announced...', 'Celebrity spotted at...'],
        ...     'url': ['example.com/politics', 'example.com/entertainment']
        ... })
        >>> result = llm_classify_news(df, provider='claude')
        >>> print(result[['text', 'llm_category_claude']])
    """
    # Import provider-specific implementations
    # These will be created in separate files
    if provider.lower() == "claude":
        from .llm_claude import ClaudeNewsClassifier

        classifier = ClaudeNewsClassifier(api_key=api_key)
    elif provider.lower() == "openai":
        from .llm_openai import OpenAINewsClassifier

        classifier = OpenAINewsClassifier(api_key=api_key)
    else:
        raise ValueError(f"Unsupported provider: {provider}. Use 'claude' or 'openai'.")

    # Set custom categories if provided
    if categories:
        classifier.set_categories(categories)

    # Classify the DataFrame
    return classifier.classify_dataframe(
        df, col=col, categories=categories, fetch_content=fetch_content, url_col=url_col
    )

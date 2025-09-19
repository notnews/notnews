#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for LLM-based news classification.

This module tests the LLM classification functionality including
category management, API key validation, and classification results.
"""

import json
import os
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from ..llm_classifier import DEFAULT_CATEGORIES, LLMNewsClassifier, llm_classify_news


class TestLLMClassifier(unittest.TestCase):
    """Test cases for LLM news classification."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_df = pd.DataFrame(
            {
                "text": [
                    "The president announced new economic policies today.",
                    "Celebrity spotted at beach with new partner.",
                    "Editorial: Why we need healthcare reform now.",
                    "Investigation reveals corruption in city government.",
                ],
                "url": [
                    "example.com/politics",
                    "example.com/entertainment",
                    "example.com/opinion",
                    "example.com/investigation",
                ],
            }
        )

    def test_default_categories(self):
        """Test that default categories are properly defined."""
        self.assertIsInstance(DEFAULT_CATEGORIES, dict)
        self.assertIn("hard_news", DEFAULT_CATEGORIES)
        self.assertIn("soft_news", DEFAULT_CATEGORIES)
        self.assertIn("opinion", DEFAULT_CATEGORIES)
        self.assertIn("feature", DEFAULT_CATEGORIES)

        # Check structure
        for cat, info in DEFAULT_CATEGORIES.items():
            self.assertIn("description", info)
            self.assertIn("examples", info)
            self.assertIsInstance(info["examples"], list)

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_claude_api_key_from_env(self):
        """Test Claude classifier gets API key from environment."""
        try:
            from ..llm_claude import ClaudeNewsClassifier

            classifier = ClaudeNewsClassifier()
            self.assertEqual(classifier.api_key, "test-key")
        except ImportError:
            # Skip test if anthropic not installed
            self.skipTest("anthropic package not installed")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_openai_api_key_from_env(self):
        """Test OpenAI classifier gets API key from environment."""
        try:
            from ..llm_openai import OpenAINewsClassifier

            classifier = OpenAINewsClassifier()
            self.assertEqual(classifier.api_key, "test-key")
        except ImportError:
            # Skip test if openai not installed
            self.skipTest("openai package not installed")

    def test_missing_column_error(self):
        """Test error when specified column doesn't exist."""
        with self.assertRaises(Exception) as context:
            llm_classify_news(self.sample_df, col="nonexistent_column")
        self.assertIn("doesn't exist", str(context.exception))

    @patch("notnews.llm_claude.ClaudeNewsClassifier")
    def test_llm_classify_news_claude(self, mock_claude_class):
        """Test llm_classify_news function with Claude provider."""
        # Setup mock
        mock_classifier = MagicMock()
        mock_claude_class.return_value = mock_classifier
        mock_classifier.classify_dataframe.return_value = self.sample_df.copy()

        # Call function
        result = llm_classify_news(
            self.sample_df, provider="claude", api_key="test-key"
        )

        # Verify
        mock_claude_class.assert_called_once_with(api_key="test-key")
        mock_classifier.classify_dataframe.assert_called_once()
        self.assertIsInstance(result, pd.DataFrame)

    @patch("notnews.llm_openai.OpenAINewsClassifier")
    def test_llm_classify_news_openai(self, mock_openai_class):
        """Test llm_classify_news function with OpenAI provider."""
        # Setup mock
        mock_classifier = MagicMock()
        mock_openai_class.return_value = mock_classifier
        mock_classifier.classify_dataframe.return_value = self.sample_df.copy()

        # Call function
        result = llm_classify_news(
            self.sample_df, provider="openai", api_key="test-key"
        )

        # Verify
        mock_openai_class.assert_called_once_with(api_key="test-key")
        mock_classifier.classify_dataframe.assert_called_once()
        self.assertIsInstance(result, pd.DataFrame)

    def test_unsupported_provider_error(self):
        """Test error with unsupported provider."""
        with self.assertRaises(ValueError) as context:
            llm_classify_news(self.sample_df, provider="unsupported")
        self.assertIn("Unsupported provider", str(context.exception))

    def test_custom_categories(self):
        """Test using custom categories."""
        custom_categories = {
            "news": {
                "description": "General news",
                "examples": ["Breaking news", "Current events"],
            },
            "analysis": {
                "description": "In-depth analysis",
                "examples": ["Market analysis", "Political analysis"],
            },
        }

        with patch("notnews.llm_claude.ClaudeNewsClassifier") as mock_claude:
            mock_classifier = MagicMock()
            mock_claude.return_value = mock_classifier
            mock_classifier.classify_dataframe.return_value = self.sample_df.copy()

            llm_classify_news(
                self.sample_df,
                provider="claude",
                categories=custom_categories,
                api_key="test-key",
            )

            # Verify custom categories were set
            mock_classifier.set_categories.assert_called_once_with(custom_categories)


class TestLLMUtils(unittest.TestCase):
    """Test cases for LLM utility functions."""

    def test_truncate_for_token_limit(self):
        """Test text truncation for token limits."""
        from ..llm_utils import truncate_for_token_limit

        # Short text should not be truncated
        short_text = "This is a short text."
        result = truncate_for_token_limit(short_text, max_tokens=100)
        self.assertEqual(result, short_text)

        # Long text should be truncated
        long_text = "a" * 5000
        result = truncate_for_token_limit(long_text, max_tokens=100)
        self.assertLess(len(result), len(long_text))
        self.assertTrue(result.endswith("..."))

    def test_clean_text_content(self):
        """Test text cleaning functionality."""
        from ..llm_utils import clean_text_content

        # Test removing excessive whitespace
        messy_text = "This   has    too     much\n\n\nwhitespace"
        cleaned = clean_text_content(messy_text)
        self.assertNotIn("   ", cleaned)
        self.assertNotIn("\n\n\n", cleaned)

        # Test removing short lines
        text_with_short_lines = (
            "Navigation\nThis is a real sentence that should be kept.\nAd"
        )
        cleaned = clean_text_content(text_with_short_lines)
        self.assertIn("real sentence", cleaned)
        self.assertNotIn("Navigation", cleaned)

    @patch("requests.get")
    def test_fetch_web_content_success(self, mock_get):
        """Test successful web content fetching."""
        from ..llm_utils import fetch_web_content

        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"""
        <html>
            <body>
                <article>
                    <p>This is the main article content that we want to extract.</p>
                    <p>It has multiple paragraphs with useful information.</p>
                </article>
            </body>
        </html>
        """
        mock_get.return_value = mock_response

        # Test fetching
        content = fetch_web_content("https://example.com/article")

        self.assertIsNotNone(content)
        self.assertIn("main article content", content)
        self.assertIn("multiple paragraphs", content)

    @patch("requests.get")
    def test_fetch_web_content_failure(self, mock_get):
        """Test handling of web content fetch failures."""
        from ..llm_utils import fetch_web_content

        # Setup mock to raise exception
        mock_get.side_effect = Exception("Network error")

        # Test fetching
        content = fetch_web_content("https://example.com/article")

        # Should return None on failure
        self.assertIsNone(content)

    def test_prepare_content_for_llm(self):
        """Test content preparation for LLM processing."""
        from ..llm_utils import prepare_content_for_llm

        # Test basic text preparation
        text = "This is the article text."
        prepared = prepare_content_for_llm(text)
        self.assertIn(text, prepared)

        # Test with URL but no fetch
        prepared = prepare_content_for_llm(
            text, url="https://example.com", fetch_full=False
        )
        self.assertIn(text, prepared)


class TestCLIInterface(unittest.TestCase):
    """Test cases for CLI interface."""

    @patch("pandas.read_csv")
    @patch("notnews.llm_classify.llm_classify_news")
    def test_cli_basic_usage(self, mock_classify, mock_read_csv):
        """Test basic CLI usage."""
        from ..llm_classify import main

        # Setup mocks
        mock_df = pd.DataFrame({"text": ["Article 1", "Article 2"]})
        mock_read_csv.return_value = mock_df
        mock_classify.return_value = mock_df

        # Test with minimal arguments
        with patch("pandas.DataFrame.to_csv") as mock_to_csv:
            result = main(["input.csv", "-o", "output.csv"])

            self.assertEqual(result, 0)
            mock_read_csv.assert_called_once_with("input.csv")
            mock_to_csv.assert_called_once_with("output.csv", index=False)

    @patch("pandas.read_csv")
    def test_cli_missing_column(self, mock_read_csv):
        """Test CLI error handling for missing column."""
        from ..llm_classify import main

        # Setup mock with missing column
        mock_df = pd.DataFrame({"content": ["Article 1"]})
        mock_read_csv.return_value = mock_df

        # Should return error code
        result = main(["input.csv", "--text", "missing_column"])
        self.assertEqual(result, 1)

    @patch("pandas.read_csv")
    @patch("notnews.llm_classify.llm_classify_news")
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_cli_with_options(self, mock_classify, mock_read_csv):
        """Test CLI with various options."""
        from ..llm_classify import main

        # Setup mocks
        mock_df = pd.DataFrame({"text": ["Article"], "url": ["example.com"]})
        mock_read_csv.return_value = mock_df
        mock_classify.return_value = mock_df

        with patch("pandas.DataFrame.to_csv"):
            result = main(
                [
                    "input.csv",
                    "--provider",
                    "claude",
                    "--model",
                    "claude-3-opus-20240229",
                    "--fetch-content",
                    "--url-col",
                    "url",
                    "--verbose",
                ]
            )

            self.assertEqual(result, 0)

            # Verify classify was called with correct options
            mock_classify.assert_called_once()
            call_kwargs = mock_classify.call_args.kwargs
            self.assertEqual(call_kwargs["provider"], "claude")
            self.assertEqual(call_kwargs["fetch_content"], True)
            self.assertEqual(call_kwargs["url_col"], "url")


if __name__ == "__main__":
    unittest.main()

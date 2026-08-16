#!/usr/bin/env python

"""
Tests for LLM-based news classification.

This module tests the LLM classification functionality including
category management, API key validation, and classification results.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from notnews.llm import DEFAULT_CATEGORIES, classify_with_llm


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
        for _cat, info in DEFAULT_CATEGORIES.items():
            self.assertIn("description", info)
            self.assertIn("examples", info)
            self.assertIsInstance(info["examples"], list)

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @patch("notnews.llm._classify_with_claude")
    def test_claude_api_key_from_env(self, mock_claude_func):
        """Test Claude classifier gets API key from environment."""
        mock_claude_func.return_value = {
            "category": "hard_news",
            "confidence": 0.9,
            "reasoning": "Political content",
        }

        classify_with_llm(self.sample_df, provider="claude")

        self.assertEqual(mock_claude_func.call_args.args[2], "test-key")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("notnews.llm._classify_with_openai")
    def test_openai_api_key_from_env(self, mock_openai_func):
        """Test OpenAI classifier gets API key from environment."""
        mock_openai_func.return_value = {
            "category": "hard_news",
            "confidence": 0.9,
            "reasoning": "Political content",
        }

        classify_with_llm(self.sample_df, provider="openai")

        self.assertEqual(mock_openai_func.call_args.args[2], "test-key")

    def test_missing_column_error(self):
        """Test error when specified column doesn't exist."""
        with self.assertRaises(ValueError) as context:
            classify_with_llm(self.sample_df, text_col="nonexistent_column")
        self.assertIn("not found", str(context.exception))

    @patch("notnews.llm._classify_with_claude")
    def test_classify_with_llm_claude(self, mock_claude_func):
        """Test classify_with_llm function with Claude provider."""
        # Setup mock
        mock_claude_func.return_value = {
            "category": "hard_news",
            "confidence": 0.9,
            "reasoning": "Political content",
        }

        # Call function
        result = classify_with_llm(
            self.sample_df, provider="claude", api_key="test-key"
        )

        # Verify result structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("llm_category", result.columns)
        self.assertIn("llm_confidence", result.columns)
        self.assertIn("llm_reasoning", result.columns)

    @patch("notnews.llm._classify_with_openai")
    def test_classify_with_llm_openai(self, mock_openai_func):
        """Test classify_with_llm function with OpenAI provider."""
        # Setup mock
        mock_openai_func.return_value = {
            "category": "soft_news",
            "confidence": 0.8,
            "reasoning": "Entertainment content",
        }

        # Call function
        result = classify_with_llm(
            self.sample_df, provider="openai", api_key="test-key"
        )

        # Verify result structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("llm_category", result.columns)
        self.assertIn("llm_confidence", result.columns)
        self.assertIn("llm_reasoning", result.columns)

    def test_unsupported_provider_error(self):
        """Test error with unsupported provider."""
        with self.assertRaises(ValueError) as context:
            classify_with_llm(self.sample_df, provider="unsupported")
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

        with patch("notnews.llm._classify_with_claude") as mock_claude:
            mock_claude.return_value = {
                "category": "news",
                "confidence": 0.9,
                "reasoning": "General news content",
            }

            result = classify_with_llm(
                self.sample_df,
                provider="claude",
                categories=custom_categories,
                api_key="test-key",
            )

            # Verify custom categories were used
            self.assertIsInstance(result, pd.DataFrame)
            mock_claude.assert_called()

    @patch("notnews.llm._classify_with_claude")
    def test_duplicate_indices_are_classified_independently(self, mock_claude):
        """Duplicate labels do not mix the corresponding rows."""
        mock_claude.side_effect = [
            {"category": "hard_news", "confidence": 0.9, "reasoning": "First"},
            {"category": "soft_news", "confidence": 0.8, "reasoning": "Second"},
        ]
        source = pd.DataFrame(
            {"text": ["Election result", "Celebrity wedding"]}, index=[4, 4]
        )

        result = classify_with_llm(source, provider="claude", api_key="test-key")

        self.assertEqual(result["llm_category"].tolist(), ["hard_news", "soft_news"])
        self.assertEqual(
            [call.args[0] for call in mock_claude.call_args_list],
            ["Election result", "Celebrity wedding"],
        )


class TestLLMUtils(unittest.TestCase):
    """Test cases for LLM utility functions."""

    def test_truncate_text(self):
        """Test text truncation for token limits."""
        from notnews.llm import _truncate_text

        # Short text should not be truncated
        short_text = "This is a short text."
        result = _truncate_text(short_text, max_tokens=100)
        self.assertEqual(result, short_text)

        # Long text should be truncated
        long_text = "a" * 5000
        result = _truncate_text(long_text, max_tokens=100)
        self.assertLess(len(result), len(long_text))
        self.assertTrue(result.endswith("..."))

    @patch("requests.get")
    def test_fetch_web_content_success(self, mock_get):
        """Test successful web content fetching."""
        from notnews.utils import fetch_web_content

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
        mock_response.raise_for_status.return_value = None

        # Test fetching
        content = fetch_web_content("https://example.com/article")

        self.assertIsNotNone(content)
        self.assertIn("main article content", content)
        self.assertIn("multiple paragraphs", content)

    @patch("requests.get")
    def test_fetch_web_content_failure(self, mock_get):
        """Test handling of web content fetch failures."""
        from notnews.utils import fetch_web_content

        # Setup mock to raise exception
        mock_get.side_effect = Exception("Network error")

        # Test fetching
        content = fetch_web_content("https://example.com/article")

        # Should return None on failure
        self.assertIsNone(content)


# CLI tests removed - CLI structure has been updated and would need new test structure


if __name__ == "__main__":
    unittest.main()

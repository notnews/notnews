#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CLI interface for LLM-based news classification.

This module provides a command-line interface for classifying news articles
using Large Language Models (Claude and OpenAI), following the same patterns
as existing notnews classifiers.
"""

import sys
import argparse
import logging
import pandas as pd

from .llm_classifier import llm_classify_news, DEFAULT_CATEGORIES


def main(argv=sys.argv[1:]):
    """Main CLI function for LLM-based news classification."""
    title = "Classify news articles using Large Language Models (Claude/OpenAI)"
    parser = argparse.ArgumentParser(description=title)

    # Required arguments
    parser.add_argument("input", help="Input CSV file containing articles to classify")

    # Output file
    parser.add_argument(
        "-o",
        "--output",
        default="llm-news-classification-output.csv",
        help="Output file with classification results (default: llm-news-classification-output.csv)",
    )

    # Column specification
    parser.add_argument(
        "-t",
        "--text",
        default="text",
        help="Name of the column containing the text to classify (default: text)",
    )

    # LLM provider selection
    parser.add_argument(
        "-p",
        "--provider",
        default="claude",
        choices=["claude", "openai"],
        help="LLM provider to use for classification (default: claude)",
    )

    # Model selection
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help="Model to use (default: claude-3-haiku-20240307 for Claude, gpt-3.5-turbo for OpenAI)",
    )

    # API key
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for the LLM provider (uses environment variable if not provided)",
    )

    # Web content fetching
    parser.add_argument(
        "--fetch-content",
        action="store_true",
        help="Fetch full content from URLs if available",
    )

    parser.add_argument(
        "--url-col",
        default="url",
        help="Name of column containing URLs (default: url, used with --fetch-content)",
    )

    # Custom categories
    parser.add_argument(
        "--use-default-categories",
        action="store_true",
        help="Use default news categories (hard_news, soft_news, opinion, feature)",
    )

    parser.add_argument(
        "--categories",
        default=None,
        help="Path to JSON file with custom categories (overrides default categories)",
    )

    # Logging
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args(argv)

    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    logger = logging.getLogger(__name__)

    # Read input file
    try:
        df = pd.read_csv(args.input)
        logger.info(f"Loaded {len(df)} rows from {args.input}")
    except Exception as e:
        logger.error(f"Failed to read input file: {e}")
        return 1

    # Check if text column exists
    if args.text not in df.columns:
        logger.error(f"The column '{args.text}' doesn't exist in the dataframe.")
        logger.info(f"Available columns: {', '.join(df.columns)}")
        return 1

    # Load custom categories if provided
    categories = None
    if args.categories:
        try:
            import json

            with open(args.categories, "r") as f:
                categories = json.load(f)
            logger.info(f"Loaded custom categories from {args.categories}")
        except Exception as e:
            logger.error(f"Failed to load categories file: {e}")
            return 1
    elif args.use_default_categories or not categories:
        categories = DEFAULT_CATEGORIES
        logger.info("Using default news categories")

    # Determine model to use
    model = args.model
    if not model:
        if args.provider == "claude":
            model = "claude-3-haiku-20240307"
        else:
            model = "gpt-3.5-turbo"

    # Validate API key for the chosen provider
    if args.provider == "claude":
        if not args.api_key:
            import os

            if not os.getenv("ANTHROPIC_API_KEY"):
                logger.error(
                    "No API key provided. Set ANTHROPIC_API_KEY environment variable or use --api-key"
                )
                return 1
    elif args.provider == "openai":
        if not args.api_key:
            import os

            if not os.getenv("OPENAI_API_KEY"):
                logger.error(
                    "No API key provided. Set OPENAI_API_KEY environment variable or use --api-key"
                )
                return 1

    # Perform classification
    logger.info(f"Starting classification with {args.provider} ({model})...")
    try:
        result_df = llm_classify_news(
            df,
            col=args.text,
            provider=args.provider,
            categories=categories,
            api_key=args.api_key,
            fetch_content=args.fetch_content,
            url_col=args.url_col,
        )

        # Count successful classifications
        category_col = f"llm_category_{args.provider}"
        classified_count = result_df[category_col].notna().sum()
        logger.info(f"Successfully classified {classified_count}/{len(df)} articles")

        # Show category distribution
        if classified_count > 0:
            category_counts = result_df[category_col].value_counts()
            logger.info("Category distribution:")
            for cat, count in category_counts.items():
                logger.info(f"  {cat}: {count} ({count/classified_count*100:.1f}%)")

    except Exception as e:
        logger.error(f"Classification failed: {e}")
        return 1

    # Save results
    try:
        result_df.to_csv(args.output, index=False)
        logger.info(f"Saved results to {args.output}")
    except Exception as e:
        logger.error(f"Failed to save output file: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example usage of LLM-based news classification in notnews.

This example demonstrates how to classify news articles using
Large Language Models (Claude and OpenAI) with the notnews package.
"""

import pandas as pd
import os
from notnews import llm_classify_news, DEFAULT_CATEGORIES


def basic_classification_example():
    """Basic example of classifying news articles with default categories."""

    # Sample news articles
    df = pd.DataFrame(
        {
            "text": [
                "The Federal Reserve announced a quarter-point interest rate hike today, "
                "citing persistent inflation concerns. This marks the third consecutive "
                "rate increase this year.",
                "Pop star Taylor Swift was spotted having dinner with NFL player Travis "
                "Kelce at a trendy Manhattan restaurant last night. Sources say they "
                "looked very happy together.",
                "Opinion: The current approach to climate change is fundamentally flawed. "
                "We need to rethink our entire strategy if we want to avoid catastrophe.",
                "A six-month investigation by our reporters has uncovered widespread "
                "corruption in the state transportation department, with millions in "
                "contracts awarded to friends and family members of officials.",
            ],
            "source": ["Reuters", "TMZ", "NYT Opinion", "ProPublica"],
        }
    )

    print("Original DataFrame:")
    print(df[["text", "source"]].head())
    print("\n")

    # Classify using Claude (requires ANTHROPIC_API_KEY environment variable)
    if os.getenv("ANTHROPIC_API_KEY"):
        print("Classifying with Claude...")
        result_claude = llm_classify_news(df, col="text", provider="claude")

        print("\nClaude Classification Results:")
        print(result_claude[["source", "llm_category_claude", "llm_confidence_claude"]])
        print("\nReasoning:")
        for idx, row in result_claude.iterrows():
            print(f"{idx + 1}. {row['source']}: {row['llm_reasoning_claude']}")
    else:
        print("Skipping Claude (ANTHROPIC_API_KEY not set)")

    print("\n" + "=" * 60 + "\n")

    # Classify using OpenAI (requires OPENAI_API_KEY environment variable)
    if os.getenv("OPENAI_API_KEY"):
        print("Classifying with OpenAI...")
        result_openai = llm_classify_news(df, col="text", provider="openai")

        print("\nOpenAI Classification Results:")
        print(result_openai[["source", "llm_category_openai", "llm_confidence_openai"]])
        print("\nReasoning:")
        for idx, row in result_openai.iterrows():
            print(f"{idx + 1}. {row['source']}: {row['llm_reasoning_openai']}")
    else:
        print("Skipping OpenAI (OPENAI_API_KEY not set)")


def custom_categories_example():
    """Example using custom news categories."""

    # Define custom categories for specialized classification
    custom_categories = {
        "politics": {
            "description": "Political news, elections, government policy, legislation",
            "examples": [
                "Election coverage",
                "Legislative updates",
                "Political scandals",
                "Government appointments",
            ],
        },
        "business": {
            "description": "Business, finance, economy, markets, corporate news",
            "examples": [
                "Stock market updates",
                "Corporate earnings",
                "Economic indicators",
                "Business deals and mergers",
            ],
        },
        "technology": {
            "description": "Technology news, innovation, startups, digital trends",
            "examples": [
                "Product launches",
                "Tech company news",
                "AI and innovation",
                "Cybersecurity",
            ],
        },
        "entertainment": {
            "description": "Entertainment, celebrity, movies, music, culture",
            "examples": [
                "Celebrity news",
                "Movie releases",
                "Music industry",
                "Award shows",
            ],
        },
        "sports": {
            "description": "Sports news, scores, athlete profiles, competitions",
            "examples": [
                "Game results",
                "Player transfers",
                "Tournament coverage",
                "Sports analysis",
            ],
        },
    }

    # Sample articles
    df = pd.DataFrame(
        {
            "text": [
                "Apple unveiled its latest iPhone model today with advanced AI capabilities "
                "and a revolutionary camera system that promises professional-quality photos.",
                "The Lakers defeated the Celtics 112-108 in overtime, with LeBron James "
                "scoring 35 points in what many are calling one of his best performances.",
                "Congress passed the infrastructure bill with bipartisan support, allocating "
                "$1.2 trillion for roads, bridges, and broadband expansion.",
                "Netflix stock surged 15% after reporting better-than-expected subscriber "
                "growth in the third quarter, adding 8.8 million new users globally.",
                "The Academy Awards ceremony will feature a new host lineup this year, "
                "with comedians bringing fresh energy to the prestigious event.",
            ]
        }
    )

    print("Using Custom Categories for Classification")
    print("Categories:", list(custom_categories.keys()))
    print("\n")

    # Classify with custom categories
    if os.getenv("ANTHROPIC_API_KEY"):
        result = llm_classify_news(
            df, col="text", provider="claude", categories=custom_categories
        )

        print("Classification Results with Custom Categories:")
        for idx, row in result.iterrows():
            print(
                f"\n{idx + 1}. Category: {row['llm_category_claude']} "
                f"(Confidence: {row['llm_confidence_claude']:.2f})"
            )
            print(f"   Text: {row['text'][:100]}...")
            print(f"   Reasoning: {row['llm_reasoning_claude']}")
    else:
        print("Set ANTHROPIC_API_KEY to run this example")


def web_content_fetching_example():
    """Example of fetching and classifying web content."""

    # DataFrame with URLs
    df = pd.DataFrame(
        {
            "text": [
                "Breaking news about the economy",
                "Celebrity wedding announcement",
            ],
            "url": [
                "https://example.com/economy-news",
                "https://example.com/celebrity-wedding",
            ],
        }
    )

    print("Web Content Fetching Example")
    print("(Note: This example uses placeholder URLs)")
    print("\n")

    if os.getenv("ANTHROPIC_API_KEY"):
        # Classify with web content fetching enabled
        # Note: In real usage, provide actual URLs that can be fetched
        result = llm_classify_news(
            df,
            col="text",
            provider="claude",
            fetch_content=True,  # Enable web content fetching
            url_col="url",  # Specify column with URLs
        )

        print("Results with Web Content Fetching:")
        print(result[["text", "llm_category_claude", "llm_fetched_content"]])
    else:
        print("Set ANTHROPIC_API_KEY to run this example")


def batch_processing_example():
    """Example of batch processing a large dataset efficiently."""

    # Generate a larger dataset
    import random

    headlines = [
        "Government announces new policy on climate change",
        "Stock market reaches all-time high",
        "Celebrity couple announces divorce",
        "Local team wins championship",
        "New technology breakthrough in AI",
        "Election results surprise analysts",
        "Movie breaks box office records",
        "Economic indicators show growth",
        "Fashion week highlights trends",
        "Scientific discovery changes understanding",
    ]

    # Create 20 sample articles
    articles = []
    for i in range(20):
        headline = random.choice(headlines)
        articles.append(
            {"id": i + 1, "text": f"{headline}. " + "Additional context " * 20}
        )

    df = pd.DataFrame(articles)

    print(f"Batch Processing {len(df)} Articles")
    print("\n")

    if os.getenv("ANTHROPIC_API_KEY"):
        # Process in batch
        result = llm_classify_news(df, col="text", provider="claude")

        # Show summary statistics
        category_counts = result["llm_category_claude"].value_counts()
        print("Category Distribution:")
        for cat, count in category_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {cat}: {count} articles ({percentage:.1f}%)")

        # Show average confidence by category
        print("\nAverage Confidence by Category:")
        for cat in category_counts.index:
            avg_conf = result[result["llm_category_claude"] == cat][
                "llm_confidence_claude"
            ].mean()
            print(f"  {cat}: {avg_conf:.3f}")
    else:
        print("Set ANTHROPIC_API_KEY to run this example")


def main():
    """Run all examples."""
    print("=" * 60)
    print("LLM News Classification Examples")
    print("=" * 60)
    print("\n")

    print("Note: These examples require API keys to be set as environment variables:")
    print("  - ANTHROPIC_API_KEY for Claude")
    print("  - OPENAI_API_KEY for OpenAI")
    print("\n")

    print("1. BASIC CLASSIFICATION")
    print("-" * 30)
    basic_classification_example()
    print("\n")

    print("2. CUSTOM CATEGORIES")
    print("-" * 30)
    custom_categories_example()
    print("\n")

    print("3. WEB CONTENT FETCHING")
    print("-" * 30)
    web_content_fetching_example()
    print("\n")

    print("4. BATCH PROCESSING")
    print("-" * 30)
    batch_processing_example()
    print("\n")

    print("=" * 60)
    print("Examples Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

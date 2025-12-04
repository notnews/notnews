#!/usr/bin/env python

"""
Command-line interface for notnews.

Consolidated CLI providing all news classification functionality
through a modern Click-based interface.
"""

import logging
import sys
from pathlib import Path

import click
import pandas as pd

from . import classifiers, llm

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging level."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def validate_file_exists(file_path: str) -> Path:
    """Validate input file exists."""
    path = Path(file_path)
    if not path.exists():
        raise click.FileError(f"File not found: {file_path}")
    return path


def save_output(df: pd.DataFrame, output_path: str, verbose: bool = False):
    """Save DataFrame to CSV with logging."""
    df.to_csv(output_path, index=False)
    if verbose:
        click.echo(f"Saved results to {output_path}")


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(verbose):
    """notnews: News classification tools."""
    setup_logging(verbose)


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--output", "-o", default="url_classification.csv", help="Output CSV file"
)
@click.option("--url-col", default="url", help="Column containing URLs")
@click.option(
    "--region",
    type=click.Choice(["us", "uk"]),
    default="us",
    help="Region for URL patterns",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def classify_urls(input_file, output, url_col, region, verbose):
    """Classify news articles by URL patterns."""
    setup_logging(verbose)

    try:
        df = pd.read_csv(input_file)
        click.echo(f"Loaded {len(df)} rows from {input_file}")

        if url_col not in df.columns:
            available_cols = ", ".join(df.columns)
            click.echo(
                f"Error: Column '{url_col}' not found. Available: {available_cols}",
                err=True,
            )
            sys.exit(1)

        result_df = classifiers.classify_by_url(df, url_col, region)

        # Show classification summary
        hard_count = result_df["hard_news"].notna().sum()
        soft_count = result_df["soft_news"].notna().sum()
        click.echo(
            f"Classified {hard_count} hard news, {soft_count} soft news articles"
        )

        save_output(result_df, output, verbose)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--output", "-o", default="ml_prediction.csv", help="Output CSV file")
@click.option("--text-col", default="text", help="Column containing text")
@click.option(
    "--region",
    type=click.Choice(["us", "uk"]),
    default="us",
    help="Region for model selection",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def predict_ml(input_file, output, text_col, region, verbose):
    """Predict soft news using ML models."""
    setup_logging(verbose)

    try:
        df = pd.read_csv(input_file)
        click.echo(f"Loaded {len(df)} rows from {input_file}")

        if text_col not in df.columns:
            available_cols = ", ".join(df.columns)
            click.echo(
                f"Error: Column '{text_col}' not found. Available: {available_cols}",
                err=True,
            )
            sys.exit(1)

        result_df = classifiers.predict_soft_news(df, text_col, region)

        # Show prediction summary
        prob_col = f"prob_soft_news_{region}"
        valid_predictions = result_df[prob_col].notna().sum()
        avg_prob = result_df[prob_col].mean()
        click.echo(
            f"Generated {valid_predictions} predictions, "
            f"average soft news probability: {avg_prob:.3f}"
        )

        save_output(result_df, output, verbose)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--output", "-o", default="category_prediction.csv", help="Output CSV file"
)
@click.option("--text-col", default="text", help="Column containing text")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def predict_categories(input_file, output, text_col, verbose):
    """Predict detailed news categories (US model)."""
    setup_logging(verbose)

    try:
        df = pd.read_csv(input_file)
        click.echo(f"Loaded {len(df)} rows from {input_file}")

        if text_col not in df.columns:
            available_cols = ", ".join(df.columns)
            click.echo(
                f"Error: Column '{text_col}' not found. Available: {available_cols}",
                err=True,
            )
            sys.exit(1)

        result_df = classifiers.predict_news_category(df, text_col)

        # Show category distribution
        category_counts = result_df["pred_category"].value_counts()
        click.echo("Category distribution:")
        for category, count in category_counts.head(10).items():
            click.echo(f"  {category}: {count}")

        save_output(result_df, output, verbose)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--output", "-o", default="llm_classification.csv", help="Output CSV file"
)
@click.option("--text-col", default="text", help="Column containing text")
@click.option(
    "--provider",
    type=click.Choice(["claude", "openai"]),
    default="claude",
    help="LLM provider",
)
@click.option("--model", help="Specific model to use")
@click.option("--api-key", help="API key (uses environment variable if not provided)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def classify_llm(input_file, output, text_col, provider, model, api_key, verbose):
    """Classify news using Large Language Models."""
    setup_logging(verbose)

    try:
        df = pd.read_csv(input_file)
        click.echo(f"Loaded {len(df)} rows from {input_file}")

        if text_col not in df.columns:
            available_cols = ", ".join(df.columns)
            click.echo(
                f"Error: Column '{text_col}' not found. Available: {available_cols}",
                err=True,
            )
            sys.exit(1)

        # Check API key availability
        import os

        if not api_key:
            env_var = "ANTHROPIC_API_KEY" if provider == "claude" else "OPENAI_API_KEY"
            if not os.getenv(env_var):
                click.echo(
                    f"Error: No API key provided. Set {env_var} or use --api-key",
                    err=True,
                )
                sys.exit(1)

        click.echo(f"Classifying with {provider}...")
        result_df = llm.classify_news(
            df, text_col=text_col, provider=provider, model=model, api_key=api_key
        )

        # Show classification summary
        classified_count = result_df["llm_category"].notna().sum()
        avg_confidence = result_df["llm_confidence"].mean()

        click.echo(f"Successfully classified {classified_count}/{len(df)} articles")
        click.echo(f"Average confidence: {avg_confidence:.3f}")

        # Show category distribution
        if classified_count > 0:
            category_counts = result_df["llm_category"].value_counts()
            click.echo("Category distribution:")
            for category, count in category_counts.items():
                pct = count / classified_count * 100
                click.echo(f"  {category}: {count} ({pct:.1f}%)")

        save_output(result_df, output, verbose)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--output", "-o", default="combined_classification.csv", help="Output CSV file"
)
@click.option("--text-col", default="text", help="Column containing text")
@click.option("--url-col", default="url", help="Column containing URLs")
@click.option(
    "--region",
    type=click.Choice(["us", "uk"]),
    default="us",
    help="Region for classification",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def classify_all(input_file, output, text_col, url_col, region, verbose):
    """Run all classification methods (URL + ML)."""
    setup_logging(verbose)

    try:
        df = pd.read_csv(input_file)
        click.echo(f"Loaded {len(df)} rows from {input_file}")

        # Validate columns
        missing_cols = []
        if text_col not in df.columns:
            missing_cols.append(text_col)
        if url_col not in df.columns:
            missing_cols.append(url_col)

        if missing_cols:
            click.echo(f"Error: Missing columns: {', '.join(missing_cols)}", err=True)
            click.echo(f"Available columns: {', '.join(df.columns)}")
            sys.exit(1)

        # URL classification
        click.echo("Running URL pattern classification...")
        result_df = classifiers.classify_by_url(df, url_col, region)

        # ML prediction
        click.echo("Running ML prediction...")
        result_df = classifiers.predict_soft_news(result_df, text_col, region)

        # Show combined summary
        hard_count = result_df["hard_news"].notna().sum()
        soft_count = result_df["soft_news"].notna().sum()
        prob_col = f"prob_soft_news_{region}"
        ml_predictions = result_df[prob_col].notna().sum()

        click.echo(
            f"URL classification: {hard_count} hard news, {soft_count} soft news"
        )
        click.echo(f"ML predictions: {ml_predictions} articles")

        save_output(result_df, output, verbose)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()

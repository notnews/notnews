# notnews

[![CI](https://github.com/notnews/notnews/actions/workflows/ci.yml/badge.svg)](https://github.com/notnews/notnews/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/notnews.svg)](https://pypi.python.org/pypi/notnews)
[![Build and Deploy Documentation](https://github.com/notnews/notnews/actions/workflows/docs.yml/badge.svg)](https://github.com/notnews/notnews/actions/workflows/docs.yml)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97-models-yellow)](https://huggingface.co/gojiberries/notnews)
[![Downloads](https://static.pepy.tech/badge/notnews)](https://pepy.tech/project/notnews)

A Python library for classifying news articles using URL patterns, trained
US and UK models, or optional large language models.

## Features

The package supports three classification methods:

- URL pattern analysis
- trained models for US and UK news
- optional Claude or OpenAI classification

The Python API and Click CLI expose the same US and UK classifiers.

**Streamlit Demo:** https://notnews-notnews-streamlitstreamlit-app-u8j3a6.streamlit.app/

## Quick Start

### Python API

```python
import pandas as pd
import notnews

# Load your data
df = pd.read_csv("news_articles.csv")

# Method 1: URL Pattern Classification (fastest)
df_url = notnews.classify_by_url(df, url_col="url", region="us")
print(df_url[["url", "hard_news", "soft_news"]].head())

# Method 2: ML Model Prediction (most accurate)
df_ml = notnews.predict_soft_news(df, text_col="text", region="us")
print(df_ml[["text", "prob_soft_news_us"]].head())

# Method 3: LLM Classification (most flexible)
# Requires ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable
df_llm = notnews.classify_with_llm(df, text_col="text", provider="claude")
print(df_llm[["text", "llm_category", "llm_confidence"]].head())

# Detailed Categories (US only)
df_categories = notnews.predict_news_category(df, text_col="text")
print(df_categories[["text", "pred_category", "prob_soft_news"]].head())
```

### Command Line Interface

```bash
# Install the package
pip install notnews
# or with uv
uv add notnews

# URL pattern classification
notnews classify-urls articles.csv --region us --output results.csv

# ML model prediction  
notnews predict-ml articles.csv --region uk --text-col content

# LLM classification
notnews classify-llm articles.csv --provider claude --api-key your_key

# Run all methods together
notnews classify-all articles.csv --region us

# Get help
notnews --help
notnews classify-urls --help
```

## Installation

### Standard Installation

```bash
pip install notnews
```

### Fast Installation with UV

```bash
uv add notnews
```

### Requirements

- **Python:** 3.11, 3.12, or 3.13
- **Core:** pandas, numpy, pyarrow
- **Web:** requests, beautifulsoup4
- **CLI:** click 8.0+
- **Optional:** anthropic, openai (for LLM classification)

### LLM Setup

For LLM classification, set your API key:

```bash
# For Claude
export ANTHROPIC_API_KEY="your_key_here"

# For OpenAI  
export OPENAI_API_KEY="your_key_here"
```

## API Reference

### Core Functions

#### `classify_by_url(df, url_col="url", region="us")`

Classify articles using URL pattern matching.

**Args:**
- `df`: DataFrame with articles
- `url_col`: Column containing URLs
- `region`: "us" or "uk" for region-specific patterns

**Returns:** DataFrame with `hard_news` and `soft_news` columns

#### `predict_soft_news(df, text_col="text", region="us")`

Predict soft news probability using trained ML models.

**Args:**
- `df`: DataFrame with articles  
- `text_col`: Column containing article text
- `region`: "us" or "uk" for model selection

**Returns:** DataFrame with `prob_soft_news_{region}` column

#### `classify_with_llm(df, text_col="text", provider="claude", **kwargs)`

Classify articles using Large Language Models.

**Args:**
- `df`: DataFrame with articles
- `text_col`: Column containing article text  
- `provider`: "claude" or "openai"
- `categories`: Optional custom categories dict
- `api_key`: Optional API key (uses env var if not provided)

**Returns:** DataFrame with `llm_category`, `llm_confidence`, `llm_reasoning` columns

### Advanced Usage

```python
# Custom LLM categories
custom_categories = {
    "breaking": {"description": "Breaking news and urgent updates"},
    "analysis": {"description": "In-depth analysis and commentary"},
    "lifestyle": {"description": "Lifestyle and entertainment content"}
}

df_custom = notnews.classify_with_llm(
    df, 
    provider="claude",
    categories=custom_categories
)

# Fetch content from URLs
content = notnews.fetch_web_content("https://example.com")
```

## Model Information

### URL Patterns
- **US:** Politics, economics, international affairs vs. sports, entertainment, lifestyle
- **UK:** Includes UK-specific patterns like "uk-news", "scottish-news"

### ML Models
- **US:** NYT-based models trained on headline and content text
- **UK:** URL-based model trained on UK news outlets
- Runtime inference uses version-independent, typed Parquet exports of the
  original calibrated linear models.

## Data Sources

- **US Model:** Based on [NYT data](https://github.com/notnews/us_not_news)
- **UK Model:** Based on [UK news analysis](https://github.com/notnews/uk_not_news)

The fitted classifiers and vectorizers are published at
[gojiberries/notnews](https://huggingface.co/gojiberries/notnews). Notnews
downloads them from an immutable Hugging Face commit on first use. Set
`NOTNEWS_MODEL_DIR` to use an explicitly managed local copy.

## Applications

Research using notnews:
- [US Soft News Analysis](https://github.com/notnews/us_not_news)
- [UK Soft News Analysis](https://github.com/notnews/uk_not_news)

## Documentation

Full documentation: [notnews.github.io/notnews](https://notnews.github.io/notnews/)

## Contributing

We welcome contributions! Please see our [Contributor Code of Conduct](http://contributor-covenant.org/version/1/0/0/).

### Development Setup

```bash
git clone https://github.com/notnews/notnews.git
cd notnews
uv sync --all-groups
uv run pytest
```

## Authors

- Suriyan Laohaprapanon
- Gaurav Sood

## License

[MIT License](https://opensource.org/licenses/MIT)

# Overview

[![CI](https://github.com/notnews/notnews/actions/workflows/ci.yml/badge.svg)](https://github.com/notnews/notnews/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/notnews.svg)](https://pypi.python.org/pypi/notnews)
[![Build and Deploy Documentation](https://github.com/notnews/notnews/actions/workflows/docs.yml/badge.svg)](https://github.com/notnews/notnews/actions/workflows/docs.yml)
[![Downloads](https://static.pepy.tech/badge/notnews)](https://pepy.tech/project/notnews)

The **notnews** package provides classifiers for soft news based on story text and URL structure for both US and UK news media. We also provide ways to infer the 'kind' of news---Arts, Books, Science, Sports, Travel, etc.---for US news media.

## Modern Features

- **Traditional ML classifiers** - Fast, offline classification using trained models
- **LLM-based classification** - Flexible classification using Claude and OpenAI with custom categories  
- **Web content fetching** - Automatically fetch and classify content from URLs

## Getting Started

1. **{doc}`installation`** - Install notnews with optional LLM dependencies
2. **{doc}`quickstart`** - Quick examples to get you started

## API Reference

- **{doc}`api`** - Traditional ML-based classifiers (5 functions)
- **{doc}`llm_api`** - Modern LLM-based classification with Claude/OpenAI  
- **{doc}`cli`** - Command line tools for batch processing

## Applications & Data

```{include} _shared/additional.md
```

**Streamlit App:** [Try notnews online](https://notnews-notnews-streamlitstreamlit-app-u8j3a6.streamlit.app/)
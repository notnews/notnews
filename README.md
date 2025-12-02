# notnews: predict soft news using story text and the url structure

[![CI](https://github.com/notnews/notnews/actions/workflows/ci.yml/badge.svg)](https://github.com/notnews/notnews/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/notnews.svg)](https://pypi.python.org/pypi/notnews)
[![Build and Deploy Documentation](https://github.com/notnews/notnews/actions/workflows/docs.yml/badge.svg)](https://github.com/notnews/notnews/actions/workflows/docs.yml)
[![Downloads](https://static.pepy.tech/badge/notnews)](https://pepy.tech/project/notnews)

The package provides classifiers for soft news based on story text and URL structure for both US and UK news media. We also provide ways to infer the 'kind' of news---Arts, Books, Science, Sports, Travel, etc.---for US news media.

**Modern Features:**
- **Traditional ML classifiers** - Fast, offline classification using trained models
- **LLM-based classification** - Flexible classification using Claude and OpenAI with custom categories
- **Web content fetching** - Automatically fetch and classify content from URLs

**Streamlit App:** https://notnews-notnews-streamlitstreamlit-app-u8j3a6.streamlit.app/

## Quick Start

```python
>>> import pandas as pd
>>> from notnews import *

>>> # Get help
>>> help(soft_news_url_cat_us)

Help on method soft_news_url_cat in module notnews.soft_news_url_cat:

soft_news_url_cat(df, col='url') method of builtins.type instance
    Soft News Categorize by URL pattern.

    Using the URL pattern to categorize the soft/hard news of the input
    DataFrame.

    Args:
        df (:obj:`DataFrame`): Pandas DataFrame containing the URL
            column.
        col (str or int): Column's name or location of the URL in
            DataFrame (default: url).

    Returns:
        DataFrame: Pandas DataFrame with additional columns:
            - `soft_lab` set to 1 if URL match with soft news URL pattern.
            - `hard_lab` set to 1 if URL match with hard news URL pattern.

>>> # Load data
>>> df = pd.read_csv('./tests/sample_us.csv')
>>> df
            src                                                url                                               text
0             nyt  http://www.nytimes.com/2017/02/11/us/politics/...  Mr. Kushner on something of a crash course in ...
1  huffingtonpost  http://grvrdr.huffingtonpost.com/302/redirect?...  Authorities are still searching for a man susp...
2             nyt  http://www.nytimes.com/2016/09/19/us/politics/...  Photo  WASHINGTON — In releasing a far more so...
3          google  http://www.foxnews.com/world/2016/07/17/turkey...  The Turkish government on Sunday ratcheted up ...
4             nyt  http://www.nytimes.com/interactive/2016/08/29/...  NYTimes.com no longer supports Internet Explor...
5           yahoo  https://www.yahoo.com/news/pittsburgh-symphony...  PITTSBURGH AP — Pittsburgh Symphony Orchestra ...
6         foxnews  http://www.foxnews.com/politics/2016/08/13/cli...  Hillary Clintons campaign is questioning a rep...
7         foxnews  http://www.foxnews.com/us/2017/04/15/april-gir...  April the giraffe has given birth at a New Yor...
8         foxnews  http://www.foxnews.com/politics/2017/05/03/hil...  Want FOX News Halftime Report in your inbox ev...
9             nyt  http://www.nytimes.com/2016/09/06/obituaries/p...  Shes an extremely liberated woman Ms. DeCrow s...
>>>
>>> # Get the Soft News URL category
>>> df_soft_news_url_cat_us  = soft_news_url_cat_us(df, col='url')
>>> df_soft_news_url_cat_us
            src                                                url                                               text  soft_lab  hard_lab
0             nyt  http://www.nytimes.com/2017/02/11/us/politics/...  Mr. Kushner on something of a crash course in ...       NaN       1.0
1  huffingtonpost  http://grvrdr.huffingtonpost.com/302/redirect?...  Authorities are still searching for a man susp...       NaN       NaN
2             nyt  http://www.nytimes.com/2016/09/19/us/politics/...  Photo  WASHINGTON — In releasing a far more so...       NaN       1.0
3          google  http://www.foxnews.com/world/2016/07/17/turkey...  The Turkish government on Sunday ratcheted up ...       NaN       1.0
4             nyt  http://www.nytimes.com/interactive/2016/08/29/...  NYTimes.com no longer supports Internet Explor...       NaN       1.0
5           yahoo  https://www.yahoo.com/news/pittsburgh-symphony...  PITTSBURGH AP — Pittsburgh Symphony Orchestra ...       1.0       NaN
6         foxnews  http://www.foxnews.com/politics/2016/08/13/cli...  Hillary Clintons campaign is questioning a rep...       NaN       1.0
7         foxnews  http://www.foxnews.com/us/2017/04/15/april-gir...  April the giraffe has given birth at a New Yor...       NaN       NaN
8         foxnews  http://www.foxnews.com/politics/2017/05/03/hil...  Want FOX News Halftime Report in your inbox ev...       NaN       1.0
9             nyt  http://www.nytimes.com/2016/09/06/obituaries/p...  Shes an extremely liberated woman Ms. DeCrow s...       NaN       NaN
>>>
```

## Installation

Installation is as easy as typing in:

```bash
pip install notnews
```

For faster installation using UV:

```bash
uv add notnews
```

### Requirements

- Python 3.11, 3.12, or 3.13
- scikit-learn 1.3+ (models trained with sklearn 0.22+ are automatically compatible)
- pandas, numpy, nltk, and other standard scientific Python packages

### Compatibility

This package includes automatic compatibility layers to ensure models trained with older scikit-learn versions (0.22+) work seamlessly with modern scikit-learn versions (1.3-1.5). Version warnings from scikit-learn are expected and harmless.

## API

For detailed API documentation including all 6 functions (soft_news_url_cat_us, pred_soft_news_us, pred_what_news_us, soft_news_url_cat_uk, pred_soft_news_uk, llm_classify_news), command line usage, and examples, please see [project documentation](http://notnews.readthedocs.io/en/latest/).

## Underlying Data

* For more information about how to get the underlying data for UK model, see [here](https://github.com/notnews/uk_not_news). For information about the data underlying the US model, see [here](https://github.com/notnews/us_not_news)

## Applications

We use the model to estimate the supply of not news in the [US](https://github.com/notnews/us_not_news) and the [UK](https://github.com/notnews/uk_not_news).

## Documentation

For more information, please see [project documentation](http://notnews.readthedocs.io/en/latest/).

## Authors

Suriyan Laohaprapanon and Gaurav Sood

## Contributor Code of Conduct

The project welcomes contributions from everyone! In fact, it depends on it. To maintain this welcoming atmosphere, and to collaborate in a fun and productive way, we expect contributors to the project to abide by the [Contributor Code of Conduct](http://contributor-covenant.org/version/1/0/0/)

## License

The package is released under the [MIT License](https://opensource.org/licenses/MIT).
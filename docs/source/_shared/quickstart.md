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

### LLM-based Classification

```python
>>> from notnews import llm_classify_news
>>>
>>> # Modern LLM classification with Claude or OpenAI
>>> # Requires: pip install notnews[llm] and ANTHROPIC_API_KEY env var
>>> 
>>> df_sample = pd.DataFrame({
...     'text': [
...         'Federal Reserve raises interest rates by 0.25% citing inflation',
...         'Taylor Swift breaks attendance records at sold-out concert'
...     ]
... })
>>>
>>> result = llm_classify_news(df_sample, provider='claude')
>>> print(result[['text', 'llm_category_claude', 'llm_confidence_claude']])
                                               text llm_category_claude  llm_confidence_claude
0  Federal Reserve raises interest rates by 0.25...           hard_news                   0.95
1  Taylor Swift breaks attendance records at sol...           soft_news                   0.92
>>>
```
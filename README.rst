notnews: predict soft news using story text and the url structure
=================================================================

.. image:: https://travis-ci.org/notnews/notnews.svg?branch=master
    :target: https://travis-ci.org/notnews/notnews
.. image:: https://ci.appveyor.com/api/projects/status/qfvbu8h99ymtw2ub?svg=true
    :target: https://ci.appveyor.com/project/notnews/notnews
.. image:: https://img.shields.io/pypi/v/notnews.svg
    :target: https://pypi.python.org/pypi/notnews
.. image:: https://readthedocs.org/projects/notnews/badge/?version=latest
    :target: http://notnews.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://pepy.tech/badge/notnews
    :target: https://pepy.tech/project/notnews

The package provides classifiers for soft news based on the story text and the url structure for both the US and UK news media. We provide also provide a way to infer the 'kind' of news---Arts, Books, Science, Sports, Travel, etc.---for the US news media.

Quick Start
-----------

::

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
    >>> df = pd.read_csv('./notnews/tests/sample_us.csv')
    >>> df
            src_name                                                url                                               text
    0             nyt  http://www.nytimes.com/2017/02/11/us/politics/...  mr kushner someth crash cours diplomaci speak ...
    1  huffingtonpost  http://grvrdr.huffingtonpost.com/302/redirect?...  author still search man suspect shoot kill vic...
    2             nyt  http://www.nytimes.com/2016/09/19/us/politics/...  photo washington — releas far sober account do...
    3          google  http://www.foxnews.com/world/2016/07/17/turkey...  turkish govern sunday ratchet crackdown alleg ...
    4             nyt  http://www.nytimes.com/interactive/2016/08/29/...  nytimescom longer support internet explor earl...
    5           yahoo  https://www.yahoo.com/news/pittsburgh-symphony...  pittsburgh ap — pittsburgh symphoni orchestra ...
    6         foxnews  http://www.foxnews.com/politics/2016/08/13/cli...  hillari clinton campaign question report top a...
    7         foxnews  http://www.foxnews.com/us/2017/04/15/april-gir...  april giraff given birth new york zoo million ...
    8         foxnews  http://www.foxnews.com/politics/2017/05/03/hil...  want fox news halftim report inbox everi day s...
    9             nyt  http://www.nytimes.com/2016/09/06/obituaries/p...  she extrem liber woman ms decrow said intervie...
    df = pd.read_csv('./notnews/tests/sample_us.csv')

    >>> # Get the Soft News URL category
    >>> df_soft_news_url_cat_us  = soft_news_url_cat_us(df, col='url')
    >>> df_soft_news_url_cat_us
            src_name                                                url                                               text  soft_lab  hard_lab
    0             nyt  http://www.nytimes.com/2017/02/11/us/politics/...  mr kushner someth crash cours diplomaci speak ...       NaN       1.0
    1  huffingtonpost  http://grvrdr.huffingtonpost.com/302/redirect?...  author still search man suspect shoot kill vic...       NaN       NaN
    2             nyt  http://www.nytimes.com/2016/09/19/us/politics/...  photo washington — releas far sober account do...       NaN       1.0
    3          google  http://www.foxnews.com/world/2016/07/17/turkey...  turkish govern sunday ratchet crackdown alleg ...       NaN       1.0
    4             nyt  http://www.nytimes.com/interactive/2016/08/29/...  nytimescom longer support internet explor earl...       NaN       1.0
    5           yahoo  https://www.yahoo.com/news/pittsburgh-symphony...  pittsburgh ap — pittsburgh symphoni orchestra ...       1.0       NaN
    6         foxnews  http://www.foxnews.com/politics/2016/08/13/cli...  hillari clinton campaign question report top a...       NaN       1.0
    7         foxnews  http://www.foxnews.com/us/2017/04/15/april-gir...  april giraff given birth new york zoo million ...       NaN       NaN
    8         foxnews  http://www.foxnews.com/politics/2017/05/03/hil...  want fox news halftim report inbox everi day s...       NaN       1.0
    9             nyt  http://www.nytimes.com/2016/09/06/obituaries/p...  she extrem liber woman ms decrow said intervie...       NaN       NaN
    >>>


Installation
------------

Installation is as easy as typing in:

::

    pip install notnews

API
---

1. **soft_news_url_cat_us** Uses URL patterns in prominent outlets to classify the type of news. It is based on a slightly amended version of the regular expression used to classify news, and non-news in `Exposure to ideologically diverse news and opinion on Facebook <https://science.sciencemag.org/content/348/6239/1130>`__ by Bakshy, Messing, and Adamic in Science in 2015. Our only amendment: sport rather than sports. The classifier success is liable to vary over time and across outlets.

-  **Arguments:**

      -  ``df``:
      -  ``url``: column with the domain names/URLs.
         Default is ``url``

-  **What it does:**

      - converts url to lower case
      - regex

      ::

          URL containing any of the following words is classified as soft news:
          sport|entertainment|arts|fashion|style|lifestyle|leisure|celeb|movie|music|gossip|food|travel|horoscope|weather|gadget

          URL conta ining any of following words is classified as hard news:
          politi|usnews|world|national|state|elect|vote|govern|campaign|war|polic|econ|unemploy|racis|energy|abortion|educa|healthcare|immigration

-  **Output:**

      -  Given both the regex can return true, the potential set is: soft, hard, soft and hard, or empty string.
      -  By default it creates two columns, ```hard_lab``` and ```soft_lab```

-  **Examples:**

      ::

        >>> import pandas as pd
        >>> from notnews import soft_news_url_cat_us
        >>>
        >>> df = pd.DataFrame([{'url': 'http://nytimes.com/sports/'}])
        >>> df
                                url
        0  http://nytimes.com/sports/
        >>>
        >>> soft_news_url_cat_us(df)
                                url  soft_lab hard_lab
        0  http://nytimes.com/sports/         1     None


2. **pred_soft_news_us**: We use data from NY Times to train a `model <notnews/models/us_not_news_soft_news.ipynb>`__. The function
   uses the trained model to predict soft news.

-  **Arguments:**

      -  ``df``: pandas dataframe. No default.
      -  ``text``: column with the story text.

-  **Functionality:**

      -  Normalizes the text and gets the bi-grams and tri-grams
      -  Outputs calibrated probability of soft news using the trained model

-  **Output**

      -  Appends a column with probability of soft news (``prob_soft_news_us``)

-  **Examples:**

      ::

        >>> import pandas as pd
        >>> from notnews import pred_soft_news_us
        >>>
        >>> df = pd.read_csv('notnews/tests/sample_us.csv')
        >>> df
                    src                                                url                                               text
        0  huffingtonpost  http://www.huffingtonpost.com/entry/dnc-black-...  WASHINGTON ― Dont offer support for the concre...
        1  washingtonpost  https://www.washingtonpost.com/posteverything/...    Daniel Fishel for The Washington Post  When ...
        2          google   http://www.bbc.com/news/world-us-canada-37805525  Media caption Hilary Clinton said The American...
        3          google  http://www.politico.com/story/2016/08/how-big-...  Hillary Clinton just outbounced Donald Trump. ...
        4         foxnews  http://www.foxnews.com/us/2017/03/07/2-men-dre...  Two men with weapons and dressed as clowns sca...
        5             nyt  https://www.nytimes.com/2017/05/05/us/politics...  It also plays to the Democratic line of attack...
        6             nyt  http://www.nytimes.com/interactive/2016/11/08/...  Regarding Indiana Im not particularly interest...
        7  washingtonpost  https://www.washingtonpost.com/news/the-fix/wp...  The story of Humayun Khan — and his parents — ...
        8             nyt  http://www.nytimes.com/2017/02/10/opinion/when...  What will you do when terrorists attack or U.S...
        9             nyt  http://www.nytimes.com/interactive/2017/04/07/...  Tonights strike in Syria appears to be a propo...
        >>>
        >>> pred_soft_news_us(df)
        Using model data from /opt/notebooks/not_news/notnews_pub/notnews/data/us_model/nyt_us_soft_news_classifier.joblib...
        Using vectorizer data from /opt/notebooks/not_news/notnews_pub/notnews/data/us_model/nyt_us_soft_news_vectorizer.joblib...
        Loading the model and vectorizer data file...
                    src                                                url                                               text  prob_soft_news_us
        0  huffingtonpost  http://www.huffingtonpost.com/entry/dnc-black-...  WASHINGTON ― Dont offer support for the concre...           0.171090
        1  washingtonpost  https://www.washingtonpost.com/posteverything/...    Daniel Fishel for The Washington Post  When ...           0.202707
        2          google   http://www.bbc.com/news/world-us-canada-37805525  Media caption Hilary Clinton said The American...           0.107870
        3          google  http://www.politico.com/story/2016/08/how-big-...  Hillary Clinton just outbounced Donald Trump. ...           0.592496
        4         foxnews  http://www.foxnews.com/us/2017/03/07/2-men-dre...  Two men with weapons and dressed as clowns sca...           0.048173
        5             nyt  https://www.nytimes.com/2017/05/05/us/politics...  It also plays to the Democratic line of attack...           0.041430
        6             nyt  http://www.nytimes.com/interactive/2016/11/08/...  Regarding Indiana Im not particularly interest...           0.511113
        7  washingtonpost  https://www.washingtonpost.com/news/the-fix/wp...  The story of Humayun Khan — and his parents — ...           0.043211
        8             nyt  http://www.nytimes.com/2017/02/10/opinion/when...  What will you do when terrorists attack or U.S...           0.004672
        9             nyt  http://www.nytimes.com/interactive/2017/04/07/...  Tonights strike in Syria appears to be a propo...           0.090724
        >>>


3. **pred_what_news_us**: We use a `model <notnews/models/us_not_news.ipynb>`__ trained on the
    `annotated NY Times corpus <https://github.com/notnews/nytimes-corpus-extractor>`__ to predict the
     type of news---Arts, Books, Business Finance, Classifieds, Dining, Editorial, Foreign News, Health, Leisure,
     Local, National, Obits, Other, Real Estate, Science, Sports, Style, and Travel.

-  **Arguments:**

      -  ``df``: pandas dataframe. No default.
      -  ``text``: column with the story text.

-  **Functionality:**

      -  Normalizes the text and gets the bi-grams and tri-grams
      -  Outputs calibrated probability of the type of news using the trained model

-  **Output**

      -  Appends a column of predicted catetory (``pred_what_news_us``) and the columns for probability of each category.
         (``prob_*``)

-  **Examples:**

      ::

        >>> import pandas as pd
        >>> from notnews import pred_what_news_us
        >>>
        >>> df = pd.read_csv('notnews/tests/sample_us.csv')
        >>> df
                    src                                                url                                               text
        0  huffingtonpost  http://www.huffingtonpost.com/entry/dnc-black-...  WASHINGTON ― Dont offer support for the concre...
        1  washingtonpost  https://www.washingtonpost.com/posteverything/...    Daniel Fishel for The Washington Post  When ...
        2          google   http://www.bbc.com/news/world-us-canada-37805525  Media caption Hilary Clinton said The American...
        3          google  http://www.politico.com/story/2016/08/how-big-...  Hillary Clinton just outbounced Donald Trump. ...
        4         foxnews  http://www.foxnews.com/us/2017/03/07/2-men-dre...  Two men with weapons and dressed as clowns sca...
        5             nyt  https://www.nytimes.com/2017/05/05/us/politics...  It also plays to the Democratic line of attack...
        6             nyt  http://www.nytimes.com/interactive/2016/11/08/...  Regarding Indiana Im not particularly interest...
        7  washingtonpost  https://www.washingtonpost.com/news/the-fix/wp...  The story of Humayun Khan — and his parents — ...
        8             nyt  http://www.nytimes.com/2017/02/10/opinion/when...  What will you do when terrorists attack or U.S...
        9             nyt  http://www.nytimes.com/interactive/2017/04/07/...  Tonights strike in Syria appears to be a propo...
        >>>
        >>> pred_what_news_us(df)
        Using model data from /opt/notebooks/not_news/notnews_pub/notnews/data/us_model/nyt_us_classifier.joblib...
        Using vectorizer data from /opt/notebooks/not_news/notnews_pub/notnews/data/us_model/nyt_us_vectorizer.joblib...
        Loading the model and vectorizer data file...
                    src                                                url                                               text  ... prob_sports  prob_style  prob_travel
        0  huffingtonpost  http://www.huffingtonpost.com/entry/dnc-black-...  WASHINGTON ― Dont offer support for the concre...  ...    0.001404    0.003325     0.000519
        1  washingtonpost  https://www.washingtonpost.com/posteverything/...    Daniel Fishel for The Washington Post  When ...  ...    0.000359    0.004530     0.000000
        2          google   http://www.bbc.com/news/world-us-canada-37805525  Media caption Hilary Clinton said The American...  ...    0.000000    0.004288     0.000000
        3          google  http://www.politico.com/story/2016/08/how-big-...  Hillary Clinton just outbounced Donald Trump. ...  ...    0.045792    0.009661     0.000072
        4         foxnews  http://www.foxnews.com/us/2017/03/07/2-men-dre...  Two men with weapons and dressed as clowns sca...  ...    0.002152    0.002666     0.072833
        5             nyt  https://www.nytimes.com/2017/05/05/us/politics...  It also plays to the Democratic line of attack...  ...    0.000000    0.001695     0.000000
        6             nyt  http://www.nytimes.com/interactive/2016/11/08/...  Regarding Indiana Im not particularly interest...  ...    0.430395    0.010029     0.002200
        7  washingtonpost  https://www.washingtonpost.com/news/the-fix/wp...  The story of Humayun Khan — and his parents — ...  ...    0.000909    0.027996     0.000507
        8             nyt  http://www.nytimes.com/2017/02/10/opinion/when...  What will you do when terrorists attack or U.S...  ...    0.000055    0.001063     0.000000
        9             nyt  http://www.nytimes.com/interactive/2017/04/07/...  Tonights strike in Syria appears to be a propo...  ...    0.000665    0.035810     0.000372

        [10 rows x 22 columns]
        >>>


4. **soft_news_url_cat_uk** Uses URL patterns in prominent outlets to classify the type of news. It is based on a slightly amended version of the regular expression used to classify news, and non-news in Exposure to ideologically diverse news and opinion on Facebook by Bakshy, Messing, and Adamic. Science. 2015. Amendment: sport rather than sports. The classifier success is liable to vary over time and across outlets.

-  **Arguments:**

      -  ``df``: pandas dataframe. No default.
      -  ``url``: column with the domain names/URLs.
         Default is ``url``

-  **What it does:**

      - converts url to lower case
      - regex

    ::

        URL containing any of the following words is classified as soft news:
        sport|entertainment|arts|fashion|style|lifestyle|leisure|celeb|movie|music|gossip|food|travel|horoscope|weather|gadget

        URL containing any of following words is classified as hard news:
        politi|usnews|world|national|state|elect|vote|govern|campaign|war|polic|econ|unemploy|racis|energy|abortion|educa|healthcare|immigration

-  **Output:**

    -  Given both the regex can return true, the potential set is: soft, hard, soft and hard, or empty string.
    -  By default it creates two columns, ```hard_lab``` and ```soft_lab```

-  **Examples:**

    ::

        >>> import pandas as pd
        >>> from notnews import soft_news_url_cat_uk
        >>>
        >>> df = pd.DataFrame([{'url': 'https://www.theguardian.com/us/sport'}])
        >>> df
                                            url
        0  https://www.theguardian.com/us/sport
        >>>
        >>> soft_news_url_cat_uk(df)
                                            url  soft_lab hard_lab
        0  https://www.theguardian.com/us/sport         1     None
        >>>


5. **pred_soft_news_uk**: We use the `model <notnews/models/uk_not_news.ipynb>`__
       to predict soft news for UK news media.

-  **Arguments:**

    -  ``df``: pandas dataframe. No default.
    -  ``text``: column with the story text.

-  **Functionality:**

      -  Normalizes the text and gets the bi-grams and tri-grams
      -  Outputs calibrated probability of soft news using the trained model

-  **Output**

      -  Appends a column with probability of soft news (``prob_soft_news_uk``)

-  **Examples:**

    ::
        >>> import pandas as pd
        >>> from notnews import pred_soft_news_uk
        >>>
        >>> df = pd.read_csv('notnews/tests/sample_uk.csv')
        >>> df
                            src_name                                                url                                               text
        0           your local guardian  http://www.yourlocalguardian.co.uk/news/local/...  friday octob comment say speed bump dug counci...
        1          liverpool daily post  http://icliverpool.icnetwork.co.uk/0100news/03...  man shot dead takeaway four mask gunmen victim...
        2           the daily telegraph  http://telegraph.feedsportal.com/c/32726/f/534...  euromillion jackpot reach imag euromillion tic...
        3                liverpool echo  http://icliverpool.icnetwork.co.uk/0100news/03...  father one three men kill last summer riot sai...
        4           the daily telegraph  http://telegraph.feedsportal.com/c/32726/f/579...  duchess cambridg rush duchess cambridg yet nam...
        5              buckingham today  http://www.buckinghamtoday.co.uk/latest-scotti...  man accus murder nineyearold girl innoc court ...
        6        northumberland gazette  http://www.northumberlandgazette.co.uk/latest-...  singersongwrit ami winehous appeal fine mariju...
        7                  daily record  http://www.dailyrecord.co.uk/entertainment/ent...  apr beverley lyon laura sutherland former crea...
        8  international business times  http://www.ibtimes.com/articles/331256/2012042...  deep valu found small medtech jason mill sourc...
        9                the daily mail  http://www.dailymail.co.uk/news/article-252383...  ca nt afford third child foot bill key down st...
        >>>
        >>> pred_soft_news_uk(df)
        Using model data from /opt/notebooks/not_news/notnews/notnews/data/uk_model/url_uk_classifier.joblib...
        Using vectorizer data from /opt/notebooks/not_news/notnews/notnews/data/uk_model/url_uk_vectorizer.joblib...
        Loading the model and vectorizer data file...
                            src_name                                                url                                               text  prob_soft_news_uk
        0           your local guardian  http://www.yourlocalguardian.co.uk/news/local/...  friday octob comment say speed bump dug counci...           0.152979
        1          liverpool daily post  http://icliverpool.icnetwork.co.uk/0100news/03...  man shot dead takeaway four mask gunmen victim...           0.038663
        2           the daily telegraph  http://telegraph.feedsportal.com/c/32726/f/534...  euromillion jackpot reach imag euromillion tic...           0.944237
        3                liverpool echo  http://icliverpool.icnetwork.co.uk/0100news/03...  father one three men kill last summer riot sai...           0.119689
        4           the daily telegraph  http://telegraph.feedsportal.com/c/32726/f/579...  duchess cambridg rush duchess cambridg yet nam...           0.903285
        5              buckingham today  http://www.buckinghamtoday.co.uk/latest-scotti...  man accus murder nineyearold girl innoc court ...           0.049645
        6        northumberland gazette  http://www.northumberlandgazette.co.uk/latest-...  singersongwrit ami winehous appeal fine mariju...           0.070025
        7                  daily record  http://www.dailyrecord.co.uk/entertainment/ent...  apr beverley lyon laura sutherland former crea...           0.926814
        8  international business times  http://www.ibtimes.com/articles/331256/2012042...  deep valu found small medtech jason mill sourc...           0.491505
        9                the daily mail  http://www.dailymail.co.uk/news/article-252383...  ca nt afford third child foot bill key down st...           0.004905
        >>>


Command Line
------------

We also implement the scripts to process the input file in the CSV format:

1. **soft_news_url_cat_us**

    ::

        usage: soft_news_url_cat_us [-h] [-o OUTPUT] [-u URL] input

        US Soft News Category by URL pattern

        positional arguments:
        input                 Input file

        optional arguments:
        -h, --help            show this help message and exit
        -o OUTPUT, --output OUTPUT
                                Output file with category data
        -u URL, --url URL     Name or index location of column contains the domain
                                or URL (default: url)

2. **pred_soft_news_us**

    ::

        usage: pred_soft_news_us [-h] [-o OUTPUT] [-t TEXT] input

        Predict Soft News by text using NYT Soft News model

        positional arguments:
        input                 Input file

        optional arguments:
        -h, --help            show this help message and exit
        -o OUTPUT, --output OUTPUT
                                Output file with prediction data
        -t TEXT, --text TEXT  Name or index location of column contains the text
                                (default: text)

3. **pred_what_news_us**

    ::

        usage: pred_what_news_us [-h] [-o OUTPUT] [-t TEXT] input

        Predict What News by text using NYT What News model

        positional arguments:
        input                 Input file

        optional arguments:
        -h, --help            show this help message and exit
        -o OUTPUT, --output OUTPUT
                                Output file with prediction data
        -t TEXT, --text TEXT  Name or index location of column contains the text
                                (default: text)

4. **soft_news_url_cat_uk**

    ::

        usage: soft_news_url_cat_uk [-h] [-o OUTPUT] [-u URL] input

        UK Soft News Category by URL pattern

        positional arguments:
        input                 Input file

        optional arguments:
        -h, --help            show this help message and exit
        -o OUTPUT, --output OUTPUT
                                Output file with category data
        -u URL, --url URL     Name or index location of column contains the domain
                                or URL (default: url)


5. **pred_soft_news_uk**

    ::

        usage: pred_soft_news_uk [-h] [-o OUTPUT] [-t TEXT] input

        Predict Soft News by text using UK URL Soft News model

        positional arguments:
        input                 Input file

        optional arguments:
        -h, --help            show this help message and exit
        -o OUTPUT, --output OUTPUT
                                Output file with prediction data
        -t TEXT, --text TEXT  Name or index location of column contains the text
                                (default: text)

Underlying Data
---------------

* For more information about how to get the underlying data for UK model, see `here <https://github.com/notnews/uk_not_news>`__. For information about the data underlying the US model, see `here <https://github.com/notnews/us_not_news>`__

Applications
------------

We use the model to estimate the supply of not news in the `US <https://github.com/notnews/us_not_news>`__ and the `UK <https://github.com/notnews/uk_not_news>`__.

Documentation
-------------

For more information, please see `project documentation <http://notnews.readthedocs.io/en/latest/>`__.

Authors
-------

Suriyan Laohaprapanon and Gaurav Sood

Contributor Code of Conduct
---------------------------

The project welcomes contributions from everyone! In fact, it depends on
it. To maintain this welcoming atmosphere, and to collaborate in a fun
and productive way, we expect contributors to the project to abide by
the `Contributor Code of
Conduct <http://contributor-covenant.org/version/1/0/0/>`__

License
-------

The package is released under the `MIT
License <https://opensource.org/licenses/MIT>`__.

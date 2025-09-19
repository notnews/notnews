#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
import argparse
import pandas as pd
import logging


class SoftNewsURLCategorizer(object):
    hard_lab = None
    soft_lab = None

    @classmethod
    def is_hard_lab(cls, c: str) -> int | None:
        if cls.hard_lab and not pd.isnull(c):
            m = cls.hard_lab.search(c)
            return 1 if m else None
        else:
            return None

    @classmethod
    def is_soft_lab(cls, c: str) -> int | None:
        if cls.soft_lab and not pd.isnull(c):
            m = cls.soft_lab.search(c)
            return 1 if m else None
        else:
            return None

    @classmethod
    def soft_news_url_cat(cls, df: pd.DataFrame, col: str = "url") -> pd.DataFrame:
        """Categorize news articles as soft or hard based on URL patterns.

        This method analyzes URLs in the specified DataFrame column to classify
        articles as soft news (entertainment, sports, lifestyle) or hard news
        (politics, economics, world events) based on regex patterns matching
        common URL path segments.

        Args:
            df: Pandas DataFrame containing the URL column to analyze.
            col: Name of the column containing URLs or domains. Default is "url".

        Returns:
            Original DataFrame with two additional columns:
                - soft_lab: 1 if URL matches soft news patterns, None otherwise
                - hard_lab: 1 if URL matches hard news patterns, None otherwise

        Raises:
            Exception: If the specified column doesn't exist in the DataFrame.

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({'url': ['example.com/sports/game', 'example.com/politics/election']})
            >>> result = soft_news_url_cat(df, 'url')
            >>> print(result[['url', 'soft_lab', 'hard_lab']])
        """

        if col and (col not in df.columns):
            raise Exception(f"The column {col} doesn't exist in the dataframe.")

        nn = df[col].notnull()
        if df[nn].shape[0] == 0:
            return df

        df["__url"] = df[col].str.strip().str.lower()

        df["soft_lab"] = df["__url"].apply(lambda c: cls.is_soft_lab(c))
        df["hard_lab"] = df["__url"].apply(lambda c: cls.is_hard_lab(c))

        del df["__url"]

        return df

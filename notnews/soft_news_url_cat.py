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
    def is_hard_lab(cls, c):
        if cls.hard_lab and not pd.isnull(c):
            m = cls.hard_lab.search(c)
            return 1 if m else None
        else:
            return None

    @classmethod
    def is_soft_lab(cls, c):
        if cls.soft_lab and not pd.isnull(c):
            m = cls.soft_lab.search(c)
            return 1 if m else None
        else:
            return None

    @classmethod
    def soft_news_url_cat(cls, df, col='url'):
        """Soft News Categorize by URL pattern.

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

        """

        if col not in df.columns:
            logging.info(f"No column {col} in the DataFrame")
            return df

        nn = df[col].notnull()
        if df[nn].shape[0] == 0:
            return df

        df['__url'] = df[col].str.strip()
        df['__url'] = df['__url'].str.lower()

        df['soft_lab'] = df['__url'].apply(lambda c: cls.is_soft_lab(c))
        df['hard_lab'] = df['__url'].apply(lambda c: cls.is_hard_lab(c))
  
        del df['__url']

        return df

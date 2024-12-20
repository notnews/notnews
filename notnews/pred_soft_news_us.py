#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
import argparse
import joblib
import pandas as pd
import logging 

from sklearn.feature_extraction.text import TfidfTransformer

from .pred_soft_news import SoftNewsModel
from .normalizer import clean_text


def custom_tokenizer(doc):
    doc = re.sub('\d+', '[NUM]', doc)
    return doc.split()


class USSoftNewsModel(SoftNewsModel):
    MODELFN = "data/us_model/nyt_us_classifier.joblib"
    VECTFN = "data/us_model/nyt_us_vectorizer.joblib"
    SOFT_NEWS_CAT = ['Arts', 'Books', 'Classifieds', 'Dining', 'Leisure',
                     'Obits', 'Other', 'Real Estate', 'Style', 'Travel']
    vect = None
    model = None

    @classmethod
    def pred_soft_news_us(cls, df: pd.DataFrame, col='text', latest:bool=False):
        """Predict Soft News by the text using NYT Soft News model.

        Using the NYT Soft News model to predict the soft news of the input
        DataFrame.

        Args:
            df (:obj:`DataFrame`): Pandas DataFrame containing the text
                column.
            col (str or int): Column's name or location of the text in
                DataFrame. (default: text)
            latest (bool): Download latest model data from the server.
                (default: False)

        Returns:
            DataFrame: Pandas DataFrame with additional columns:
                - `prob_soft_news_us` the predict result
        """

        if col and (col not in df.columns):
            raise Exception(f"The column {col} doesn't exist in the dataframe.")

        nn = df[col].notnull()
        if df[nn].shape[0] == 0:
            return df

        df['__text'] = df[col].str.strip().str.lower()

        if cls.model is None:
            # FIXME: hook up custom_tokenizer to __main__
            cls.main = sys.modules['__main__']
            cls.main.custom_tokenizer = custom_tokenizer
            cls.model, cls.vect = cls.load_model_data(latest)

        X = cls.vect.transform(df['__text'].astype(str))
        tfidf = TfidfTransformer()
        X = tfidf.fit_transform(X)
        y_pred = cls.model.predict(X)
        y_prob = cls.model.predict_proba(X)

        # take out temporary working columns
        del df['__text']
        df['pred_what_news_us'] = y_pred
        prob_df = pd.DataFrame(y_prob)
        columns = []
        for c in cls.model.classes_:
            columns.append(c)
        prob_df.columns = columns
        df['prob_soft_news_us'] = prob_df[cls.SOFT_NEWS_CAT].sum(axis=1)

        return df


class USSoftNewsModel2(SoftNewsModel):
    MODELFN = "data/us_model/nyt_us_soft_news_classifier.joblib"
    VECTFN = "data/us_model/nyt_us_soft_news_vectorizer.joblib"
    vect = None
    model = None

    @classmethod
    def pred_soft_news_us(cls, df:pd.DataFrame, col='text', latest:bool=False):
        """Predict Soft News by the text using NYT Soft News model.

        Using the NYT Soft News model to predict the soft news of the input
        DataFrame.

        Args:
            df (:obj:`DataFrame`): Pandas DataFrame containing the text
                column.
            col (str or int): Column name of the column containing the text. (default: text)
            latest (bool): Download latest model data from the server.
                (default: False)

        Returns:
            DataFrame: Pandas DataFrame with additional columns:
                - `prob_soft_news_us` the predicted probability
        """

        if col and (col not in df.columns):
            raise Exception(f"The column {col} doesn't exist in the dataframe.")

        nn = df[col].notnull()
        if df[nn].shape[0] == 0:
            return df

        df['__text'] = df[col].apply(lambda c: clean_text(c))

        if cls.model is None:
            # FIXME: hook up custom_tokenizer to __main__
            cls.main = sys.modules['__main__']
            cls.main.custom_tokenizer = custom_tokenizer
            cls.model, cls.vect = cls.load_model_data(latest)

        X = cls.vect.transform(df['__text'].astype(str))
        tfidf = TfidfTransformer()
        X = tfidf.fit_transform(X)

        y_prob = cls.model.predict_proba(X)
        df['prob_soft_news_us'] = y_prob[:, 1]

        # take out temporary working columns
        del df['__text']

        return df

pred_soft_news_us = USSoftNewsModel2.pred_soft_news_us


def main(argv=sys.argv[1:]):
    title = 'Predict Soft News by text using NYT Soft News model'
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('input', default=None,
                        help='Input file')
    parser.add_argument('-o', '--output', default='pred-soft-news-us-output.csv',
                        help='Output file with prediction data')
    parser.add_argument('-t', '--text', default='text',
                        help='Name of the column containing the text (default: text)')

    args = parser.parse_args(argv)

    print(args)

    df = pd.read_csv(args.input)

    if args.text and (args.text not in df.columns):
        raise Exception(f"The column {args.text} doesn't exist in the dataframe.")

    rdf = pred_soft_news_us(df, args.text)

    print(f"Saving output to file: {args.output}")
    rdf.to_csv(args.output, index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())

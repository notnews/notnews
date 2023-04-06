#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
import joblib
import pandas as pd
import logging

from sklearn.feature_extraction.text import TfidfTransformer

from .pred_soft_news import SoftNewsModel
from .utils import column_exists
from .normalizer import clean_text


class UKSoftNewsModel(SoftNewsModel):
    MODELFN = "data/uk_model/url_uk_classifier.joblib"
    VECTFN = "data/uk_model/url_uk_vectorizer.joblib"
    vect = None
    model = None

    @classmethod
    def pred_soft_news_uk(cls, df:pd.DataFrame, col:str='text', latest:bool=False):
        """Predict Soft News by the text using UK URL Soft News model.

        Using the URL Soft News model to predict the soft news of the input
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
                - `prob_soft_news_uk` is the prediction probability.

        """

        if column_exists(df, col):
            return df

        nn = df[col].notnull()
        if df[nn].shape[0] == 0:
            return df

        df['__text'] = df[col].apply(lambda c: clean_text(c))

        if cls.model is None:
            cls.model, cls.vect = cls.load_model_data(latest)

        X = cls.vect.transform(df['__text'].astype(str))
        tfidf = TfidfTransformer()
        X = tfidf.fit_transform(X)
        y_prob = cls.model.predict_proba(X)
        df['prob_soft_news_uk'] = y_prob[:, 1]

        # take out temporary working columns
        del df['__text']

        return df


pred_soft_news_uk = UKSoftNewsModel.pred_soft_news_uk


def main(argv=sys.argv[1:]):
    title = 'Predict Soft News by text using UK URL Soft News model'
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('input', default=None,
                        help='Input file')
    parser.add_argument('-o', '--output', default='pred-soft-news-uk-output.csv',
                        help='Output file with prediction data')
    parser.add_argument('-t', '--text', default='text',
                        help='Name of column containing the text (default: text)')

    args = parser.parse_args(argv)

    print(args)

    df = pd.read_csv(args.input)

    if not column_exists(df, args.text):
        return -1

    rdf = pred_soft_news_uk(df, args.text)

    print(f"Saving output to file: {args.output}")
    rdf.to_csv(args.output, index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())

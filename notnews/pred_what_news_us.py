#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
import argparse
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfTransformer

from .pred_soft_news import SoftNewsModel
from .utils import column_exists, fixup_columns
from .normalizer import clean_text


def custom_tokenizer(doc):
    doc = re.sub('\d+', '[NUM]', doc)
    return doc.split()


class USWhatNewsModel(SoftNewsModel):
    MODELFN = "data/us_model/nyt_us_classifier.joblib"
    VECTFN = "data/us_model/nyt_us_vectorizer.joblib"
    vect = None
    model = None

    @classmethod
    def pred_what_news_us(cls, df, col='text', latest=False):
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
                - `pred_what_news_us` the predict result
                - prob_(`Arts`, `Books`, `Business Finance`, `Classifieds`,
                  `Dining`, `Editorial`, `Foreign News`, `Health`,
                  `Leisure`, `Local`, `National`, `Obits`, `Other`,
                  `Real Estate`, `Science`, `Sports`, `Style`, `Travel`)
                   are the prediction probability for each category.
        """

        if col not in df.columns:
            print("No column `{0!s}` in the DataFrame".format(col))
            return df

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
        y_pred = cls.model.predict(X)
        y_prob = cls.model.predict_proba(X)

        # take out temporary working columns
        del df['__text']

        df['pred_what_news_us'] = y_pred
        prob_df = pd.DataFrame(y_prob)
        columns = []
        for c in cls.model.classes_:
            columns.append('prob_' + c.replace(' ', '_').lower())
        prob_df.columns = columns
        df.reset_index(inplace=True, drop=True)
        df = pd.concat([df, prob_df], axis=1)

        return df


pred_what_news_us = USWhatNewsModel.pred_what_news_us


def main(argv=sys.argv[1:]):
    title = 'Predict What News by text using NYT What News model'
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('input', default=None,
                        help='Input file')
    parser.add_argument('-o', '--output', default='pred-what-news-us-output.csv',
                        help='Output file with prediction data')
    parser.add_argument('-t', '--text', default='text',
                        help='Name or index location of column contains '
                             'the text (default: text)')

    args = parser.parse_args(argv)

    print(args)

    if not args.text.isdigit():
        df = pd.read_csv(args.input)
    else:
        df = pd.read_csv(args.input, header=None)
        args.text = int(args.text)

    if not column_exists(df, args.text):
        return -1

    rdf = pred_what_news_us(df, args.text)

    print("Saving output to file: `{0:s}`".format(args.output))
    rdf.columns = fixup_columns(rdf.columns)
    rdf.to_csv(args.output, index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())

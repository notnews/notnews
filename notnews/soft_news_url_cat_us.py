#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
import argparse
import pandas as pd

from .soft_news_url_cat import SoftNewsURLCategorizer


class USSoftNewsURLCat(SoftNewsURLCategorizer):
    soft_lab = re.compile('(sport|entertainment|arts|fashion|style|lifestyle|leisure|celeb|movie|music|gossip|food|travel|horoscope|weather|gadget)')
    hard_lab = re.compile('(politi|usnews|world|national|state|elect|vote|govern|campaign|war|polic|econ|unemploy|racis|energy|abortion|educa|healthcare|immigration)')

soft_news_url_cat_us = USSoftNewsURLCat.soft_news_url_cat


def main(argv=sys.argv[1:]):
    title = 'US Soft News Category by URL pattern'
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('input', default=None,
                        help='Input file')
    parser.add_argument('-o', '--output', default='soft-news-url-cat-us-output.csv',
                        help='Output file with category data')
    parser.add_argument('-u', '--url', default='url',
                        help='Name of the column containing the domain or the URL (default: url)')

    args = parser.parse_args(argv)

    print(args)

    df = pd.read_csv(args.input)
    
    if args.url and (args.url not in df.columns):
        raise Exception(f"The column {args.url} doesn't exist in the dataframe.")

    rdf = soft_news_url_cat_us(df, args.url)

    print(f"Saving output to file: {args.output}")
    rdf.to_csv(args.output, index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())

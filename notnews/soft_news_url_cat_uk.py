#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
import argparse
import pandas as pd

from .soft_news_url_cat import SoftNewsURLCategorizer

from .utils import column_exists, fixup_columns


class UKSoftNewsURLCat(SoftNewsURLCategorizer):
    hard_lab = re.compile('(politi|world|national|uk-news|scottish-news|news-eu|state|local|elect|vote|govern|campaign|war|polic|econ|unemploy|energy|educa|healthcare|immigration)')
    soft_lab = re.compile('(sport|football|entertainment|culture|arts|fashion|style|lifestyle|life-style|leisure|celeb|movie|music|gossip|food|travel|horoscope|weather|gadget)')

soft_news_url_cat_uk = UKSoftNewsURLCat.soft_news_url_cat


def main(argv=sys.argv[1:]):
    title = 'UK Soft News Category by URL pattern'
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('input', default=None,
                        help='Input file')
    parser.add_argument('-o', '--output', default='soft-news-url-cat-uk-output.csv',
                        help='Output file with category data')
    parser.add_argument('-u', '--url', default='url',
                        help='Name or index location of column contains '
                             'the domain or URL (default: url)')

    args = parser.parse_args(argv)

    print(args)

    if not args.url.isdigit():
        df = pd.read_csv(args.input)
    else:
        df = pd.read_csv(args.input, header=None)
        args.last = int(args.last)

    if not column_exists(df, args.url):
        return -1

    rdf = soft_news_url_cat_uk(df, args.url)

    print("Saving output to file: `{0:s}`".format(args.output))
    rdf.columns = fixup_columns(rdf.columns)
    rdf.to_csv(args.output, index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())

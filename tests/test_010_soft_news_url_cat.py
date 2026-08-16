#!/usr/bin/env python

"""
Tests for Soft News categorize by URL pattern

"""

import unittest

import pandas as pd

from notnews import classify_by_url


class TestSoftNewsURLCat(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_soft_news_url_cat_us(self):
        df = pd.read_parquet("tests/sample_us.parquet")
        odf = classify_by_url(df, "url", region="us")
        self.assertIn("soft_news", odf.columns)
        self.assertIn("hard_news", odf.columns)

    def test_soft_news_url_cat_uk(self):
        df = pd.read_parquet("tests/sample_uk.parquet")
        odf = classify_by_url(df, "url", region="uk")
        self.assertIn("soft_news", odf.columns)
        self.assertIn("hard_news", odf.columns)


if __name__ == "__main__":
    unittest.main()

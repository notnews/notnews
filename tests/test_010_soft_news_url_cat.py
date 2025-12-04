#!/usr/bin/env python

"""
Tests for Soft News categorize by URL pattern

"""

import unittest

import pandas as pd

from notnews import soft_news_url_cat_uk, soft_news_url_cat_us


class TestSoftNewsURLCat(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_soft_news_url_cat_us(self):
        df = pd.read_csv("tests/sample_us.csv")
        odf = soft_news_url_cat_us(df, "url")
        self.assertIn("soft_news", odf.columns)
        self.assertIn("hard_news", odf.columns)

    def test_soft_news_url_cat_uk(self):
        df = pd.read_csv("tests/sample_uk.csv")
        odf = soft_news_url_cat_uk(df, "url")
        self.assertIn("soft_news", odf.columns)
        self.assertIn("hard_news", odf.columns)


if __name__ == "__main__":
    unittest.main()

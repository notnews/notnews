#!/usr/bin/env python

"""
Tests for Soft News categorize by URL pattern

"""

import unittest

import pandas as pd
import pytest

from notnews import predict_news_category, predict_soft_news


@pytest.mark.live
class TestPredSoftNews(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_pred_soft_news_us(self):
        df = pd.read_parquet("tests/sample_us.parquet")
        odf = predict_soft_news(df, "text", region="us")
        self.assertIn("prob_soft_news_us", odf.columns)

    def test_pred_what_news_us(self):
        df = pd.read_parquet("tests/sample_us.parquet")
        odf = predict_news_category(df, "text")
        self.assertIn("pred_category", odf.columns)

    def test_pred_soft_news_uk(self):
        df = pd.read_parquet("tests/sample_uk.parquet")
        odf = predict_soft_news(df, "text", region="uk")
        self.assertIn("prob_soft_news_uk", odf.columns)


if __name__ == "__main__":
    unittest.main()

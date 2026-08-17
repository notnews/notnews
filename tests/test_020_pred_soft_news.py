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
        probabilities = odf["prob_soft_news_us"]
        self.assertTrue(probabilities.notna().all())
        self.assertTrue(probabilities.between(0, 1).all())

    def test_pred_what_news_us(self):
        df = pd.read_parquet("tests/sample_us.parquet")
        odf = predict_news_category(df, "text")
        self.assertTrue(odf["pred_category"].notna().all())
        self.assertFalse(odf["pred_category"].eq("Other").all())
        self.assertTrue(odf["prob_soft_news"].notna().all())
        self.assertTrue(odf["prob_soft_news"].between(0, 1).all())

    def test_pred_soft_news_uk(self):
        df = pd.read_parquet("tests/sample_uk.parquet")
        odf = predict_soft_news(df, "text", region="uk")
        probabilities = odf["prob_soft_news_uk"]
        self.assertTrue(probabilities.notna().all())
        self.assertTrue(probabilities.between(0, 1).all())


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for Soft News categorize by URL pattern

"""

import os
import shutil
import unittest
import pandas as pd
from notnews import pred_soft_news_us, pred_what_news_us, pred_soft_news_uk
from . import capture


class TestPredSoftNews(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_pred_soft_news_us(self):
        df = pd.read_csv('notnews/tests/sample_us.csv')
        odf = pred_soft_news_us(df, 'text')
        self.assertIn('prob_soft_news_us', odf.columns)

    def test_pred_what_news_us(self):
        df = pd.read_csv('notnews/tests/sample_us.csv')
        odf = pred_what_news_us(df, 'text')
        self.assertIn('pred_what_news_us', odf.columns)

    def test_pred_soft_news_uk(self):
        df = pd.read_csv('notnews/tests/sample_uk.csv')
        odf = pred_soft_news_uk(df, 'text')
        self.assertIn('prob_soft_news_uk', odf.columns)


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for Soft News categorize by URL pattern

"""

import os
import shutil
import unittest
import pandas as pd
from notnews import soft_news_url_cat_us, soft_news_url_cat_uk
from . import capture


class TestSoftNewsURLCat(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_soft_news_url_cat_us(self):
        df = pd.read_csv('notnews/tests/sample_us.csv')
        odf = soft_news_url_cat_us(df, 'url')
        self.assertIn('soft_lab', odf.columns)

    def test_soft_news_url_cat_uk(self):
        df = pd.read_csv('notnews/tests/sample_uk.csv')
        odf = soft_news_url_cat_uk(df, 'url')
        self.assertIn('soft_lab', odf.columns)


if __name__ == '__main__':
    unittest.main()

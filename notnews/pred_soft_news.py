#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import sys
import argparse
import joblib
import logging

from pkg_resources import resource_filename
from .utils import download_file, REPO_BASE_URL

class SoftNewsModel(object):
    MODELFN = None
    VECTFN = None

    @classmethod
    def load_model_data(cls, latest:bool=False):
        if cls.MODELFN:
            model_fn =  resource_filename(__name__, cls.MODELFN)
            path = os.path.dirname(model_fn)
            if not os.path.exists(path):
                os.makedirs(path)
            if not os.path.exists(model_fn) or latest:
                logging.info(f"Downloading model data from the server ({model_fn})...")
                if not download_file(REPO_BASE_URL + cls.MODELFN, model_fn):
                    logging.info("ERROR: Cannot download model data file")
                    return None, None
            else:
                print(f"Using model data from {model_fn}...")
        if cls.VECTFN:
            vect_fn =  resource_filename(__name__, cls.VECTFN)
            path = os.path.dirname(vect_fn)
            if not os.path.exists(path):
                os.makedirs(path)
            if not os.path.exists(vect_fn) or latest:
                logging.info(f"Downloading vectorizer data from the server ({vect_fn})...")
                if not download_file(REPO_BASE_URL + cls.VECTFN, vect_fn):
                    logging.info("ERROR: Cannot download vectorizer data file")
                    return None, None
            else:
                logging.info(f"Using vectorizer data from {vect_fn}...")

        logging.info("Loading the model and vectorizer data file...")

        model = joblib.load(model_fn)
        vect =joblib.load(vect_fn)

        return model, vect

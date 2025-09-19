# -*- coding: utf-8 -*-

import sys
import os
from os import path
import pandas as pd
from tqdm import tqdm
import requests
import logging

REPO_BASE_URL = (
    os.environ.get("NOTNEWS_DATA_URL")
    or "https://github.com/notnews/notnews/raw/master/notnews/"
)


def find_ngrams(vocab: list, text: str, n: int):
    """Find and return list of the index of n-grams in the vocabulary list.

    Generate the n-grams of the specific text, find them in the vocabulary list
    and return the list of index have been found.

    Args:
        vocab (:obj:`list`): Vocabulary list.
        text (str): Input text
        n (int): N-grams

    Returns:
        list: List of the index of n-grams in the vocabulary list.

    """

    wi = []

    a = zip(*[text[i:] for i in range(n)])
    for i in a:
        w = "".join(i)
        try:
            idx = vocab.index(w)
        except Exception as e:
            idx = 0
        wi.append(idx)
    return wi


def get_app_file_path(app_name, filename):
    user_dir = path.expanduser("~")
    app_data_dir = path.join(user_dir, "." + app_name)
    if not path.exists(app_data_dir):
        os.makedirs(app_data_dir)
    file_path = path.join(app_data_dir, filename)
    return file_path


def download_file(url, target):

    if "NOTNEWS_AUTH_TOKEN" in os.environ:
        auth_token = os.environ["NOTNEWS_AUTH_TOKEN"]
        headers = {"Authorization": "token {0!s}".format(auth_token)}
    else:
        headers = {}

    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True, headers=headers)

    if r.status_code == 200:
        chunk_size = 64 * 1024
        # Total size in bytes.
        total_size = int(r.headers.get("content-length", 0)) / chunk_size

        total_size += 1

        with open(target, "wb") as f:
            for data in tqdm(
                r.iter_content(chunk_size),
                total=total_size,
                unit_scale=chunk_size / 1024,
                unit="KB",
            ):
                f.write(data)
        return True
    else:
        logging.error(f"ERROR: status_code={r.status_code}")
        return False

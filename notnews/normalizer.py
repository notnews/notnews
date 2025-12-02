import logging
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")

stemmer = PorterStemmer()


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text: str) -> list[str]:
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems


def clean_text(text: str) -> str:
    try:
        text = re.sub(r"\d+", "", text)
        text = text.lower()
        text = "".join([ch for ch in text if ch not in string.punctuation])
        tokens = nltk.word_tokenize(text)
        tokens = [t for t in tokens if t not in stopwords.words("english")]
        stems = stem_tokens(tokens, stemmer)
        return " ".join(stems)
    except Exception as e:
        logging.exception(f"An error occurred: {e}")
        return text

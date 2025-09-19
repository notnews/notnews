import nltk

nltk.download("stopwords")
nltk.download("punkt")
import re
import string
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import logging
from typing import List

stemmer = PorterStemmer()


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text: str) -> List[str]:
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

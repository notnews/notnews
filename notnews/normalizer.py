import nltk
nltk.download('stopwords')
nltk.download('punkt')
import re
import string
from nltk.corpus import stopwords
from nltk import word_tokenize          
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer() 


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems


def clean_text(text):
    try:
        text = re.sub('\d+', '', text)
        text = text.lower()
        text = "".join([ch for ch in text if ch not in string.punctuation])
        tokens = nltk.word_tokenize(text)
        tokens = [t for t in tokens if t not in stopwords.words('english')]
        stems = stem_tokens(tokens, stemmer)
        return ' '.join(stems)
    except Exception as e:
        print('ERROR:', e)
        return text

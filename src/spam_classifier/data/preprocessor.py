import re
import string

import spacy
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

nlp = spacy.blank("en")
_lemmatizer = WordNetLemmatizer()
_stop_words = set(ENGLISH_STOP_WORDS)


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_with_spacy(text: str) -> str:
    tokens = [tok.text for tok in nlp(text) if not tok.is_space]
    tokens = [w for w in tokens if w not in _stop_words and len(w) > 2]
    tokens = [_lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)


def preprocess(text: str) -> str:
    return tokenize_with_spacy(clean_text(text))

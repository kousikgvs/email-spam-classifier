from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer(max_features: int = 30000, ngram_range: tuple = (1, 2)) -> TfidfVectorizer:
    return TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)


def fit_transform(vectorizer: TfidfVectorizer, texts):
    return vectorizer.fit_transform(texts)


def transform(vectorizer: TfidfVectorizer, texts):
    return vectorizer.transform(texts)

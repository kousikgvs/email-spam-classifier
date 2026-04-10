from sklearn.feature_extraction.text import TfidfVectorizer

from spam_classifier.data.preprocessor import preprocess


def predict_message(message: str, model, vectorizer: TfidfVectorizer) -> str:
    processed = preprocess(message)
    vectorized = vectorizer.transform([processed])
    pred = model.predict(vectorized)[0]
    return "Spam" if pred == 1 else "Ham"


def predict_batch(messages: list[str], model, vectorizer: TfidfVectorizer) -> list[str]:
    processed = [preprocess(m) for m in messages]
    vectorized = vectorizer.transform(processed)
    preds = model.predict(vectorized)
    return ["Spam" if p == 1 else "Ham" for p in preds]

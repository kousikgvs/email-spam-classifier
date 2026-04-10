import logging

from sklearn.model_selection import train_test_split

from spam_classifier.data.loader import load_data
from spam_classifier.data.preprocessor import preprocess
from spam_classifier.features.vectorizer import build_vectorizer, fit_transform, transform
from spam_classifier.models.evaluator import log_report, plot_confusion_matrix
from spam_classifier.models.trainer import select_best_model, tune_model
from spam_classifier.utils.io import save_artifact

logger = logging.getLogger(__name__)


def run_train_pipeline(
    data_path: str,
    artifact_dir: str,
    test_size: float = 0.2,
    random_state: int = 42,
    max_features: int = 30000,
    ngram_range: tuple = (1, 2),
    cv: int = 3,
) -> None:
    logger.info("Loading data from %s", data_path)
    df = load_data(data_path)

    logger.info("Preprocessing text...")
    df["tokenized_text"] = df["text"].apply(preprocess)

    X = df["tokenized_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info("Train: %d  Test: %d", len(X_train), len(X_test))

    logger.info("Fitting TF-IDF vectorizer...")
    vectorizer = build_vectorizer(max_features=max_features, ngram_range=ngram_range)
    X_train_vec = fit_transform(vectorizer, X_train)
    X_test_vec = transform(vectorizer, X_test)

    logger.info("Selecting best baseline model...")
    best_model_name = select_best_model(X_train_vec, y_train)

    logger.info("Tuning %s with GridSearchCV...", best_model_name)
    best_model = tune_model(best_model_name, X_train_vec, y_train, cv=cv)

    logger.info("Evaluating on test set...")
    final_preds = best_model.predict(X_test_vec)
    log_report(y_test, final_preds)
    plot_confusion_matrix(y_test, final_preds)

    save_artifact(best_model, artifact_dir, "model.joblib")
    save_artifact(vectorizer, artifact_dir, "vectorizer.joblib")
    logger.info("Artifacts saved to %s", artifact_dir)

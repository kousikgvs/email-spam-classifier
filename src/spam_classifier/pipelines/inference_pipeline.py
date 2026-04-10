import logging

from spam_classifier.models.predictor import predict_batch, predict_message
from spam_classifier.utils.io import load_artifact

logger = logging.getLogger(__name__)


def load_inference_artifacts(artifact_dir: str) -> tuple:
    model = load_artifact(artifact_dir, "model.joblib")
    vectorizer = load_artifact(artifact_dir, "vectorizer.joblib")
    logger.info("Inference artifacts loaded from %s", artifact_dir)
    return model, vectorizer


def run_inference(message: str, artifact_dir: str) -> str:
    model, vectorizer = load_inference_artifacts(artifact_dir)
    result = predict_message(message, model, vectorizer)
    logger.info("Message: %r -> Prediction: %s", message, result)
    return result


def run_batch_inference(messages: list[str], artifact_dir: str) -> list[str]:
    model, vectorizer = load_inference_artifacts(artifact_dir)
    results = predict_batch(messages, model, vectorizer)
    for msg, label in zip(messages, results):
        logger.info("Message: %r -> Prediction: %s", msg, label)
    return results

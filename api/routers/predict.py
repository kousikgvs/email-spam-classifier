import logging

from fastapi import APIRouter, HTTPException

from api.schemas import PredictRequest, PredictResponse
from spam_classifier.pipelines.inference_pipeline import load_inference_artifacts
from spam_classifier.models.predictor import predict_message

logger = logging.getLogger(__name__)

router = APIRouter()

_model = None
_vectorizer = None


def get_artifacts():
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded.")
    return _model, _vectorizer


def init_artifacts(artifact_dir: str) -> None:
    global _model, _vectorizer
    _model, _vectorizer = load_inference_artifacts(artifact_dir)


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    model, vectorizer = get_artifacts()
    try:
        label = predict_message(request.message, model, vectorizer)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc
    return PredictResponse(message=request.message, prediction=label)

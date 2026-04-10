import os

from contextlib import asynccontextmanager
from fastapi import FastAPI

from api.routers.predict import init_artifacts, router as predict_router
from spam_classifier.utils.logging import setup_logging

setup_logging()

ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "models/artifacts")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_artifacts(ARTIFACT_DIR)
    yield


app = FastAPI(
    title="Spam Classifier API",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(predict_router, prefix="/api/v1", tags=["predict"])

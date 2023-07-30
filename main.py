"""Nuha API main module."""
import os
from fastapi import FastAPI
from fastapi.requests import Request

from src.interface import PredictionRequest, PredictionResponse
from src.model import Nuha

app = FastAPI(
    title="Nuha API",
    description="API to serve ML model for hate-speech classification",
)


@app.on_event("startup")
def on_startup():
    """Load model on startup."""
    model_path = os.environ.get("MODEL_PATH")
    model_version = os.environ.get("MODEL_VERSION")
    app.state.model = Nuha(model_path=model_path, model_version=model_version)


@app.post("/predict")
def predict(
    request: Request, comments: list[PredictionRequest]
) -> list[PredictionResponse]:
    """Classify comments into hatespeech or not."""
    model = request.app.state.model

    results = model.predict([c.comment for c in comments])

    return [
        {"label": r[0], "score": r[1], "model_version": model.model_version}
        for r in results
    ]

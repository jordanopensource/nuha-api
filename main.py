"""Nuha API main module."""
import os
from fastapi import FastAPI
from fastapi.requests import Request
import huggingface_hub

from src.interface import PredictionRequest, PredictionResponse
from src.model import Nuha, PredictionResult

app = FastAPI(
    title="Nuha API",
    description="API to serve ML model for hate-speech classification",
)


@app.on_event("startup")
def on_startup():
    """Load model on startup."""
    model_path = os.environ.get("MODEL_PATH")
    model_version = os.environ.get("MODEL_VERSION")
    huggingface_token = os.environ.get("HUGGINGFACE_TOKEN")

    huggingface_hub.login(token=huggingface_token)
    app.state.model = Nuha(model_path=model_path, model_version=model_version)

@app.get('/healthcheck')
def healthcheck(request: Request):
  return 'A healthy response'

@app.post("/predict")
def predict(
    request: Request, comments: list[PredictionRequest]
) -> list[PredictionResponse]:
    """Classify comments into hatespeech or not."""
    model = request.app.state.model

    results: list[PredictionResult]
    results = model.predict([c.comment for c in comments])

    return [
        {
            "label": result.label,
            "score": result.score,
            "model_version": model.model_version,
            "comment": comment.comment,
            "post": comment.post,
        }
        for result, comment in zip(results, comments)
    ]

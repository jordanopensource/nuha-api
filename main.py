"""Nuha API main module."""
import os
from fastapi import FastAPI, Header, Response, status
from fastapi.requests import Request
import huggingface_hub
from typing import Annotated

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
    huggingface_token = os.environ.get("HUGGINGFACE_TOKEN")

    huggingface_hub.login(token=huggingface_token)
    app.state.model = Nuha(model_path=model_path, model_version=model_version)


@app.post("/predict")
async def predict(
        request: Request, authorization: Annotated[str | None, Header()], comments: list[PredictionRequest],
        response: Response
) -> list[PredictionResponse]:
    """check for valid API token first."""
    if len(authorization) == 0 or authorization[len("Bearer "):] != os.environ.get("API_TOKEN"):
        response.status_code = status.HTTP_401_UNAUTHORIZED
        return []

    """Classify comments into hatespeech or not."""
    model = request.app.state.model

    results = model.predict([c.comment for c in comments])

    return [
        {"label": r[0], "score": r[1], "model_version": model.model_version}
        for r in results
    ]

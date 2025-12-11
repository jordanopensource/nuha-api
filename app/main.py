"""
EgyNuha API - Egyptian-Arabic Text Classification Service.

Provides endpoints for single and batch text classification.
"""

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, Body
from pydantic import BaseModel, Field

from app.classifier import get_classification, get_classifications_batch, load_model

import os

_MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "1000"))

# --- Schemas ---


class ClassifyRequest(BaseModel):
    """Request body for single text classification."""

    text: Annotated[str, Field(min_length=1, description="Arabic text to classify")]

    model_config = {"json_schema_extra": {"examples": [{"text": "نص باللهجة المصرية"}]}}


class ClassifyResponse(BaseModel):
    """Response for single text classification."""

    sub_class: str = Field(description="Classification sub_class")
    main_class: str = Field(description="Classification main_class")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score (0-1)")


class BatchClassifyRequest(BaseModel):
    """Request body for batch text classification."""

    texts: Annotated[
        list[str],
        Field(min_length=1, max_length=_MAX_BATCH_SIZE, description="List of texts to classify"),
    ]

    model_config = {
        "json_schema_extra": {"examples": [{"texts": ["نص باللهجة المصرية 1", "نص باللهجة المصرية 2"]}]}
    }


class BatchClassifyResponse(BaseModel):
    """Response for batch text classification."""

    results: list[ClassifyResponse] = Field(
        description="Classification results in same order as input"
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str


# --- Application ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Loads the model on startup so first request isn't slow.
    """
    load_model()
    yield
    # Shutdown: Cleanup resources here if needed


app = FastAPI(
    title="EgyNuha API",
    description="Egyptian-Arabic Text Classification API",
    version="0.1.0",
    lifespan=lifespan,
)


# --- Endpoints ---


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Check if the service is healthy."""
    return HealthResponse(status="healthy")


@app.post("/classify", response_model=ClassifyResponse, tags=["Classification"])
async def classify_single(
    request: Annotated[ClassifyRequest, Body()],
) -> ClassifyResponse:
    """
    Classify a single text.

    Returns the predicted sub_class, main_class, and confidence score.
    """
    result = await get_classification(request.text)
    return ClassifyResponse(sub_class=result.sub_class, main_class=result.main_class, confidence=result.confidence)


@app.post("/classify/batch", response_model=BatchClassifyResponse, tags=["Classification"])
async def classify_batch(
    request: Annotated[BatchClassifyRequest, Body()],
) -> BatchClassifyResponse:
    """
    Classify multiple texts in a single request.

    Returns classification results in the same order as the input texts.
    More efficient than multiple single requests for large volumes.
    """
    results = await get_classifications_batch(request.texts)
    return BatchClassifyResponse(
        results=[ClassifyResponse(sub_class=r.sub_class, main_class=r.main_class, confidence=r.confidence) for r in results]
    )

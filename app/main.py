"""
EgyNuha API - Egyptian-Arabic Text Classification Service.

Provides endpoints for single and batch text classification.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Body, FastAPI, Query
from pydantic import BaseModel, Field

from app.classifier import (
    Language,
    get_classification,
    get_classifications_batch,
    load_model,
    shutdown_executor,
)


# -----------------------------------------------------------------------------
# Configuration from environment variables
# -----------------------------------------------------------------------------

MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "1000"))

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------


class ClassifyRequest(BaseModel):
    """Request body for single text classification."""

    text: Annotated[str, Field(
        min_length=1, description="Arabic text to classify")]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"text": "نص باللهجة المصرية"},
            ]
        }
    }


class ClassifyResponse(BaseModel):
    """Response for single text classification."""

    is_valid: bool = Field(
        description="Whether the input text was valid for classification")
    sub_class: str | None = Field(
        description="Classification sub_class (null if invalid)")
    main_class: str | None = Field(
        description="Classification main_class (null if invalid)")
    confidence: float | None = Field(
        ge=0.0, le=1.0, description="Confidence score 0-1 (null if invalid)"
    )


class BatchClassifyRequest(BaseModel):
    """Request body for batch text classification."""

    texts: Annotated[
        list[str],
        Field(
            min_length=1,
            max_length=MAX_BATCH_SIZE,
            description=f"List of texts to classify (max {MAX_BATCH_SIZE})",
        ),
    ]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"texts": ["نص باللهجة المصرية 1", "نص باللهجة المصرية 2"]},
            ]
        }
    }


class BatchClassifyResponse(BaseModel):
    """Response for batch text classification."""

    results: list[ClassifyResponse] = Field(
        description="Classification results in same order as input"
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str


class ErrorResponse(BaseModel):
    """Error response schema."""

    detail: str


# -----------------------------------------------------------------------------
# Application
# -----------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Loads the model on startup so first request isn't slow.
    Shuts down executor on shutdown.
    """
    logger.info("Starting EgyNuha API...")
    try:
        load_model()
        logger.info("Model loaded, API ready to serve requests")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    yield
    logger.info("Shutting down EgyNuha API...")
    shutdown_executor()


app = FastAPI(
    title="EgyNuha API",
    description="Egyptian-Arabic Text Classification API",
    version="0.2.0",
    lifespan=lifespan,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Check if the service is healthy."""
    return HealthResponse(status="healthy")


@app.post("/classify", response_model=ClassifyResponse, tags=["Classification"])
async def classify_single(
    request: Annotated[ClassifyRequest, Body()],
    lang: Annotated[
        Language,
        Query(description="Response language: 'ar' for Arabic, 'en' for English"),
    ] = "ar",
) -> ClassifyResponse:
    """
    Classify a single text.

    Returns the predicted sub_class, main_class, and confidence score.

    The `lang` query parameter controls the language of the returned labels:
    - `ar`: Arabic labels (default)
    - `en`: English labels
    """
    result = await get_classification(request.text, lang=lang)
    return ClassifyResponse(
        is_valid=result.is_valid,
        sub_class=result.sub_class,
        main_class=result.main_class,
        confidence=result.confidence,
    )


@app.post("/classify/batch", response_model=BatchClassifyResponse, tags=["Classification"])
async def classify_batch(
    request: Annotated[BatchClassifyRequest, Body()],
    lang: Annotated[
        Language,
        Query(description="Response language: 'ar' for Arabic, 'en' for English"),
    ] = "ar",
) -> BatchClassifyResponse:
    """
    Classify multiple texts in a single request.

    Returns classification results in the same order as the input texts.
    More efficient than multiple single requests for large volumes.

    The `lang` query parameter controls the language of the returned labels:
    - `ar`: Arabic labels (default)
    - `en`: English labels
    """
    results = await get_classifications_batch(request.texts, lang=lang)
    return BatchClassifyResponse(
        results=[
            ClassifyResponse(
                is_valid=r.is_valid,
                sub_class=r.sub_class,
                main_class=r.main_class,
                confidence=r.confidence,
            )
            for r in results
        ]
    )

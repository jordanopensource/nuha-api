"""
Nuha API - Text Classification Service.

Provides endpoints for single and batch text classification.
Each instance serves a single dialect, configured via the DIALECT env var.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.classifier import (
    DIALECT,
    DIALECT_NAMES,
    LANGUAGE_ALIASES,
    LANGUAGE_NAMES,
    SUPPORTED_LANGUAGES,
    VALID_DIALECTS,
    InferenceTimeoutError,
    ServiceOverloadedError,
    get_cache_stats,
    get_classification,
    get_classifications_batch,
    load_model,
    normalize_lang,
    shutdown_executor,
)


# -----------------------------------------------------------------------------
# Configuration from environment variables
# -----------------------------------------------------------------------------


def _parse_bounded_int(name: str, default: int, lo: int, hi: int) -> int:
    """Parse an integer env var, requiring it to fall within [lo, hi] (raises if not)."""
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from None
    if value < lo or value > hi:
        raise RuntimeError(
            f"{name} must be between {lo} and {hi}, got {value}")
    return value


MAX_BATCH_SIZE = _parse_bounded_int("MAX_BATCH_SIZE", 1000, 1, 10000)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------


class ClassifyRequest(BaseModel):
    """Request body for single text classification."""

    text: Annotated[str, Field(
        min_length=1, max_length=50000, description="Text to classify")]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"text": "نص للتصنيف"},
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
        list[Annotated[str, Field(max_length=50000)]],
        Field(
            min_length=1,
            max_length=MAX_BATCH_SIZE,
            description=f"List of texts to classify (max {MAX_BATCH_SIZE})",
        ),
    ]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"texts": ["نص للتصنيف 1", "نص للتصنيف 2"]},
            ]
        }
    }


class BatchClassifyResponse(BaseModel):
    """Response for batch text classification."""

    results: list[ClassifyResponse] = Field(
        description="Classification results in same order as input"
    )


class CacheStats(BaseModel):
    """Inference cache statistics."""

    size: int = Field(description="Current number of cached entries")
    maxsize: int = Field(description="Maximum cache capacity")
    hits: int = Field(description="Total cache hits")
    misses: int = Field(description="Total cache misses")
    hit_rate: float = Field(description="Cache hit rate (0.0 to 1.0)")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    cache: CacheStats | None = Field(
        default=None, description="Inference cache statistics")


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
    logger.info("Starting Nuha API (dialect=%s)...", DIALECT)
    try:
        load_model()
        logger.info("Model loaded for dialect '%s', API ready", DIALECT)
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        raise
    yield
    logger.info("Shutting down Nuha API...")
    shutdown_executor()


_ENABLE_DOCS = os.getenv("DISABLE_DOCS") is None
# Cache stats are operational internals; off by default so /health (which nginx
# serves unauthenticated) doesn't leak traffic/hit-rate signals. Opt in for
# trusted/internal monitoring via EXPOSE_CACHE_STATS.
_EXPOSE_CACHE_STATS = os.getenv("EXPOSE_CACHE_STATS") is not None

# API parameter descriptions are built from the dialect config so the docs never
# drift from the configured dialects/languages and carry no hardcoded knowledge.
_DIALECT_PARAM_DESC = (
    "Input dialect: "
    + ", ".join(f"'{code}' ({name})" for code, name in DIALECT_NAMES.items())
    + ". Defaults to this instance's configured dialect."
)


def _lang_forms(code: str) -> str:
    """Render a language as its canonical code plus aliases, e.g. "'ara'/'ar' (Arabic)"."""
    codes = [
        code, *sorted(a for a, c in LANGUAGE_ALIASES.items() if c == code)]
    return "/".join(f"'{c}'" for c in codes) + f" ({LANGUAGE_NAMES[code]})"


_LANG_PARAM_DESC = (
    "Response language (controls label language, not which model runs). "
    "Accepts the canonical ISO 639-3 code or a two-letter alias. "
    "Supported by this dialect: "
    + ", ".join(_lang_forms(code)
                for code in LANGUAGE_NAMES if code in SUPPORTED_LANGUAGES)
    + "."
)
# Every code accepted for this dialect (canonical plus aliases), for error messages.
_ACCEPTED_LANGS = sorted(
    set(SUPPORTED_LANGUAGES) | {
        a for a, c in LANGUAGE_ALIASES.items() if c in SUPPORTED_LANGUAGES}
)

app = FastAPI(
    title="Nuha API",
    description="Text Classification API",
    version="0.3.0",
    lifespan=lifespan,
    docs_url="/docs" if _ENABLE_DOCS else None,
    redoc_url="/redoc" if _ENABLE_DOCS else None,
    openapi_url="/openapi.json" if _ENABLE_DOCS else None,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {
            "model": ErrorResponse,
            "description": "Service overloaded (all inference workers busy)",
        },
        504: {"model": ErrorResponse, "description": "Inference timed out"},
    },
)


@app.exception_handler(ServiceOverloadedError)
async def overloaded_handler(request: Request, exc: ServiceOverloadedError):
    """Return 503 when all inference workers are busy."""
    return JSONResponse(
        status_code=503, content={"detail": "Service temporarily overloaded, try again shortly"}
    )


@app.exception_handler(InferenceTimeoutError)
async def inference_timeout_handler(request: Request, exc: InferenceTimeoutError):
    """Return 504 when an inference runs past INFERENCE_TIMEOUT (something is wrong)."""
    logger.error("Inference timed out: %s", exc)
    return JSONResponse(
        status_code=504, content={"detail": "Inference timed out, try again shortly"}
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Prevent unhandled exceptions from leaking internal details to clients."""
    logger.error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def _validate_dialect(dialect: str | None) -> None:
    """Raise 422 if dialect was provided and is invalid or doesn't match this instance."""
    if dialect is not None:
        if dialect not in VALID_DIALECTS:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid dialect '{dialect}'. Must be one of: {sorted(VALID_DIALECTS)}",
            )
        if dialect != DIALECT:
            raise HTTPException(
                status_code=422,
                detail=f"This instance serves dialect '{DIALECT}', got '{dialect}'",
            )


def _resolve_lang(lang: str) -> str:
    """Normalize a language code to canonical and ensure this dialect supports it.

    Accepts either a canonical ISO 639-3 code or a known alias (e.g. 'ar' for
    'ara'), returning the canonical code used for label lookup. Raises 422 if the
    resolved language is not supported by this dialect.
    """
    canonical = normalize_lang(lang)
    if canonical not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=422,
            detail=f"lang='{lang}' is not supported for dialect='{DIALECT}'. "
            f"Supported: {_ACCEPTED_LANGS}",
        )
    return canonical


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
    response_model_exclude_none=True,
    tags=["Health"],
)
async def health_check() -> HealthResponse:
    """Check if the service is healthy."""
    # cache is None unless EXPOSE_CACHE_STATS is set; response_model_exclude_none
    # then drops the key entirely so /health omits it rather than returning null.
    cache = CacheStats(**get_cache_stats()) if _EXPOSE_CACHE_STATS else None
    return HealthResponse(status="healthy", cache=cache)


@app.post("/classify", response_model=ClassifyResponse, tags=["Classification"])
async def classify_single(
    request: Annotated[ClassifyRequest, Body()],
    lang: Annotated[str, Query(description=_LANG_PARAM_DESC)] = "ara",
    dialect: Annotated[str | None, Query(
        description=_DIALECT_PARAM_DESC)] = None,
) -> ClassifyResponse:
    """
    Classify a single text.

    Returns the predicted sub_class, main_class, and confidence score.

    The `lang` query parameter controls the language of the returned labels;
    the languages this dialect supports are listed in the `lang` parameter
    description. `lang` does not change which model runs.
    """
    _validate_dialect(dialect)
    lang = _resolve_lang(lang)
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
    lang: Annotated[str, Query(description=_LANG_PARAM_DESC)] = "ara",
    dialect: Annotated[str | None, Query(
        description=_DIALECT_PARAM_DESC)] = None,
) -> BatchClassifyResponse:
    """
    Classify multiple texts in a single request.

    Returns classification results in the same order as the input texts.
    More efficient than multiple single requests for large volumes.

    The `lang` query parameter controls the language of the returned labels;
    the languages this dialect supports are listed in the `lang` parameter
    description. `lang` does not change which model runs.
    """
    _validate_dialect(dialect)
    lang = _resolve_lang(lang)
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

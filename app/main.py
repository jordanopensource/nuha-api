"""
Nuha API - Text Classification Service.

Provides endpoints for single and batch text classification.
Each instance serves a single dialect, configured via the DIALECT env var.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Body, FastAPI, HTTPException, Path, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.classifier import (
    DEFAULT_LANGUAGE,
    DIALECT,
    DIALECT_NAMES,
    LANGUAGE_ALIASES,
    LANGUAGE_NAMES,
    SUPPORTED_LANGUAGES,
    VALID_DIALECTS,
    InferenceTimeoutError,
    ServiceOverloadedError,
    _parse_bounded_int,
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

# _parse_bounded_int is defined once, in app.classifier, and reused here.

MAX_BATCH_SIZE = _parse_bounded_int("MAX_BATCH_SIZE", 1000, 1, 10000)

# App-layer request body cap in bytes. nginx client_max_body_size (10 MiB) is the
# primary bound; this is the backstop for a backend exposed without the proxy
# (e.g. a bare `docker run -p 8000:8000`), where uvicorn would otherwise buffer a
# body of any size and ride the container to an OOM kill. Keep it in sync with
# the nginx limit; raise both together to serve larger legitimate batches.
MAX_BODY_SIZE = _parse_bounded_int("MAX_BODY_SIZE", 10 * 1024 * 1024, 1024, 1024**3)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Request body size cap
# -----------------------------------------------------------------------------


class BodyTooLargeError(Exception):
    """Raised mid-read when a request body exceeds MAX_BODY_SIZE."""


_TOO_LARGE_RESPONSE = {
    "type": "http.response.start",
    "status": 413,
    "headers": [(b"content-type", b"application/json")],
}
_TOO_LARGE_BODY = {
    "type": "http.response.body",
    "body": b'{"detail":"Request body too large"}',
}


class BodySizeLimitMiddleware:
    """Reject request bodies over ``max_bytes`` with a 413, without buffering.

    Pure ASGI (no BaseHTTPMiddleware), so the overhead is one header scan per
    request plus an integer add per body chunk; nothing is copied or buffered.
    Two layers:

    * A declared ``Content-Length`` over the cap is refused up front, before the
      app sees the request or any body is read.
    * Chunked bodies (or a lying Content-Length) are counted as they stream; the
      moment the running total passes the cap, the read raises
      ``BodyTooLargeError``, aborting the read so nothing is buffered past the
      cap -- this is the memory-safety guarantee, on every path. Under FastAPI
      that abort surfaces to the client as a generic 400 body-parse error
      (FastAPI wraps body reads); the app-level ``BodyTooLargeError`` handler and
      this middleware's own 413 send are backstops for the case the error
      propagates instead (e.g. a non-FastAPI mount). Either way the oversized
      body is never fully read.
    """

    def __init__(self, app, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        for name, value in scope["headers"]:
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    break  # malformed; the server/app will reject it
                if declared > self.max_bytes:
                    await send(_TOO_LARGE_RESPONSE)
                    await send(_TOO_LARGE_BODY)
                    return
                break

        received = 0
        response_started = False

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise BodyTooLargeError(
                        f"Request body exceeded {self.max_bytes} bytes mid-read"
                    )
            return message

        async def tracking_send(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except BodyTooLargeError:
            if response_started:
                raise
            await tracking_send(_TOO_LARGE_RESPONSE)
            await tracking_send(_TOO_LARGE_BODY)


# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------

# Language descriptions are built from the dialect config (defined here, before
# the request models, because the `lang` body field below uses them) so the docs
# never drift from the configured languages and carry no hardcoded knowledge.


def _lang_forms(code: str) -> str:
    """Render a language as its canonical code plus aliases, e.g. "'ara'/'ar' (Arabic)"."""
    codes = [code, *sorted(a for a, c in LANGUAGE_ALIASES.items() if c == code)]
    return "/".join(f"'{c}'" for c in codes) + f" ({LANGUAGE_NAMES[code]})"


_LANG_PARAM_DESC = (
    "Response language (controls label language, not which model runs). "
    "Accepts the canonical ISO 639-3 code or a two-letter alias. "
    "Supported by this dialect: "
    + ", ".join(_lang_forms(code) for code in LANGUAGE_NAMES)
    + f". Defaults to '{DEFAULT_LANGUAGE}' if omitted."
)
# Every code accepted for this dialect (canonical plus aliases), for error messages.
_ACCEPTED_LANGS = sorted(
    set(SUPPORTED_LANGUAGES) | {a for a, c in LANGUAGE_ALIASES.items() if c in SUPPORTED_LANGUAGES}
)
# `lang` is a short ISO 639 code (2-3 letters); cap the field at the longest code
# this dialect actually accepts. Coarse bound: it rejects an over-long value as a
# stripped 422 before _resolve_lang runs, so abuse text in `lang` is never echoed
# back. Derived from the accepted set, so it can never reject a valid code.
_MAX_LANG_LEN = max(len(code) for code in _ACCEPTED_LANGS)


class ClassifyRequest(BaseModel):
    """Request body for single text classification."""

    text: Annotated[str, Field(min_length=1, max_length=50000, description="Text to classify")]
    lang: Annotated[str, Field(max_length=_MAX_LANG_LEN, description=_LANG_PARAM_DESC)] = (
        DEFAULT_LANGUAGE
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"text": "نص للتصنيف", "lang": "ar"},
            ]
        }
    }


class ClassifyResponse(BaseModel):
    """Response for single text classification."""

    is_valid: bool = Field(description="Whether the input text was valid for classification")
    sub_class: str | None = Field(description="Classification sub_class (null if invalid)")
    main_class: str | None = Field(description="Classification main_class (null if invalid)")
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
    lang: Annotated[str, Field(max_length=_MAX_LANG_LEN, description=_LANG_PARAM_DESC)] = (
        DEFAULT_LANGUAGE
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"texts": ["نص للتصنيف 1", "نص للتصنيف 2"], "lang": "ar"},
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
    cache: CacheStats | None = Field(default=None, description="Inference cache statistics")


class ErrorResponse(BaseModel):
    """Error response schema."""

    detail: str


class ValidationErrorItem(BaseModel):
    """One field error. The 422 handler strips FastAPI's ``input``/``ctx`` echo,
    so the documented shape carries only the machine-readable location and reason."""

    loc: list[str | int]
    msg: str
    type: str


class ValidationErrorResponse(BaseModel):
    """422 body: validation errors only, never the rejected input."""

    detail: list[ValidationErrorItem]


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
# (The lang description lives with the request models above, which reference it.)

app = FastAPI(
    title="Nuha API",
    description="Text Classification API",
    version="0.3.0",
    lifespan=lifespan,
    docs_url="/docs" if _ENABLE_DOCS else None,
    redoc_url="/redoc" if _ENABLE_DOCS else None,
    openapi_url="/openapi.json" if _ENABLE_DOCS else None,
    responses={
        413: {"model": ErrorResponse, "description": "Request body too large"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)

app.add_middleware(BodySizeLimitMiddleware, max_bytes=MAX_BODY_SIZE)


@app.exception_handler(BodyTooLargeError)
async def body_too_large_handler(request: Request, exc: BodyTooLargeError):
    """Return 413 when a streamed body passes MAX_BODY_SIZE mid-read."""
    return JSONResponse(status_code=413, content={"detail": "Request body too large"})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return 422 without echoing the rejected input.

    FastAPI's default validation response includes the offending value (its
    ``input`` and ``ctx`` fields). Inputs here are abuse text and can be ~10 MB
    batches, so reflect only the machine-readable location and reason.
    """
    errors = [
        {"loc": err.get("loc", ()), "msg": err.get("msg", ""), "type": err.get("type", "")}
        for err in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errors})


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


def _validate_dialect(dialect: str) -> None:
    """Raise 422 if the path dialect is unknown or isn't the one this instance serves."""
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


# The dialect is a path segment (/<dialect>/classify, /<dialect>/classify/batch),
# which the nginx proxy routes to the matching backend and the backend validates
# against its own DIALECT. The shared _do_* helpers hold the classification logic
# so the single and batch endpoints cannot drift.

_DIALECT_PATH_DESC = (
    "Dialect to classify with; must match this instance's configured dialect. One of: "
    + ", ".join(f"'{code}' ({name})" for code, name in DIALECT_NAMES.items())
    + "."
)


async def _do_classify(text: str, lang: str) -> ClassifyResponse:
    """Single-text classification. Dialect is validated by the caller; `lang`
    controls the label language (not which model runs) and is resolved here."""
    result = await get_classification(text, lang=_resolve_lang(lang))
    return ClassifyResponse(
        is_valid=result.is_valid,
        sub_class=result.sub_class,
        main_class=result.main_class,
        confidence=result.confidence,
    )


async def _do_classify_batch(texts: list[str], lang: str) -> BatchClassifyResponse:
    """Batch classification. Dialect is validated by the caller; results come
    back in input order."""
    results = await get_classifications_batch(texts, lang=_resolve_lang(lang))
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


# Errors only the classification endpoints can return: 400 (an over-cap streamed
# body surfaces as a FastAPI body-parse error, which happens only on routes that
# read a body), 422 (body/path validation), 503 (the inference gate sheds), 504
# (inference timed out). Only 413 (Content-Length precheck, before routing) and
# 500 (the catch-all handler) are truly app-wide, so they stay on the app.
_CLASSIFY_ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Malformed request body"},
    422: {
        "model": ValidationErrorResponse,
        "description": "Validation error (rejected input is not echoed)",
    },
    503: {
        "model": ErrorResponse,
        "description": "Service overloaded (all inference workers busy)",
    },
    504: {"model": ErrorResponse, "description": "Inference timed out"},
}


@app.post(
    "/{dialect}/classify/batch",
    response_model=BatchClassifyResponse,
    responses=_CLASSIFY_ERROR_RESPONSES,
    tags=["Classification"],
)
async def classify_batch(
    dialect: Annotated[str, Path(description=_DIALECT_PATH_DESC)],
    request: Annotated[BatchClassifyRequest, Body()],
) -> BatchClassifyResponse:
    """
    Classify multiple texts in a single request.

    Returns classification results in the same order as the input texts. More
    efficient than multiple single requests for large volumes. The dialect is the
    path segment and must match this instance; `lang` goes in the body and
    controls the label language, not which model runs.
    """
    _validate_dialect(dialect)
    return await _do_classify_batch(request.texts, request.lang)


@app.post(
    "/{dialect}/classify",
    response_model=ClassifyResponse,
    responses=_CLASSIFY_ERROR_RESPONSES,
    tags=["Classification"],
)
async def classify_single(
    dialect: Annotated[str, Path(description=_DIALECT_PATH_DESC)],
    request: Annotated[ClassifyRequest, Body()],
) -> ClassifyResponse:
    """
    Classify a single text.

    Returns the predicted sub_class, main_class, and confidence score. The dialect
    is the path segment and must match this instance; `lang` goes in the body and
    controls the label language, not which model runs.
    """
    _validate_dialect(dialect)
    return await _do_classify(request.text, request.lang)

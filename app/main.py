"""
Nuha API - dialect-dynamic text classification service (2.0).

ONE stateless image. The ``{dialect}`` path segment selects a model loaded from
the models volume (see app.registry); inference, preprocessing, caching, and
labels run in-process (see app.classifier). Which dialects exist is decided at
runtime by what the volume holds when the process boots, so operators add or
remove a dialect with the fetch command plus a restart, never an image rebuild.
"""

import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Body, FastAPI, HTTPException, Path, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from app import registry
from app.classifier import (
    InferenceTimeoutError,
    LoadedDialect,
    ServiceOverloadedError,
    get_classification,
    get_classifications_batch,
    shutdown_executor,
)
from app.common.config import parse_bounded_int
from app.common.http import (
    DETAIL_OVERLOADED,
    DETAIL_TIMEOUT,
    HEADER_REQUEST_ID,
    BodySizeLimitMiddleware,
    SecurityHeadersMiddleware,
    install_common_handlers,
)
from app.common.logging import setup_logging
from app.common.schemas import (
    BatchClassifyResponse,
    BatchTextField,
    CacheStats,
    ClassifyRequest,
    ClassifyResponse,
    ErrorResponse,
    HealthResponse,
    LangField,
    ValidationErrorResponse,
)


setup_logging()
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

MAX_BATCH_SIZE = parse_bounded_int("MAX_BATCH_SIZE", 1000, 1, 10000)
MAX_BODY_SIZE = parse_bounded_int("MAX_BODY_SIZE", 10 * 1024 * 1024, 1024, 1024**3)

# Readiness is separate from liveness: the process is live (answering /health) as
# soon as uvicorn is up, but not READY until the startup scan has loaded the
# models. Compose/k8s gate traffic on /ready so requests never hit a
# still-loading process.
_ready = False


# -----------------------------------------------------------------------------
# Schemas (the batch request lives here because its cap is env-tunable)
# -----------------------------------------------------------------------------


class BatchClassifyRequest(BaseModel):
    """Batch request, capped at MAX_BATCH_SIZE (an over-cap list is a 422)."""

    texts: Annotated[
        list[BatchTextField],
        Field(
            min_length=1,
            max_length=MAX_BATCH_SIZE,
            description=f"List of texts to classify (max {MAX_BATCH_SIZE})",
        ),
    ]
    lang: LangField = None

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"texts": ["نص للتصنيف 1", "نص للتصنيف 2"], "lang": "ar"}]}
    )


# -----------------------------------------------------------------------------
# Application
# -----------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Scan the models volume once, flip readiness on, and drain on shutdown.

    Loads run before uvicorn accepts a request, so the first request never hits
    a cold model. A dialect that fails to load stays out (its requests get 400)
    while its siblings serve; an empty or missing volume starts a zero-dialect
    process (bootstrap flow: bring the stack up, run fetch, restart).
    """
    global _ready
    logger.info("Starting Nuha API (models dir: %s)...", registry.MODELS_DIR)
    result = registry.scan_once()
    _ready = True
    logger.info(
        "Nuha API ready (loaded=%s failed=%s)", sorted(result.loaded), sorted(result.failed)
    )
    yield
    _ready = False
    logger.info("Shutting down Nuha API...")
    shutdown_executor()


_ENABLE_DOCS = os.getenv("DISABLE_DOCS") is None
_EXPOSE_CACHE_STATS = os.getenv("EXPOSE_CACHE_STATS") is not None

app = FastAPI(
    title="Nuha API",
    description="Text Classification API",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs" if _ENABLE_DOCS else None,
    redoc_url="/redoc" if _ENABLE_DOCS else None,
    openapi_url="/openapi.json" if _ENABLE_DOCS else None,
    responses={
        413: {"model": ErrorResponse, "description": "Request body too large"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)

# Middleware order (add order is inner-to-outer; the last added is outermost):
# security headers outermost so every response, including a 413, carries them.
# Both are pure-ASGI and allocation-free. In-flight work is bounded by the
# inference gate (CLASSIFIER_WORKERS + INFERENCE_QUEUE_SIZE), and request rate
# is the platform edge's job, so no extra in-flight cap is added here.
app.add_middleware(BodySizeLimitMiddleware, max_bytes=MAX_BODY_SIZE)
app.add_middleware(SecurityHeadersMiddleware)

install_common_handlers(app)


def _rid_headers(request: Request) -> dict[str, str] | None:
    """The correlation-id echo for error responses, when the route stashed one."""
    rid = getattr(request.state, "request_id", None)
    return {HEADER_REQUEST_ID: rid} if rid else None


@app.exception_handler(ServiceOverloadedError)
async def overloaded_handler(request: Request, exc: ServiceOverloadedError):
    """The admission gate shed this request -> 503."""
    return JSONResponse(
        status_code=503, content={"detail": DETAIL_OVERLOADED}, headers=_rid_headers(request)
    )


@app.exception_handler(InferenceTimeoutError)
async def inference_timeout_handler(request: Request, exc: InferenceTimeoutError):
    """An inference ran past INFERENCE_TIMEOUT -> 504."""
    rid = getattr(request.state, "request_id", None)
    logger.error("Inference timed out (rid=%s): %s", rid, exc)
    return JSONResponse(
        status_code=504, content={"detail": DETAIL_TIMEOUT}, headers=_rid_headers(request)
    )


def _get_entry_or_400(dialect: str) -> LoadedDialect:
    """The loaded dialect for a path segment, or the contract 400.

    The valid set is dynamic: exactly the dialects the startup scan loaded from
    the models volume. The detail format is frozen; only the list inside it
    tracks the volume.
    """
    entry = registry.get(dialect)
    if entry is None:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid dialect. Must be one of: {', '.join(sorted(registry.codes()))}.",
        )
    return entry


def _resolve_lang(entry: LoadedDialect, lang: str | None) -> str:
    """Resolve an optional request language to a canonical, serviceable code.

    Omitted -> the dialect's default (the alphabetically first of its declared
    codes, always serviceable). Otherwise normalize an alias to canonical and
    reject (422) a code this dialect does not declare, the same lang contract
    as 1.x.
    """
    if lang is None:
        return entry.default_language
    canonical = entry.aliases.get(lang, lang)
    if canonical not in entry.supported_languages:
        # Do not echo the client-supplied lang value back (keeps the "422 never
        # reflects input" invariant); just name what this dialect supports.
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported lang for dialect '{entry.code}'. "
                f"Supported: {sorted(entry.supported_languages)}"
            ),
        )
    return canonical


# A correlation id is echoed on the response, so only accept a conservative
# token (URL/log/header-safe, bounded length) from the client; anything else is
# replaced with a fresh id rather than reflected verbatim.
_REQUEST_ID_RE = re.compile(r"\A[A-Za-z0-9._-]{1,128}\Z")


def _request_id(request: Request) -> str:
    """Reuse a well-formed inbound correlation id, else mint one.

    Also stashed on ``request.state`` so the 503/504 handlers can echo the same
    id on failure responses.
    """
    inbound = request.headers.get(HEADER_REQUEST_ID)
    rid = inbound if inbound and _REQUEST_ID_RE.match(inbound) else uuid.uuid4().hex
    request.state.request_id = rid
    return rid


# The loaded set is runtime state, so the docs describe the mechanism, not a list.
_DIALECT_PATH_DESC = (
    "Dialect to classify with: the code of a model installed on the models "
    "volume. An unknown code gets a 400 naming the loaded codes."
)

_CLASSIFY_ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Unknown dialect"},
    422: {
        "model": ValidationErrorResponse,
        "description": "Invalid request body or lang (rejected input is not echoed)",
    },
    503: {"model": ErrorResponse, "description": "Service overloaded"},
    504: {"model": ErrorResponse, "description": "Inference timed out"},
}


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------


@app.get(
    "/health", response_model=HealthResponse, response_model_exclude_none=True, tags=["Health"]
)
async def health_check() -> HealthResponse:
    """Liveness plus the loaded dialect codes (does NOT imply readiness).

    Per-dialect cache statistics are included only when EXPOSE_CACHE_STATS is
    set (they reveal traffic patterns, so they are opt-in).
    """
    loaded = sorted(registry.codes())
    cache = None
    if _EXPOSE_CACHE_STATS:
        cache = {code: CacheStats(**registry.get(code).cache.stats) for code in loaded}
    return HealthResponse(status="healthy", dialects=loaded, cache=cache)


@app.get("/ready", tags=["Health"], responses={503: {"description": "Not ready (draining)"}})
async def readiness_check():
    """Readiness for compose/k8s to gate traffic: 200 once the startup scan ran.

    The lifespan scans and loads before uvicorn accepts any request, so during
    boot the probe is refused (which also reads as not-ready), not answered 503;
    the 503 path here covers the shutdown drain. Both live responses carry a
    ``{"status": ...}`` body. Ready with zero dialects is deliberate: the
    process is serving its contract (400s) while the volume is being populated.
    """
    if not _ready:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ready"}


@app.post(
    "/{dialect}/classify/batch",
    response_model=BatchClassifyResponse,
    responses=_CLASSIFY_ERROR_RESPONSES,
    tags=["Classification"],
)
async def classify_batch(
    dialect: Annotated[str, Path(description=_DIALECT_PATH_DESC)],
    request: Annotated[BatchClassifyRequest, Body()],
    http_request: Request,
):
    """Classify multiple texts with one dialect's model; results in input order."""
    entry = _get_entry_or_400(dialect)
    rid = _request_id(http_request)
    lang = _resolve_lang(entry, request.lang)
    results = await get_classifications_batch(entry, request.texts, lang)
    body = BatchClassifyResponse(
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
    return JSONResponse(
        status_code=200, content=body.model_dump(), headers={HEADER_REQUEST_ID: rid}
    )


@app.post(
    "/{dialect}/classify",
    response_model=ClassifyResponse,
    responses=_CLASSIFY_ERROR_RESPONSES,
    tags=["Classification"],
)
async def classify_single(
    dialect: Annotated[str, Path(description=_DIALECT_PATH_DESC)],
    request: Annotated[ClassifyRequest, Body()],
    http_request: Request,
):
    """Classify a single text with one dialect's model; `lang` goes in the body."""
    entry = _get_entry_or_400(dialect)
    rid = _request_id(http_request)
    lang = _resolve_lang(entry, request.lang)
    result = await get_classification(entry, request.text, lang)
    body = ClassifyResponse(
        is_valid=result.is_valid,
        sub_class=result.sub_class,
        main_class=result.main_class,
        confidence=result.confidence,
    )
    return JSONResponse(
        status_code=200, content=body.model_dump(), headers={HEADER_REQUEST_ID: rid}
    )

"""Request/response Pydantic models for the public contract.

The public response contract is frozen from 1.x: ``{is_valid, sub_class,
main_class, confidence}`` and ``{results: [...]}``. ``lang`` is optional; when
omitted the app applies the requested dialect's default language (the
alphabetically first of its declared codes). The batch REQUEST lives in
app/main.py, built from the shared field aliases here, because its list cap is
the env-tunable MAX_BATCH_SIZE.
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from app.common.config import MAX_LANG_LEN


TEXT_MAX_LEN = 50000

# Shared field aliases so the single and batch requests stay identical field-wise.
TextField = Annotated[
    str, Field(min_length=1, max_length=TEXT_MAX_LEN, description="Text to classify")
]
BatchTextField = Annotated[str, Field(max_length=TEXT_MAX_LEN)]
LangField = Annotated[
    str | None,
    Field(
        default=None,
        max_length=MAX_LANG_LEN,
        description=(
            "Response language (ISO 639-3 code or a two-letter alias). Controls the "
            "label language, not which model runs. Omit to use the dialect's default."
        ),
    ),
]


class ClassifyRequest(BaseModel):
    """Single-text request."""

    text: TextField
    lang: LangField = None

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"text": "نص للتصنيف", "lang": "ar"}]}
    )


class ClassifyResponse(BaseModel):
    """Single classification result (the frozen public shape)."""

    is_valid: bool = Field(description="Whether the input text was valid for classification")
    sub_class: str | None = Field(description="Classification sub_class (null if invalid)")
    main_class: str | None = Field(description="Classification main_class (null if invalid)")
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Confidence score 0-1 (null if invalid)"
    )


class BatchClassifyResponse(BaseModel):
    """Batch result, in input order."""

    results: list[ClassifyResponse] = Field(
        description="Classification results in same order as input"
    )


class ErrorResponse(BaseModel):
    """Generic single-message error body ({"detail": "..."})."""

    detail: str


class ValidationErrorItem(BaseModel):
    """One field error; the 422 handler strips FastAPI's input/ctx echo."""

    loc: list[str | int]
    msg: str
    type: str


class ValidationErrorResponse(BaseModel):
    """422 body: validation errors only, never the rejected input."""

    detail: list[ValidationErrorItem]


class CacheStats(BaseModel):
    """Inference cache statistics (/health, opt-in via EXPOSE_CACHE_STATS)."""

    size: int = Field(description="Current number of cached entries")
    maxsize: int = Field(description="Maximum cache capacity")
    hits: int = Field(description="Total cache hits")
    misses: int = Field(description="Total cache misses")
    hit_rate: float = Field(description="Cache hit rate (0.0 to 1.0)")


class HealthResponse(BaseModel):
    """Health/liveness response, listing the loaded dialects."""

    status: str
    dialects: list[str] | None = Field(
        default=None, description="Sorted codes of the currently loaded dialects"
    )
    cache: dict[str, CacheStats] | None = Field(
        default=None, description="Per-dialect inference cache statistics, keyed by code"
    )

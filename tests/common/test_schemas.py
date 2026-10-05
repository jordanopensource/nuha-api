"""Tests for app.common.schemas: the shared request/response Pydantic models.

The public response contract is frozen from 1.x. These tests pin the field
constraints the endpoints build from (text 1..50000, optional lang capped at
MAX_LANG_LEN, confidence in [0,1]); schemas are pure pydantic, no ML anywhere.
"""

import pytest
from pydantic import ValidationError

from app.common.config import MAX_LANG_LEN
from app.common.schemas import (
    TEXT_MAX_LEN,
    BatchClassifyResponse,
    ClassifyRequest,
    ClassifyResponse,
    ErrorResponse,
    HealthResponse,
    ValidationErrorResponse,
)


class TestClassifyRequest:
    def test_minimal_valid(self):
        req = ClassifyRequest(text="hello")
        assert req.text == "hello"
        assert req.lang is None  # lang is optional, default None

    def test_empty_text_rejected(self):
        with pytest.raises(ValidationError):
            ClassifyRequest(text="")

    def test_text_at_max_length_ok(self):
        ClassifyRequest(text="a" * TEXT_MAX_LEN)

    def test_text_over_max_length_rejected(self):
        with pytest.raises(ValidationError):
            ClassifyRequest(text="a" * (TEXT_MAX_LEN + 1))

    def test_lang_at_cap_ok(self):
        ClassifyRequest(text="hello", lang="a" * MAX_LANG_LEN)

    def test_lang_over_cap_rejected(self):
        with pytest.raises(ValidationError):
            ClassifyRequest(text="hello", lang="a" * (MAX_LANG_LEN + 1))


class TestClassifyResponse:
    def test_valid_result(self):
        r = ClassifyResponse(is_valid=True, sub_class="s", main_class="m", confidence=0.5)
        assert r.confidence == 0.5

    def test_invalid_result_nulls(self):
        r = ClassifyResponse(is_valid=False, sub_class=None, main_class=None, confidence=None)
        assert r.sub_class is None and r.main_class is None and r.confidence is None

    def test_confidence_bounds_enforced(self):
        for bad in (-0.01, 1.01):
            with pytest.raises(ValidationError):
                ClassifyResponse(is_valid=True, sub_class="s", main_class="m", confidence=bad)


class TestOtherSchemas:
    def test_batch_response_wraps_results(self):
        item = ClassifyResponse(is_valid=False, sub_class=None, main_class=None, confidence=None)
        batch = BatchClassifyResponse(results=[item, item])
        assert len(batch.results) == 2

    def test_error_response_detail_is_string(self):
        assert ErrorResponse(detail="boom").detail == "boom"

    def test_validation_error_response_detail_is_list(self):
        body = ValidationErrorResponse(
            detail=[{"loc": ["body", "text"], "msg": "field required", "type": "missing"}]
        )
        assert body.detail[0].loc == ["body", "text"]

    def test_health_response_omits_cache_by_default(self):
        h = HealthResponse(status="healthy")
        assert h.cache is None
        # response_model_exclude_none on the endpoint drops the None fields.
        assert h.model_dump(exclude_none=True) == {"status": "healthy"}

    def test_health_response_carries_dialects_and_keyed_cache(self):
        """The 2.0 health body lists the loaded dialect codes and, when cache
        stats are exposed, keys them by dialect code."""
        from app.common.schemas import CacheStats

        stats = CacheStats(size=1, maxsize=8, hits=2, misses=3, hit_rate=0.4)
        h = HealthResponse(status="healthy", dialects=["aaa", "bbb"], cache={"aaa": stats})
        dumped = h.model_dump(exclude_none=True)
        assert dumped["dialects"] == ["aaa", "bbb"]
        assert set(dumped["cache"]) == {"aaa"}
        assert dumped["cache"]["aaa"]["hits"] == 2

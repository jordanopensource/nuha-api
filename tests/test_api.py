"""API contract tests for FastAPI endpoints.

These tests verify request/response contracts, validation, and error
handling -- NOT classification accuracy. The model is fully mocked.
"""

import os
from unittest.mock import patch

import pytest


# =============================================================================
# /health endpoint
# =============================================================================


class TestHealthEndpoint:
    """Tests for the GET /health endpoint."""

    def test_returns_200(self, test_client):
        """Health check returns 200."""
        resp = test_client.get("/health")
        assert resp.status_code == 200

    def test_returns_healthy_status(self, test_client):
        """Health response includes status=healthy."""
        resp = test_client.get("/health")
        data = resp.json()
        assert data["status"] == "healthy"

    def test_content_type_json(self, test_client):
        """Response content-type is application/json."""
        resp = test_client.get("/health")
        assert "application/json" in resp.headers["content-type"]

    def test_cache_stats_present_when_enabled(self, test_client):
        """Health response includes cache statistics when EXPOSE_CACHE_STATS is on."""
        with patch("app.main._EXPOSE_CACHE_STATS", True):
            resp = test_client.get("/health")
        data = resp.json()
        assert "cache" in data
        cache = data["cache"]
        assert "size" in cache
        assert "maxsize" in cache
        assert "hits" in cache
        assert "misses" in cache
        assert "hit_rate" in cache

    def test_cache_stats_hidden_by_default(self, test_client):
        """Health response omits cache statistics unless explicitly enabled."""
        with patch("app.main._EXPOSE_CACHE_STATS", False):
            resp = test_client.get("/health")
        data = resp.json()
        assert data["status"] == "healthy"
        assert "cache" not in data


# =============================================================================
# /classify endpoint
# =============================================================================


class TestClassifySingle:
    """Tests for the POST /classify endpoint."""

    def test_valid_request_returns_200(self, test_client):
        """Valid classification request returns 200."""
        resp = test_client.post("/classify", json={"text": "مرحبا بالعالم"})
        assert resp.status_code == 200

    def test_response_has_required_fields(self, test_client):
        """Response contains is_valid, sub_class, main_class, confidence."""
        resp = test_client.post("/classify", json={"text": "مرحبا بالعالم"})
        data = resp.json()
        assert "is_valid" in data
        assert "sub_class" in data
        assert "main_class" in data
        assert "confidence" in data

    def test_valid_text_is_valid_true(self, test_client):
        """Valid Arabic text returns is_valid=true."""
        resp = test_client.post("/classify", json={"text": "مرحبا بالعالم"})
        data = resp.json()
        assert data["is_valid"] is True
        assert data["sub_class"] is not None
        assert data["main_class"] is not None
        assert data["confidence"] is not None

    def test_empty_text_returns_422(self, test_client):
        """Empty text triggers Pydantic validation error (422)."""
        resp = test_client.post("/classify", json={"text": ""})
        assert resp.status_code == 422

    def test_text_exceeding_max_length_returns_422(self, test_client):
        """Text exceeding 10000 chars returns 422."""
        long_text = "ا" * 10001
        resp = test_client.post("/classify", json={"text": long_text})
        assert resp.status_code == 422

    def test_text_at_max_length_accepted(self, test_client):
        """Text exactly at 10000 chars is accepted."""
        text = "ا" * 10000
        resp = test_client.post("/classify", json={"text": text})
        # Should not be a validation error; may return is_valid=false
        # due to preprocessing, but the HTTP request is accepted
        assert resp.status_code == 200

    def test_missing_text_field_returns_422(self, test_client):
        """Missing 'text' field returns 422."""
        resp = test_client.post("/classify", json={})
        assert resp.status_code == 422

    def test_dialect_param_matching_succeeds(self, test_client):
        """dialect param matching the container's dialect succeeds."""
        dialect = os.environ.get("DIALECT", "arz")
        resp = test_client.post(f"/classify?dialect={dialect}", json={"text": "مرحبا بالعالم"})
        assert resp.status_code == 200

    def test_dialect_param_mismatching_returns_422(self, test_client):
        """dialect param mismatching returns 422."""
        dialect = os.environ.get("DIALECT", "arz")
        other = "acm" if dialect != "acm" else "ckb"
        resp = test_client.post(f"/classify?dialect={other}", json={"text": "مرحبا بالعالم"})
        assert resp.status_code == 422

    def test_dialect_param_omitted_succeeds(self, test_client):
        """Omitting dialect param succeeds (uses container's dialect)."""
        resp = test_client.post("/classify", json={"text": "مرحبا بالعالم"})
        assert resp.status_code == 200

    def test_lang_ar_succeeds(self, test_client):
        """lang=ar is accepted."""
        resp = test_client.post("/classify?lang=ar", json={"text": "مرحبا بالعالم"})
        assert resp.status_code == 200

    def test_lang_en_succeeds(self, test_client):
        """lang=en is accepted."""
        resp = test_client.post("/classify?lang=en", json={"text": "مرحبا بالعالم"})
        assert resp.status_code == 200

    def test_lang_ckb_on_non_ckb_returns_422(self, test_client):
        """lang=ckb on non-Kurdish dialect returns 422."""
        dialect = os.environ.get("DIALECT", "arz")
        if dialect == "ckb":
            pytest.skip("This test only applies to non-ckb dialects")
        resp = test_client.post("/classify?lang=ckb", json={"text": "مرحبا بالعالم"})
        assert resp.status_code == 422

    def test_lang_invalid_returns_422(self, test_client):
        """Invalid lang value returns 422."""
        resp = test_client.post("/classify?lang=xxx", json={"text": "مرحبا بالعالم"})
        assert resp.status_code == 422

    def test_dialect_invalid_returns_422(self, test_client):
        """Invalid dialect value returns 422."""
        resp = test_client.post("/classify?dialect=xxx", json={"text": "مرحبا بالعالم"})
        assert resp.status_code == 422

    def test_response_schema_types(self, test_client):
        """Response field types match ClassifyResponse schema."""
        resp = test_client.post("/classify", json={"text": "مرحبا بالعالم"})
        data = resp.json()
        assert isinstance(data["is_valid"], bool)
        if data["is_valid"]:
            assert isinstance(data["sub_class"], str)
            assert isinstance(data["main_class"], str)
            assert isinstance(data["confidence"], (int, float))
            assert 0.0 <= data["confidence"] <= 1.0
        else:
            assert data["sub_class"] is None
            assert data["main_class"] is None
            assert data["confidence"] is None

    def test_confidence_between_0_and_1(self, test_client):
        """Confidence score is between 0 and 1 when valid."""
        resp = test_client.post("/classify", json={"text": "مرحبا بالعالم"})
        data = resp.json()
        if data["is_valid"]:
            assert 0.0 <= data["confidence"] <= 1.0


# =============================================================================
# /classify/batch endpoint
# =============================================================================


class TestClassifyBatch:
    """Tests for the POST /classify/batch endpoint."""

    def test_valid_batch_returns_200(self, test_client):
        """Valid batch request returns 200."""
        resp = test_client.post(
            "/classify/batch",
            json={"texts": ["مرحبا بالعالم", "نص آخر للتصنيف"]},
        )
        assert resp.status_code == 200

    def test_results_list_matches_input_length(self, test_client):
        """Results list length matches number of input texts."""
        texts = ["مرحبا بالعالم", "نص آخر", "ثالث"]
        resp = test_client.post("/classify/batch", json={"texts": texts})
        data = resp.json()
        assert len(data["results"]) == len(texts)

    def test_empty_texts_list_returns_422(self, test_client):
        """Empty texts list returns 422."""
        resp = test_client.post("/classify/batch", json={"texts": []})
        assert resp.status_code == 422

    def test_batch_exceeding_max_size_returns_422(self, test_client):
        """Batch exceeding MAX_BATCH_SIZE returns 422."""
        from app.main import MAX_BATCH_SIZE

        texts = ["مرحبا"] * (MAX_BATCH_SIZE + 1)
        resp = test_client.post("/classify/batch", json={"texts": texts})
        assert resp.status_code == 422

    def test_individual_text_exceeding_max_length_returns_422(self, test_client):
        """Individual text exceeding max_length in batch returns 422."""
        texts = ["مرحبا", "ا" * 10001]
        resp = test_client.post("/classify/batch", json={"texts": texts})
        assert resp.status_code == 422

    def test_empty_string_in_batch_returns_is_valid_false(self, test_client):
        """Empty string in batch returns 200 with is_valid=false for that item."""
        resp = test_client.post(
            "/classify/batch",
            json={"texts": ["مرحبا بالعالم", "", "نص آخر للتصنيف"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["results"][1]["is_valid"] is False
        assert data["results"][1]["sub_class"] is None

    def test_mixed_valid_invalid_texts(self, test_client):
        """Mixed valid/invalid texts: valid ones classified, invalid get is_valid=false."""
        resp = test_client.post(
            "/classify/batch",
            json={"texts": ["مرحبا بالعالم", "", "   ", "نص عربي صحيح"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        results = data["results"]
        assert len(results) == 4
        assert results[0]["is_valid"] is True
        assert results[3]["is_valid"] is True
        assert results[1]["is_valid"] is False
        assert results[2]["is_valid"] is False

    def test_batch_dialect_matching_succeeds(self, test_client):
        """dialect param matching container's dialect succeeds for batch."""
        dialect = os.environ.get("DIALECT", "arz")
        resp = test_client.post(
            f"/classify/batch?dialect={dialect}",
            json={"texts": ["مرحبا بالعالم"]},
        )
        assert resp.status_code == 200

    def test_batch_dialect_mismatching_returns_422(self, test_client):
        """dialect param mismatching returns 422 for batch."""
        dialect = os.environ.get("DIALECT", "arz")
        other = "acm" if dialect != "acm" else "ckb"
        resp = test_client.post(
            f"/classify/batch?dialect={other}",
            json={"texts": ["مرحبا بالعالم"]},
        )
        assert resp.status_code == 422

    def test_batch_lang_en_succeeds(self, test_client):
        """lang=en succeeds for batch."""
        resp = test_client.post(
            "/classify/batch?lang=en",
            json={"texts": ["مرحبا بالعالم"]},
        )
        assert resp.status_code == 200

    def test_batch_lang_ckb_on_non_ckb_returns_422(self, test_client):
        """lang=ckb on non-Kurdish dialect returns 422 for batch."""
        dialect = os.environ.get("DIALECT", "arz")
        if dialect == "ckb":
            pytest.skip("This test only applies to non-ckb dialects")
        resp = test_client.post(
            "/classify/batch?lang=ckb",
            json={"texts": ["مرحبا بالعالم"]},
        )
        assert resp.status_code == 422

    def test_batch_each_result_has_required_fields(self, test_client):
        """Each result in batch response has the required fields."""
        resp = test_client.post(
            "/classify/batch",
            json={"texts": ["مرحبا بالعالم"]},
        )
        data = resp.json()
        for result in data["results"]:
            assert "is_valid" in result
            assert "sub_class" in result
            assert "main_class" in result
            assert "confidence" in result

    def test_missing_texts_field_returns_422(self, test_client):
        """Missing 'texts' field returns 422."""
        resp = test_client.post("/classify/batch", json={})
        assert resp.status_code == 422

    def test_single_text_batch_works(self, test_client):
        """Batch with a single text works correctly."""
        resp = test_client.post(
            "/classify/batch",
            json={"texts": ["مرحبا بالعالم"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1


# =============================================================================
# Error handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling behavior."""

    def test_unhandled_exception_returns_500(self, test_client):
        """Unhandled exception returns 500 with generic error detail."""
        with patch(
            "app.main.get_classification",
            side_effect=RuntimeError("unexpected boom"),
        ):
            resp = test_client.post("/classify", json={"text": "مرحبا بالعالم"})
        assert resp.status_code == 500
        data = resp.json()
        assert data["detail"] == "Internal server error"
        assert "boom" not in str(data)

    def test_overload_returns_503(self, test_client):
        """ServiceOverloadedError returns 503."""
        from app.classifier import ServiceOverloadedError

        with patch(
            "app.main.get_classification",
            side_effect=ServiceOverloadedError("All workers busy"),
        ):
            resp = test_client.post("/classify", json={"text": "مرحبا بالعالم"})
        assert resp.status_code == 503
        data = resp.json()
        assert "detail" in data

    def test_inference_timeout_returns_504(self, test_client):
        """InferenceTimeoutError returns 504 without leaking internals."""
        from app.classifier import InferenceTimeoutError

        with patch(
            "app.main.get_classification",
            side_effect=InferenceTimeoutError("Inference did not complete within 120s"),
        ):
            resp = test_client.post("/classify", json={"text": "مرحبا بالعالم"})
        assert resp.status_code == 504
        data = resp.json()
        assert "detail" in data
        assert "120s" not in str(data)

    def test_batch_inference_timeout_returns_504(self, test_client):
        """Batch endpoint also returns 504 on inference timeout."""
        from app.classifier import InferenceTimeoutError

        with patch(
            "app.main.get_classifications_batch",
            side_effect=InferenceTimeoutError("timed out"),
        ):
            resp = test_client.post("/classify/batch", json={"texts": ["مرحبا"]})
        assert resp.status_code == 504

    def test_500_does_not_leak_stack_trace(self, test_client):
        """500 response does not contain stack trace information."""
        with patch(
            "app.main.get_classification",
            side_effect=ValueError("secret internal error"),
        ):
            resp = test_client.post("/classify", json={"text": "مرحبا بالعالم"})
        assert resp.status_code == 500
        body = resp.text
        assert "Traceback" not in body
        assert "ValueError" not in body
        assert "secret internal error" not in body

    def test_batch_unhandled_exception_returns_500(self, test_client):
        """Batch endpoint also returns 500 on unhandled exception."""
        with patch(
            "app.main.get_classifications_batch",
            side_effect=RuntimeError("batch boom"),
        ):
            resp = test_client.post("/classify/batch", json={"texts": ["مرحبا بالعالم"]})
        assert resp.status_code == 500
        data = resp.json()
        assert data["detail"] == "Internal server error"


# =============================================================================
# Content type and general HTTP behavior
# =============================================================================


class TestHttpBehavior:
    """Tests for general HTTP behavior."""

    def test_classify_returns_json_content_type(self, test_client):
        """Classify endpoint returns application/json."""
        resp = test_client.post("/classify", json={"text": "مرحبا بالعالم"})
        assert "application/json" in resp.headers["content-type"]

    def test_batch_returns_json_content_type(self, test_client):
        """Batch endpoint returns application/json."""
        resp = test_client.post("/classify/batch", json={"texts": ["مرحبا بالعالم"]})
        assert "application/json" in resp.headers["content-type"]

    def test_nonexistent_endpoint_returns_404(self, test_client):
        """Request to nonexistent endpoint returns 404."""
        resp = test_client.get("/nonexistent")
        assert resp.status_code == 404

    def test_get_on_classify_returns_405(self, test_client):
        """GET on /classify (which expects POST) returns 405."""
        resp = test_client.get("/classify")
        assert resp.status_code == 405

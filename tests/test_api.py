"""API contract tests for FastAPI endpoints.

These tests verify request/response contracts, validation, and error
handling -- NOT classification accuracy. The model is fully mocked.
"""

import os
from unittest.mock import patch

import pytest

from tests.conftest import _DIALECT_FILES, ALL_DIALECTS


# =============================================================================
# /health endpoint
# =============================================================================


class TestHealthEndpoint:
    """Tests for the GET /health endpoint."""

    def test_returns_200_healthy(self, test_client):
        """Health check returns 200 with status=healthy."""
        resp = test_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

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

    def test_valid_text_classified(self, test_client):
        """Valid Arabic text returns 200 with is_valid=true and all fields set."""
        resp = test_client.post("/classify", json={"text": "مرحبا بالعالم"})
        assert resp.status_code == 200
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
        """Text exceeding 50000 chars returns 422."""
        long_text = "ا" * 50001
        resp = test_client.post("/classify", json={"text": long_text})
        assert resp.status_code == 422

    def test_text_at_max_length_accepted(self, test_client):
        """Text exactly at 50000 chars is accepted."""
        text = "ا" * 50000
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
        dialect = os.environ["DIALECT"]
        resp = test_client.post(f"/classify?dialect={dialect}", json={"text": "مرحبا بالعالم"})
        assert resp.status_code == 200

    def test_dialect_param_mismatching_returns_422(self, test_client):
        """A VALID dialect that isn't this instance's returns 422."""
        dialect = os.environ["DIALECT"]
        other = next((d for d in ALL_DIALECTS if d != dialect), None)
        if other is None:
            pytest.skip("Only one dialect file exists; no valid mismatch possible")
        resp = test_client.post(f"/classify?dialect={other}", json={"text": "مرحبا بالعالم"})
        assert resp.status_code == 422

    def test_dialect_param_omitted_succeeds(self, test_client):
        """Omitting dialect param succeeds (uses container's dialect)."""
        resp = test_client.post("/classify", json={"text": "مرحبا بالعالم"})
        assert resp.status_code == 200

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
        texts = ["مرحبا", "ا" * 50001]
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
        dialect = os.environ["DIALECT"]
        resp = test_client.post(
            f"/classify/batch?dialect={dialect}",
            json={"texts": ["مرحبا بالعالم"]},
        )
        assert resp.status_code == 200

    def test_batch_dialect_mismatching_returns_422(self, test_client):
        """A VALID dialect that isn't this instance's returns 422 for batch."""
        dialect = os.environ["DIALECT"]
        other = next((d for d in ALL_DIALECTS if d != dialect), None)
        if other is None:
            pytest.skip("Only one dialect file exists; no valid mismatch possible")
        resp = test_client.post(
            f"/classify/batch?dialect={other}",
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

    def test_nonexistent_endpoint_returns_404(self, test_client):
        """Request to nonexistent endpoint returns 404."""
        resp = test_client.get("/nonexistent")
        assert resp.status_code == 404

    def test_get_on_classify_returns_405(self, test_client):
        """GET on /classify (which expects POST) returns 405."""
        resp = test_client.get("/classify")
        assert resp.status_code == 405


# =============================================================================
# ?lang= acceptance matrix (regression lock)
# =============================================================================
#
# Locks the behaviour of the `lang` query parameter without hardcoding any
# dialect's values: every canonical code and alias the active dialect's file
# declares must be accepted on both endpoints; any code it does NOT declare
# (a real language served by another dialect, or a nonsense code) must be
# rejected with 422. Everything derives from the dialect files, so the tests
# hold for any number of dialects.

_BODY_SINGLE = {"text": "مرحبا بالعالم"}
_BODY_BATCH = {"texts": ["مرحبا بالعالم"]}


def _current_dialect() -> str:
    return os.environ["DIALECT"]


def _declared_languages(dialect: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(canonical codes, aliases) the dialect's file declares."""
    languages = _DIALECT_FILES[dialect]["languages"]
    canonical = tuple(languages)
    aliases = tuple(a for meta in languages.values() for a in meta.get("aliases", []))
    return canonical, aliases


class TestLangAcceptanceMatrix:
    """Exhaustive ?lang= matrix for the active dialect, on both endpoints."""

    def test_every_declared_code_accepted(self, test_client):
        """Every canonical code and alias the dialect file declares returns 200
        on /classify and /classify/batch."""
        canonical, aliases = _declared_languages(_current_dialect())
        for code in canonical + aliases:
            single = test_client.post(f"/classify?lang={code}", json=_BODY_SINGLE)
            batch = test_client.post(f"/classify/batch?lang={code}", json=_BODY_BATCH)
            assert single.status_code == 200, f"lang={code} should be 200 on /classify"
            assert batch.status_code == 200, f"lang={code} should be 200 on /classify/batch"

    def test_undeclared_language_rejected(self, test_client):
        """A real language code that other dialects serve but this dialect's
        file does not declare returns 422 on both endpoints."""
        mine = set().union(*_declared_languages(_current_dialect()))
        others = {
            code
            for dialect in ALL_DIALECTS
            for group in _declared_languages(dialect)
            for code in group
        }
        undeclared = sorted(others - mine)
        if not undeclared:
            pytest.skip("Every language any dialect serves is declared by this dialect")
        for code in undeclared:
            single = test_client.post(f"/classify?lang={code}", json=_BODY_SINGLE)
            batch = test_client.post(f"/classify/batch?lang={code}", json=_BODY_BATCH)
            assert single.status_code == 422, f"undeclared lang={code} should be 422"
            assert batch.status_code == 422, f"undeclared lang={code} should be 422 (batch)"

    def test_unknown_lang_rejected(self, test_client):
        """A nonsense language code returns 422 on both endpoints."""
        assert test_client.post("/classify?lang=xx", json=_BODY_SINGLE).status_code == 422
        assert test_client.post("/classify/batch?lang=xx", json=_BODY_BATCH).status_code == 422

    def test_omitted_lang_uses_first_declared_alphabetically(self, test_client):
        """Omitting ?lang= is equivalent to passing the first declared language
        alphabetically (app.classifier.DEFAULT_LANGUAGE) -- the default is
        derived from the dialect file, not hardcoded to any specific language."""
        default_lang = sorted(_DIALECT_FILES[_current_dialect()]["languages"])[0]
        omitted = test_client.post("/classify", json=_BODY_SINGLE)
        explicit = test_client.post(f"/classify?lang={default_lang}", json=_BODY_SINGLE)
        assert omitted.status_code == explicit.status_code == 200
        assert omitted.json() == explicit.json()

    def test_canonical_and_alias_agree(self, test_client):
        """Each declared alias returns the identical response to its canonical
        code (same label set, same everything)."""
        languages = _DIALECT_FILES[_current_dialect()]["languages"]
        pairs = [
            (canonical, alias)
            for canonical, meta in languages.items()
            for alias in meta.get("aliases", [])
        ]
        if not pairs:
            pytest.skip("The active dialect declares no aliases")
        for canonical, alias in pairs:
            canonical_resp = test_client.post(f"/classify?lang={canonical}", json=_BODY_SINGLE)
            alias_resp = test_client.post(f"/classify?lang={alias}", json=_BODY_SINGLE)
            assert canonical_resp.status_code == alias_resp.status_code == 200
            assert canonical_resp.json() == alias_resp.json()


# =============================================================================
# 422 validation responses must not echo the rejected input
# =============================================================================


class TestValidationErrorResponse:
    """422 bodies keep loc/msg/type but never reflect the submitted payload
    (inputs are abuse text; a rejected 10 MB batch must not be mirrored back)."""

    def test_too_long_text_not_echoed(self, test_client):
        marker = "SENTINEL-DO-NOT-ECHO"
        resp = test_client.post("/classify", json={"text": marker + "ا" * 50001})
        assert resp.status_code == 422
        assert marker not in resp.text

    def test_wrong_type_value_not_echoed(self, test_client):
        marker = "SENTINEL-DO-NOT-ECHO"
        resp = test_client.post("/classify/batch", json={"texts": {"oops": marker}})
        assert resp.status_code == 422
        assert marker not in resp.text

    def test_error_structure_preserved(self, test_client):
        """Clients still get machine-readable errors: loc, msg, and type."""
        resp = test_client.post("/classify", json={})
        assert resp.status_code == 422
        errors = resp.json()["detail"]
        assert isinstance(errors, list) and errors
        for err in errors:
            assert set(err) == {"loc", "msg", "type"}


# =============================================================================
# Request body size cap (app-layer backstop for the nginx 10 MiB limit)
# =============================================================================


class TestBodySizeLimit:
    """The ASGI body-size middleware rejects oversized bodies with 413 without
    buffering them, so a directly-exposed backend can't be OOM'd by a huge POST."""

    def _mini_app(self, max_bytes):
        """Wrap a trivial ASGI app that would happily read any body."""
        from app.main import BodySizeLimitMiddleware

        async def echo_app(scope, receive, send):
            body = b""
            while True:
                message = await receive()
                body += message.get("body", b"")
                if not message.get("more_body"):
                    break
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"text/plain")],
                }
            )
            await send({"type": "http.response.body", "body": b"ok:%d" % len(body)})

        return BodySizeLimitMiddleware(echo_app, max_bytes=max_bytes)

    def _request(self, app, body: bytes, content_length: int | None):
        """Drive the ASGI app with one request; return (status, received chunks)."""
        import asyncio

        headers = []
        if content_length is not None:
            headers.append((b"content-length", str(content_length).encode()))
        scope = {"type": "http", "method": "POST", "path": "/", "headers": headers}
        messages = [
            {"type": "http.request", "body": body[i : i + 10], "more_body": i + 10 < len(body)}
            for i in range(0, max(len(body), 1), 10)
        ]
        sent = []

        async def receive():
            return messages.pop(0)

        async def send(message):
            sent.append(message)

        asyncio.run(app(scope, receive, send))
        status = next(m["status"] for m in sent if m["type"] == "http.response.start")
        return status, sent

    def test_declared_content_length_over_cap_rejected(self):
        """An oversized Content-Length is rejected up front, body unread."""
        app = self._mini_app(max_bytes=50)
        status, _ = self._request(app, b"x" * 100, content_length=100)
        assert status == 413

    def test_streamed_body_over_cap_rejected(self):
        """A chunked body (no Content-Length) is cut off once it passes the cap.
        This drives the middleware directly with an app that lets the error
        propagate, so it exercises the middleware's own 413 backstop. Under the
        real FastAPI app the same abort surfaces to the client as a 400 body-parse
        error (FastAPI wraps body reads); either way the body is never fully read
        -- the memory-safety guarantee, which is what this asserts."""
        app = self._mini_app(max_bytes=50)
        status, _ = self._request(app, b"x" * 100, content_length=None)
        assert status == 413

    def test_body_under_cap_passes(self):
        """A body under the cap reaches the app untouched."""
        app = self._mini_app(max_bytes=50)
        status, sent = self._request(app, b"x" * 30, content_length=30)
        assert status == 200
        assert any(b"ok:30" in m.get("body", b"") for m in sent)

    def test_endpoint_rejects_oversized_body(self, test_client):
        """End to end: a request body over MAX_BODY_SIZE returns 413."""
        import app.main as main_mod

        oversized = b'{"text": "' + b"a" * (main_mod.MAX_BODY_SIZE + 16) + b'"}'
        resp = test_client.post(
            "/classify", content=oversized, headers={"content-type": "application/json"}
        )
        assert resp.status_code == 413
        assert resp.json() == {"detail": "Request body too large"}

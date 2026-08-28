"""The public HTTP contract of the single Nuha service.

Merges the old gateway routing suite and the model-server contract suite: one
app now owns dialect routing, request validation, lang resolution, the
engine-failure surface, and the correlation id. The engine entry points are
mocked (label-faithful fakes over the real per-dialect tables; see
tests/conftest.py), while routing, schema validation, the startup scan, and
lang resolution run for real. Everything parametrizes over the discovered
dialects, so it holds for one dialect file or many.
"""

from unittest.mock import patch

import pytest

from app.classifier import InferenceTimeoutError, ServiceOverloadedError
from app.common.config import MAX_LANG_LEN
from app.common.http import HEADER_REQUEST_ID
from app.common.schemas import TEXT_MAX_LEN
from tests.conftest import _DIALECT_FILES, ALL_DIALECTS, make_model_volume


ALL = sorted(ALL_DIALECTS)

# The exact frozen client-visible status set. No 502, ever.
ALLOWED_STATUSES = {200, 400, 404, 405, 413, 422, 429, 500, 503, 504}

# The frozen public error strings, pinned as literals on purpose: a drift in
# app.common.http's constants must fail here, not silently update the pins.
_OVERLOADED_DETAIL = "Service temporarily overloaded, try again shortly"
_TIMEOUT_DETAIL = "Inference timed out, try again shortly"
_INTERNAL_DETAIL = "Internal server error"

_SINGLE = {"text": "مرحبا بالعالم"}


def _declared_languages(dialect: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    languages = _DIALECT_FILES[dialect]["languages"]
    canonical = tuple(languages)
    aliases = tuple(a for meta in languages.values() for a in meta.get("aliases", []))
    return canonical, aliases


# =============================================================================
# /health: liveness + the loaded dialect codes
# =============================================================================


class TestHealth:
    def test_returns_200_healthy_with_loaded_dialects(self, api):
        resp = api.client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["dialects"] == ALL

    def test_cache_stats_present_when_enabled(self, api):
        with patch("app.main._EXPOSE_CACHE_STATS", True):
            resp = api.client.get("/health")
        cache = resp.json()["cache"]
        assert set(cache) == set(ALL)
        for stats in cache.values():
            assert {"size", "maxsize", "hits", "misses", "hit_rate"} <= set(stats)

    def test_cache_stats_hidden_by_default(self, api):
        with patch("app.main._EXPOSE_CACHE_STATS", False):
            resp = api.client.get("/health")
        assert "cache" not in resp.json()


# =============================================================================
# /ready: readiness split
# =============================================================================


class TestReadiness:
    def test_ready_returns_200_after_startup_scan(self, api):
        resp = api.client.get("/ready")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ready"}

    def test_ready_returns_503_when_not_ready(self, api):
        """Readiness is 503 outside the ready window (the shutdown drain)."""
        with patch("app.main._ready", False):
            resp = api.client.get("/ready")
        assert resp.status_code == 503
        assert resp.json() == {"status": "loading"}

    def test_ready_with_zero_dialects(self, api_factory):
        """An empty volume boots a ready, zero-dialect process on purpose: it
        serves its contract (400s) while the volume is being populated."""
        with api_factory(codes=()) as api:
            assert api.client.get("/ready").status_code == 200
            assert api.client.get("/health").json()["dialects"] == []
            assert api.client.post(f"/{ALL[0]}/classify", json=_SINGLE).status_code == 400


# =============================================================================
# Dialect routing
# =============================================================================


class TestRouting:
    @pytest.mark.parametrize("dialect", ALL)
    def test_single_routes_to_that_dialect(self, api, dialect):
        resp = api.client.post(f"/{dialect}/classify", json=_SINGLE)
        assert resp.status_code == 200
        # Only that dialect's model was called, with the request text.
        assert api.calls, f"{dialect} engine not called"
        code, payload, _lang = api.calls[-1]
        assert code == dialect
        assert payload == _SINGLE["text"]
        assert {c for c, _p, _l in api.calls} == {dialect}

    @pytest.mark.parametrize("dialect", ALL)
    def test_batch_routes_and_preserves_order(self, api, dialect):
        texts = ["مرحبا", "نص آخر", "ثالث"]
        resp = api.client.post(f"/{dialect}/classify/batch", json={"texts": texts})
        assert resp.status_code == 200
        assert len(resp.json()["results"]) == len(texts)
        code, payload, _lang = api.calls[-1]
        assert code == dialect
        assert payload == texts

    def test_alias_lang_resolved_before_the_engine(self, api, any_dialect):
        """The engine only ever sees canonical language codes: a declared alias
        is resolved at the route (under the superseded split this resolution
        lived on the model server; the public behavior is identical)."""
        languages = _DIALECT_FILES[any_dialect]["languages"]
        pairs = [
            (canonical, alias)
            for canonical, meta in languages.items()
            for alias in meta.get("aliases", [])
        ]
        if not pairs:
            pytest.skip("The dialect declares no aliases")
        canonical, alias = pairs[0]
        api.client.post(f"/{any_dialect}/classify", json={"text": "مرحبا", "lang": alias})
        assert api.calls[-1][2] == canonical

    def test_omitted_lang_uses_first_declared_alphabetically(self, api, any_dialect):
        default_lang = sorted(_DIALECT_FILES[any_dialect]["languages"])[0]
        api.client.post(f"/{any_dialect}/classify", json={"text": "مرحبا"})
        assert api.calls[-1][2] == default_lang

    def test_single_and_batch_paths_distinct(self, api, any_dialect):
        single = api.client.post(f"/{any_dialect}/classify", json=_SINGLE)
        batch = api.client.post(f"/{any_dialect}/classify/batch", json={"texts": ["مرحبا"]})
        assert "results" not in single.json()
        assert "results" in batch.json()


# =============================================================================
# X-Request-Id: mint, reuse, echo (success and failure)
# =============================================================================


class TestRequestId:
    def test_request_id_echoed_in_response(self, api, any_dialect):
        resp = api.client.post(f"/{any_dialect}/classify", json=_SINGLE)
        assert resp.headers.get(HEADER_REQUEST_ID)

    def test_inbound_request_id_reused(self, api, any_dialect):
        rid = "corr-1234"
        resp = api.client.post(
            f"/{any_dialect}/classify", json=_SINGLE, headers={HEADER_REQUEST_ID: rid}
        )
        assert resp.headers.get(HEADER_REQUEST_ID) == rid

    @pytest.mark.parametrize(
        ("exc", "expected_status"),
        [
            (ServiceOverloadedError("All workers busy"), 503),
            (InferenceTimeoutError("stuck"), 504),
        ],
    )
    def test_request_id_echoed_on_engine_failure(self, api, any_dialect, exc, expected_status):
        """A failed request is traceable like a successful one: the 503 and the
        504 both carry the same correlation id the client sent."""
        rid = "corr-fail-1234"
        api.fail(exc)
        resp = api.client.post(
            f"/{any_dialect}/classify", json=_SINGLE, headers={HEADER_REQUEST_ID: rid}
        )
        assert resp.status_code == expected_status
        assert resp.headers.get(HEADER_REQUEST_ID) == rid

    def test_malformed_inbound_request_id_replaced(self, api, any_dialect):
        """A client-supplied id that isn't a safe token (too long, or control /
        injection chars) is replaced with a fresh one, never reflected verbatim."""
        for bad in ["a" * 200, "bad id with spaces", "x\r\nInjected: 1", "hi;drop"]:
            resp = api.client.post(
                f"/{any_dialect}/classify",
                json=_SINGLE,
                headers={HEADER_REQUEST_ID: bad},
            )
            echoed = resp.headers.get(HEADER_REQUEST_ID)
            assert echoed != bad
            assert echoed and len(echoed) <= 128


# =============================================================================
# Unknown / missing dialect
# =============================================================================


class TestDialectValidation:
    def test_loaded_set_is_exactly_the_installed_dialects(self, api):
        """The routable set is exactly what the startup scan loaded from the
        volume, the load-bearing "install a model dir = a new dialect" invariant."""
        assert api.client.get("/health").json()["dialects"] == ALL

    def test_unknown_dialect_is_400(self, api):
        """An unknown dialect is a 400 at the edge, and the engine is not called."""
        assert api.client.post("/zzz/classify", json=_SINGLE).status_code == 400
        assert api.client.post("/zzz/classify/batch", json={"texts": ["مرحبا"]}).status_code == 400
        assert not api.calls

    def test_unknown_dialect_detail_names_the_loaded_codes(self, api):
        resp = api.client.post("/zzz/classify", json=_SINGLE)
        assert resp.json()["detail"] == f"Invalid dialect. Must be one of: {', '.join(ALL)}."

    def test_400_detail_tracks_the_volume(self, api_factory):
        """The 400 detail's list is the LOADED set, not a baked-in one: boot
        against a subset volume and the detail names exactly that subset."""
        subset = ALL[:-1] or ALL
        with api_factory(codes=subset) as api:
            resp = api.client.post("/zzz/classify", json=_SINGLE)
            assert (
                resp.json()["detail"]
                == f"Invalid dialect. Must be one of: {', '.join(sorted(subset))}."
            )

    def test_malformed_json_body_is_422_not_400(self, api, any_dialect):
        """A malformed request body is a 422 (validation), never a 400; 400 is
        reserved for an unknown dialect. Latches the documented contract."""
        resp = api.client.post(
            f"/{any_dialect}/classify",
            content=b"{not json",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 422

    def test_bare_classify_is_404(self, api):
        """A classify request must name a dialect; /classify on its own is not a
        route (404), so a no-dialect call never misroutes."""
        assert api.client.post("/classify", json=_SINGLE).status_code == 404
        assert api.client.post("/classify/batch", json={"texts": ["مرحبا"]}).status_code == 404

    def test_get_on_classify_405(self, api, any_dialect):
        assert api.client.get(f"/{any_dialect}/classify").status_code == 405


# =============================================================================
# Restart pickup: fetch/remove + restart is the add/remove story
# =============================================================================


class TestRestartPickup:
    def test_added_dialect_serves_after_restart(self, api_factory, tmp_path):
        """Install a model dir, restart (a fresh boot), and the dialect serves;
        before the restart it is an unknown-dialect 400."""
        codes = ALL
        volume = make_model_volume(tmp_path / "restart-models", codes=codes[:-1])
        added = codes[-1]
        with api_factory(volume=volume) as api:
            assert api.client.post(f"/{added}/classify", json=_SINGLE).status_code == 400
        make_model_volume(volume, codes=(added,))
        with api_factory(volume=volume) as api:
            assert api.client.post(f"/{added}/classify", json=_SINGLE).status_code == 200
            assert api.client.get("/health").json()["dialects"] == sorted(codes)

    def test_removed_dialect_400_after_restart(self, api_factory, tmp_path):
        import shutil

        volume = make_model_volume(tmp_path / "remove-models")
        removed = ALL[0]
        with api_factory(volume=volume) as api:
            assert api.client.post(f"/{removed}/classify", json=_SINGLE).status_code == 200
        shutil.rmtree(volume / removed)
        with api_factory(volume=volume) as api:
            assert api.client.post(f"/{removed}/classify", json=_SINGLE).status_code == 400
            assert removed not in api.client.get("/health").json()["dialects"]


# =============================================================================
# Body validation (MAX_BATCH_SIZE, text bounds)
# =============================================================================


class TestBodyValidation:
    def test_empty_text_422(self, api, any_dialect):
        assert api.client.post(f"/{any_dialect}/classify", json={"text": ""}).status_code == 422

    def test_missing_text_422(self, api, any_dialect):
        assert api.client.post(f"/{any_dialect}/classify", json={}).status_code == 422

    def test_text_over_max_length_422(self, api, any_dialect):
        resp = api.client.post(f"/{any_dialect}/classify", json={"text": "ا" * (TEXT_MAX_LEN + 1)})
        assert resp.status_code == 422

    def test_text_at_max_length_accepted(self, api, any_dialect):
        resp = api.client.post(f"/{any_dialect}/classify", json={"text": "ا" * TEXT_MAX_LEN})
        assert resp.status_code == 200

    def test_empty_texts_list_422(self, api, any_dialect):
        assert (
            api.client.post(f"/{any_dialect}/classify/batch", json={"texts": []}).status_code == 422
        )

    def test_missing_texts_field_422(self, api, any_dialect):
        assert api.client.post(f"/{any_dialect}/classify/batch", json={}).status_code == 422

    def test_individual_text_over_max_length_422(self, api, any_dialect):
        resp = api.client.post(
            f"/{any_dialect}/classify/batch", json={"texts": ["مرحبا", "ا" * (TEXT_MAX_LEN + 1)]}
        )
        assert resp.status_code == 422

    def test_batch_over_max_size_422(self, api, any_dialect):
        """MAX_BATCH_SIZE is the one batch cap now (the split's model-side
        backstop collapsed into it); an over-cap batch is a 422 and never
        reaches the engine."""
        from app.main import MAX_BATCH_SIZE

        resp = api.client.post(
            f"/{any_dialect}/classify/batch", json={"texts": ["م"] * (MAX_BATCH_SIZE + 1)}
        )
        assert resp.status_code == 422
        assert not api.calls

    def test_batch_at_max_size_ok(self, api, any_dialect):
        from app.main import MAX_BATCH_SIZE

        resp = api.client.post(
            f"/{any_dialect}/classify/batch", json={"texts": ["م"] * MAX_BATCH_SIZE}
        )
        assert resp.status_code == 200

    def test_empty_string_in_batch_is_valid_false(self, api, any_dialect):
        data = api.client.post(
            f"/{any_dialect}/classify/batch",
            json={"texts": ["مرحبا بالعالم", "", "نص آخر للتصنيف"]},
        ).json()
        assert data["results"][1]["is_valid"] is False
        assert data["results"][1]["sub_class"] is None

    def test_response_schema_types(self, api, any_dialect):
        data = api.client.post(f"/{any_dialect}/classify", json=_SINGLE).json()
        assert isinstance(data["is_valid"], bool)
        if data["is_valid"]:
            assert isinstance(data["sub_class"], str)
            assert isinstance(data["main_class"], str)
            assert 0.0 <= data["confidence"] <= 1.0
        else:
            assert data["sub_class"] is None and data["main_class"] is None


# =============================================================================
# lang handling: capped by the schema (stripped 422) and validated inline
# against the requested dialect's declared languages
# =============================================================================


class TestLangHandling:
    def test_over_long_lang_stripped_422_no_echo(self, api, any_dialect):
        """An over-long `lang` is a stripped 422 (the schema's MAX_LANG_LEN cap)
        before the value is ever seen, so it is never reflected back."""
        marker = "SENTINEL-DO-NOT-ECHO"
        over_long = marker + "a" * (MAX_LANG_LEN + 200)
        single = api.client.post(
            f"/{any_dialect}/classify", json={"text": "مرحبا", "lang": over_long}
        )
        batch = api.client.post(
            f"/{any_dialect}/classify/batch", json={"texts": ["مرحبا"], "lang": over_long}
        )
        assert single.status_code == batch.status_code == 422
        assert marker not in single.text and marker not in batch.text

    def test_in_cap_unknown_lang_is_422(self, api, any_dialect):
        """An in-cap but undeclared code is a 422 from the app's own lang
        resolution. (Under the split the gateway forwarded it and the model
        server answered the same 422; the public behavior is unchanged, the
        check just runs inline now.) The engine is never called."""
        single = api.client.post(f"/{any_dialect}/classify", json={"text": "مرحبا", "lang": "xx"})
        batch = api.client.post(
            f"/{any_dialect}/classify/batch", json={"texts": ["مرحبا"], "lang": "xx"}
        )
        assert single.status_code == batch.status_code == 422
        assert not api.calls

    @pytest.mark.parametrize("dialect", ALL)
    def test_every_declared_code_accepted(self, api, dialect):
        canonical, aliases = _declared_languages(dialect)
        for code in canonical + aliases:
            single = api.client.post(f"/{dialect}/classify", json={"text": "مرحبا", "lang": code})
            batch = api.client.post(
                f"/{dialect}/classify/batch", json={"texts": ["مرحبا"], "lang": code}
            )
            assert single.status_code == 200, f"lang={code} should be 200 on /classify"
            assert batch.status_code == 200, f"lang={code} should be 200 on /classify/batch"

    @pytest.mark.parametrize("dialect", ALL)
    def test_undeclared_language_rejected(self, api, dialect):
        """A real language other dialects serve but this dialect does not
        declare is 422."""
        mine = set().union(*_declared_languages(dialect))
        others = {
            code for other in ALL_DIALECTS for group in _declared_languages(other) for code in group
        }
        undeclared = sorted(others - mine)
        if not undeclared:
            pytest.skip("Every language any dialect serves is declared by this dialect")
        for code in undeclared:
            resp = api.client.post(f"/{dialect}/classify", json={"text": "مرحبا", "lang": code})
            assert resp.status_code == 422

    def test_omitted_lang_equals_explicit_default(self, api, any_dialect):
        default_lang = sorted(_DIALECT_FILES[any_dialect]["languages"])[0]
        omitted = api.client.post(f"/{any_dialect}/classify", json=_SINGLE)
        explicit = api.client.post(
            f"/{any_dialect}/classify", json={**_SINGLE, "lang": default_lang}
        )
        assert omitted.status_code == explicit.status_code == 200
        assert omitted.json() == explicit.json()

    def test_canonical_and_alias_agree(self, api, any_dialect):
        languages = _DIALECT_FILES[any_dialect]["languages"]
        pairs = [
            (canonical, alias)
            for canonical, meta in languages.items()
            for alias in meta.get("aliases", [])
        ]
        if not pairs:
            pytest.skip("The dialect declares no aliases")
        for canonical, alias in pairs:
            a = api.client.post(f"/{any_dialect}/classify", json={**_SINGLE, "lang": canonical})
            b = api.client.post(f"/{any_dialect}/classify", json={**_SINGLE, "lang": alias})
            assert a.status_code == b.status_code == 200
            assert a.json() == b.json()


# =============================================================================
# Engine-failure surface: 503 / 504 / 500
# =============================================================================


class TestErrorHandling:
    def test_overload_returns_503(self, api, any_dialect):
        api.fail(ServiceOverloadedError("All workers busy"))
        resp = api.client.post(f"/{any_dialect}/classify", json=_SINGLE)
        assert resp.status_code == 503
        assert resp.json() == {"detail": _OVERLOADED_DETAIL}

    def test_inference_timeout_returns_504(self, api, any_dialect):
        api.fail(InferenceTimeoutError("Inference did not complete within 120s"))
        resp = api.client.post(f"/{any_dialect}/classify", json=_SINGLE)
        assert resp.status_code == 504
        assert resp.json() == {"detail": _TIMEOUT_DETAIL}
        assert "120s" not in resp.text

    def test_batch_inference_timeout_returns_504(self, api, any_dialect):
        api.fail(InferenceTimeoutError("timed out"))
        resp = api.client.post(f"/{any_dialect}/classify/batch", json={"texts": ["مرحبا"]})
        assert resp.status_code == 504

    def test_unhandled_exception_returns_500(self, api, any_dialect):
        api.fail(RuntimeError("unexpected boom"))
        resp = api.client.post(f"/{any_dialect}/classify", json=_SINGLE)
        assert resp.status_code == 500
        assert resp.json()["detail"] == _INTERNAL_DETAIL
        assert "boom" not in resp.text

    def test_500_does_not_leak_stack_trace(self, api, any_dialect):
        api.fail(ValueError("secret internal error"))
        resp = api.client.post(f"/{any_dialect}/classify", json=_SINGLE)
        assert resp.status_code == 500
        assert "Traceback" not in resp.text
        assert "secret internal error" not in resp.text


# =============================================================================
# 422 responses must not echo the rejected input
# =============================================================================


class TestValidationErrorResponse:
    def test_too_long_text_not_echoed(self, api, any_dialect):
        marker = "SENTINEL-DO-NOT-ECHO"
        resp = api.client.post(
            f"/{any_dialect}/classify", json={"text": marker + "ا" * (TEXT_MAX_LEN + 1)}
        )
        assert resp.status_code == 422
        assert marker not in resp.text

    def test_error_structure_preserved(self, api, any_dialect):
        errors = api.client.post(f"/{any_dialect}/classify", json={}).json()["detail"]
        assert isinstance(errors, list) and errors
        for err in errors:
            assert set(err) == {"loc", "msg", "type"}


# =============================================================================
# The closed status set: never a 502, nothing outside the frozen set
# =============================================================================


class TestNo502AndAllowedSet:
    @pytest.mark.parametrize(
        "exc",
        [
            None,  # 200
            ServiceOverloadedError("shed"),
            InferenceTimeoutError("stuck"),
            RuntimeError("boom"),
            ValueError("boom"),
        ],
    )
    def test_status_in_allowed_set_never_502(self, api, any_dialect, exc):
        if exc is not None:
            api.fail(exc)
        resp = api.client.post(f"/{any_dialect}/classify", json=_SINGLE)
        assert resp.status_code in ALLOWED_STATUSES
        assert resp.status_code != 502

    def test_edge_statuses_in_allowed_set(self, api, any_dialect):
        for resp in (
            api.client.post("/zzz/classify", json=_SINGLE),
            api.client.post("/classify", json=_SINGLE),
            api.client.get(f"/{any_dialect}/classify"),
            api.client.post(f"/{any_dialect}/classify", json={"text": ""}),
        ):
            assert resp.status_code in ALLOWED_STATUSES
            assert resp.status_code != 502


class TestRealAppMiddlewareWiring:
    """Pins that the REAL app is wired with both edge middlewares (a wiring
    regression in app/main.py would otherwise pass the suite green)."""

    def test_app_mounts_edge_middlewares_in_order(self):
        import app.main as app_main
        from app.common.http import BodySizeLimitMiddleware, SecurityHeadersMiddleware

        classes = [m.cls for m in app_main.app.user_middleware]
        for cls in (SecurityHeadersMiddleware, BodySizeLimitMiddleware):
            assert cls in classes
        # Starlette applies user_middleware in list order, outermost first, and
        # add_middleware() prepends. Security headers were added LAST, so they are
        # outermost and every response (including a 413) carries them.
        assert classes.index(SecurityHeadersMiddleware) < classes.index(BodySizeLimitMiddleware)

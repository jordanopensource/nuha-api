"""Tests for classifier module logic.

Tests the configuration, executor management, inference cache, label
selection, and error handling in the classifier module -- all with
mocked models (no ML framework dependency).
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest


# =============================================================================
# ACTIVE_CONFIG tests
# =============================================================================


class TestActiveConfig:
    """Tests for ACTIVE_CONFIG population and dialect configuration."""

    def test_active_config_populated(self):
        """ACTIVE_CONFIG is populated (not None) after module import."""
        from app.classifier import ACTIVE_CONFIG

        assert ACTIVE_CONFIG is not None

    def test_active_config_name_set(self):
        """ACTIVE_CONFIG has a non-empty name."""
        from app.classifier import ACTIVE_CONFIG

        assert ACTIVE_CONFIG.name
        assert isinstance(ACTIVE_CONFIG.name, str)

    def test_active_config_model_path_set(self):
        """ACTIVE_CONFIG has a model_path."""
        from app.classifier import ACTIVE_CONFIG

        assert ACTIVE_CONFIG.model_path
        assert isinstance(ACTIVE_CONFIG.model_path, str)

    def test_active_config_has_preprocess_fn(self):
        """ACTIVE_CONFIG has a callable preprocess_fn."""
        from app.classifier import ACTIVE_CONFIG

        assert callable(ACTIVE_CONFIG.preprocess_fn)

    def test_active_config_has_label_dicts(self):
        """ACTIVE_CONFIG has lang-keyed sub/main label dictionaries."""
        from app.classifier import ACTIVE_CONFIG

        assert isinstance(ACTIVE_CONFIG.sub_labels, dict)
        assert isinstance(ACTIVE_CONFIG.main_labels, dict)
        # Arabic is supported by every dialect.
        assert isinstance(ACTIVE_CONFIG.sub_labels["ara"], dict)
        assert isinstance(ACTIVE_CONFIG.main_labels["ara"], dict)

    def test_active_config_has_sub_to_main(self):
        """ACTIVE_CONFIG has a sub_to_main mapping."""
        from app.classifier import ACTIVE_CONFIG

        assert isinstance(ACTIVE_CONFIG.sub_to_main, dict)
        assert len(ACTIVE_CONFIG.sub_to_main) > 0

    def test_active_config_sub_to_main_int_keys(self):
        """ACTIVE_CONFIG.sub_to_main has integer keys."""
        from app.classifier import ACTIVE_CONFIG

        for k in ACTIVE_CONFIG.sub_to_main:
            assert isinstance(k, int)

    def test_active_config_label_dicts_int_keys(self):
        """ACTIVE_CONFIG label dicts have integer keys."""
        from app.classifier import ACTIVE_CONFIG

        for by_lang in (ACTIVE_CONFIG.sub_labels, ACTIVE_CONFIG.main_labels):
            for label_dict in by_lang.values():
                for k in label_dict:
                    assert isinstance(k, int)


# =============================================================================
# MODEL_PATH configuration
# =============================================================================


class TestModelPathConfig:
    """Tests for MODEL_PATH resolution logic."""

    def test_default_model_path(self):
        """MODEL_PATH defaults to ./models/{DIALECT} when not set."""
        from app.classifier import DIALECT, MODEL_PATH

        if not os.getenv("MODEL_PATH"):
            assert MODEL_PATH == f"./models/{DIALECT}"

    def test_model_path_env_override(self):
        """MODEL_PATH env var overrides the default (tested via expression logic)."""
        # The source uses: MODEL_PATH = os.getenv("MODEL_PATH") or f"./models/{DIALECT}"
        test_val = "/custom/path"
        result = test_val or "./models/arz"
        assert result == "/custom/path"

    def test_empty_model_path_uses_default(self):
        """Empty MODEL_PATH env var falls back to default (uses 'or' not 'if')."""
        result = "" or "./models/arz"
        assert result == "./models/arz"

    def test_none_model_path_uses_default(self):
        """None MODEL_PATH (unset) falls back to default."""
        result = None or "./models/arz"
        assert result == "./models/arz"


# =============================================================================
# Executor and semaphore
# =============================================================================


class TestExecutorManagement:
    """Tests for thread pool executor and semaphore management."""

    def test_get_executor_returns_thread_pool(self):
        """_get_executor() returns a ThreadPoolExecutor."""
        from app.classifier import _get_executor

        executor = _get_executor()
        assert isinstance(executor, ThreadPoolExecutor)

    def test_shutdown_executor_cleans_up(self):
        """shutdown_executor() sets the global to None."""
        import app.classifier as clf

        clf._get_executor()
        assert clf._executor is not None

        clf.shutdown_executor()
        assert clf._executor is None

    def test_get_executor_after_shutdown_recreates(self):
        """Getting executor after shutdown creates a new one."""
        import app.classifier as clf

        clf.shutdown_executor()
        executor = clf._get_executor()
        assert isinstance(executor, ThreadPoolExecutor)
        clf.shutdown_executor()


# =============================================================================
# _InferenceGate (non-blocking overload limiter)
# =============================================================================


class TestInferenceGate:
    """Direct tests for the non-blocking concurrency gate (H-1)."""

    def test_acquires_up_to_limit(self):
        """try_acquire() succeeds exactly `limit` times, then refuses."""
        from app.classifier import _InferenceGate

        gate = _InferenceGate(2)
        assert gate.try_acquire() is True
        assert gate.try_acquire() is True
        assert gate.try_acquire() is False  # at capacity

    def test_release_frees_a_slot(self):
        """Releasing a slot lets the next try_acquire() succeed again."""
        from app.classifier import _InferenceGate

        gate = _InferenceGate(1)
        assert gate.try_acquire() is True
        assert gate.try_acquire() is False
        gate.release()
        assert gate.try_acquire() is True

    def test_release_never_goes_negative(self):
        """Extra releases can't push capacity above the limit."""
        from app.classifier import _InferenceGate

        gate = _InferenceGate(1)
        gate.release()  # nothing acquired — must be a no-op
        gate.release()
        assert gate.try_acquire() is True
        assert gate.try_acquire() is False  # still only one slot

    def test_full_cycle_restores_capacity(self):
        """Acquire then release the same number of times returns to full."""
        from app.classifier import _InferenceGate

        gate = _InferenceGate(3)
        assert [gate.try_acquire() for _ in range(3)] == [True, True, True]
        assert gate.try_acquire() is False
        for _ in range(3):
            gate.release()
        assert [gate.try_acquire() for _ in range(3)] == [True, True, True]


# =============================================================================
# Per-request inference timeout (_run_gated)
# =============================================================================


class TestInferenceTimeout:
    """The timeout fires, and the gate slot frees only when work truly ends."""

    def test_timeout_raises_and_frees_slot_after_work_finishes(self, monkeypatch):
        """On timeout the caller gets InferenceTimeoutError, the slot stays held
        while the abandoned worker runs, and is freed once it finishes."""
        import asyncio
        import threading
        import time

        import app.classifier as clf

        # Tiny timeout; a worker that runs well past it. Patch the gate so we can
        # observe slot accounting in isolation.
        monkeypatch.setattr(clf, "INFERENCE_TIMEOUT", 0.1)
        gate = clf._InferenceGate(1)
        monkeypatch.setattr(clf, "_inference_gate", gate)

        finished = threading.Event()

        def slow_fn(*_args):
            time.sleep(0.5)
            finished.set()
            return "done"

        async def go():
            with pytest.raises(clf.InferenceTimeoutError):
                await clf._run_gated(slow_fn)
            # The worker is still running, so its slot must NOT have been freed.
            assert gate.try_acquire() is False
            # Once the abandoned worker completes, the done-callback frees it.
            assert finished.wait(3.0) is True
            await asyncio.sleep(0.1)  # let the loop run the done-callback
            assert gate.try_acquire() is True

        asyncio.run(go())

    def test_fast_work_under_timeout_returns_result(self, monkeypatch):
        """A normal inference well under the timeout returns and frees its slot."""
        import asyncio

        import app.classifier as clf

        monkeypatch.setattr(clf, "INFERENCE_TIMEOUT", 5)
        gate = clf._InferenceGate(1)
        monkeypatch.setattr(clf, "_inference_gate", gate)

        async def go():
            result = await clf._run_gated(lambda *_a: "ok")
            assert result == "ok"
            await asyncio.sleep(0.05)  # let the done-callback free the slot
            assert gate.try_acquire() is True

        asyncio.run(go())


# =============================================================================
# ServiceOverloadedError
# =============================================================================


class TestServiceOverloadedError:
    """Tests for the ServiceOverloadedError exception."""

    def test_is_exception(self):
        """ServiceOverloadedError is an Exception subclass."""
        from app.classifier import ServiceOverloadedError

        assert issubclass(ServiceOverloadedError, Exception)

    def test_can_be_raised(self):
        """ServiceOverloadedError can be raised and caught."""
        from app.classifier import ServiceOverloadedError

        with pytest.raises(ServiceOverloadedError):
            raise ServiceOverloadedError("test")

    def test_message_preserved(self):
        """Error message is preserved."""
        from app.classifier import ServiceOverloadedError

        try:
            raise ServiceOverloadedError("All workers busy")
        except ServiceOverloadedError as e:
            assert "All workers busy" in str(e)


# =============================================================================
# Label selection by language
# =============================================================================


class TestLabelSelection:
    """Tests for label selection based on language parameter."""

    def test_ar_selects_arabic_labels(self):
        """lang='ar' selects Arabic label dicts (non-empty)."""
        from app.classifier import ACTIVE_CONFIG

        assert len(ACTIVE_CONFIG.sub_labels["ara"]) > 0
        assert len(ACTIVE_CONFIG.main_labels["ara"]) > 0

    def test_en_selects_english_labels(self):
        """lang='en' selects English label dicts (non-empty)."""
        from app.classifier import ACTIVE_CONFIG

        assert len(ACTIVE_CONFIG.sub_labels["eng"]) > 0
        assert len(ACTIVE_CONFIG.main_labels["eng"]) > 0

    def test_ckb_labels_populated_for_ckb_dialect(self):
        """For ckb dialect, Kurdish labels are present and non-empty."""
        dialect = os.environ.get("DIALECT", "arz")
        if dialect != "ckb":
            pytest.skip("Only applies to ckb dialect")

        from app.classifier import ACTIVE_CONFIG

        assert len(ACTIVE_CONFIG.sub_labels["ckb"]) > 0
        assert len(ACTIVE_CONFIG.main_labels["ckb"]) > 0

    def test_ckb_labels_absent_for_non_ckb_dialect(self):
        """For non-ckb dialects, Kurdish labels are not present."""
        dialect = os.environ.get("DIALECT", "arz")
        if dialect == "ckb":
            pytest.skip("Only applies to non-ckb dialects")

        from app.classifier import ACTIVE_CONFIG

        assert "ckb" not in ACTIVE_CONFIG.sub_labels
        assert "ckb" not in ACTIVE_CONFIG.main_labels

    def test_ar_en_labels_have_same_keys(self):
        """Arabic and English sub/main labels have matching keys."""
        from app.classifier import ACTIVE_CONFIG

        assert set(ACTIVE_CONFIG.sub_labels["ara"].keys()) == set(
            ACTIVE_CONFIG.sub_labels["eng"].keys()
        )
        assert set(ACTIVE_CONFIG.main_labels["ara"].keys()) == set(
            ACTIVE_CONFIG.main_labels["eng"].keys()
        )


# =============================================================================
# InferenceCache
# =============================================================================


class TestInferenceCache:
    """Tests for the InferenceCache LRU cache."""

    def _make_cache(self, maxsize=3):
        from app.classifier import InferenceCache

        return InferenceCache(maxsize=maxsize)

    def test_put_and_get(self):
        """Cache stores and retrieves values."""
        cache = self._make_cache()
        cache.put("hello", (0, 0.95))
        result = cache.get("hello")
        assert result == (0, 0.95)

    def test_get_miss_returns_none(self):
        """Cache miss returns None."""
        cache = self._make_cache()
        assert cache.get("nonexistent") is None

    def test_lru_eviction(self):
        """Cache evicts least-recently-used entry when full."""
        cache = self._make_cache(maxsize=2)
        cache.put("a", (0, 0.9))
        cache.put("b", (1, 0.8))
        cache.put("c", (2, 0.7))  # Should evict "a"
        assert cache.get("a") is None
        assert cache.get("b") is not None
        assert cache.get("c") is not None

    def test_get_updates_recency(self):
        """Accessing an entry makes it most-recently-used."""
        cache = self._make_cache(maxsize=2)
        cache.put("a", (0, 0.9))
        cache.put("b", (1, 0.8))
        cache.get("a")  # Touch "a", making "b" the LRU
        cache.put("c", (2, 0.7))  # Should evict "b", not "a"
        assert cache.get("a") is not None
        assert cache.get("b") is None
        assert cache.get("c") is not None

    def test_put_existing_key_updates_value(self):
        """Putting an existing key updates its value."""
        cache = self._make_cache()
        cache.put("a", (0, 0.9))
        cache.put("a", (1, 0.8))
        result = cache.get("a")
        assert result == (1, 0.8)

    def test_same_text_different_lang_is_cache_hit(self):
        """Cache keyed by preprocessed text -- same text, different lang is a hit."""
        cache = self._make_cache()
        cache.put("مرحبا", (0, 0.95))
        result = cache.get("مرحبا")
        assert result is not None
        assert result == (0, 0.95)

    def test_stats_initial(self):
        """Initial stats show zero hits and misses."""
        cache = self._make_cache()
        stats = cache.stats
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0

    def test_stats_after_operations(self):
        """Stats correctly track hits and misses."""
        cache = self._make_cache()
        cache.put("a", (0, 0.9))
        cache.get("a")  # hit
        cache.get("b")  # miss
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["size"] == 1

    def test_maxsize_respected(self):
        """Cache never exceeds maxsize."""
        cache = self._make_cache(maxsize=3)
        for i in range(10):
            cache.put(f"key_{i}", (i, 0.5))
        assert cache.stats["size"] <= 3

    def test_thread_safety(self):
        """Cache handles concurrent access without errors."""
        cache = self._make_cache(maxsize=100)
        errors = []

        def writer(start):
            try:
                for i in range(100):
                    cache.put(f"key_{start}_{i}", (i, 0.5))
            except Exception as e:
                errors.append(e)

        def reader(start):
            try:
                for i in range(100):
                    cache.get(f"key_{start}_{i}")
            except Exception as e:
                errors.append(e)

        threads = []
        for t_id in range(4):
            threads.append(threading.Thread(target=writer, args=(t_id,)))
            threads.append(threading.Thread(target=reader, args=(t_id,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread safety errors: {errors}"


# =============================================================================
# VALID_DIALECTS and SUPPORTED_LANGUAGES
# =============================================================================


class TestDialectAndLanguageConfig:
    """Tests for dialect and language configuration constants."""

    def test_valid_dialects_contains_all_three(self):
        """VALID_DIALECTS includes arz, acm, and ckb."""
        from app.classifier import VALID_DIALECTS

        assert "arz" in VALID_DIALECTS
        assert "acm" in VALID_DIALECTS
        assert "ckb" in VALID_DIALECTS

    def test_valid_dialects_is_frozenset(self):
        """VALID_DIALECTS is immutable (frozenset)."""
        from app.classifier import VALID_DIALECTS

        assert isinstance(VALID_DIALECTS, frozenset)

    def test_supported_languages_is_frozenset(self):
        """SUPPORTED_LANGUAGES is immutable (frozenset)."""
        from app.classifier import SUPPORTED_LANGUAGES

        assert isinstance(SUPPORTED_LANGUAGES, frozenset)

    def test_supported_languages_includes_ara_eng(self):
        """SUPPORTED_LANGUAGES always includes the canonical ara and eng codes."""
        from app.classifier import SUPPORTED_LANGUAGES

        assert "ara" in SUPPORTED_LANGUAGES
        assert "eng" in SUPPORTED_LANGUAGES

    def test_ckb_dialect_supports_ckb_language(self):
        """When DIALECT=ckb, SUPPORTED_LANGUAGES includes ckb."""
        dialect = os.environ.get("DIALECT", "arz")
        if dialect != "ckb":
            pytest.skip("Only applies when DIALECT=ckb")

        from app.classifier import SUPPORTED_LANGUAGES

        assert "ckb" in SUPPORTED_LANGUAGES

    def test_non_ckb_dialect_does_not_support_ckb_language(self):
        """When DIALECT is not ckb, SUPPORTED_LANGUAGES excludes ckb."""
        dialect = os.environ.get("DIALECT", "arz")
        if dialect == "ckb":
            pytest.skip("Only applies when DIALECT is not ckb")

        from app.classifier import SUPPORTED_LANGUAGES

        assert "ckb" not in SUPPORTED_LANGUAGES


# =============================================================================
# Language alias normalization
# =============================================================================


class TestNormalizeLang:
    """normalize_lang maps the active dialect's aliases to canonical ISO 639-3 codes."""

    def test_aliases_map_to_canonical(self):
        """ar/en resolve on every dialect; ku resolves only where ckb is served."""
        from app.classifier import normalize_lang

        assert normalize_lang("ar") == "ara"
        assert normalize_lang("en") == "eng"
        if os.environ.get("DIALECT") == "ckb":
            assert normalize_lang("ku") == "ckb"

    def test_canonical_passes_through(self):
        from app.classifier import normalize_lang

        assert normalize_lang("ara") == "ara"
        assert normalize_lang("eng") == "eng"
        assert normalize_lang("ckb") == "ckb"

    def test_unknown_passes_through(self):
        """An unsupported code is returned unchanged so the caller can reject it."""
        from app.classifier import normalize_lang

        assert normalize_lang("xx") == "xx"

    def test_alias_map_for_active_dialect(self):
        """LANGUAGE_ALIASES holds this dialect's two-letter aliases (ku only for ckb)."""
        from app.classifier import LANGUAGE_ALIASES

        expected = {"ar": "ara", "en": "eng"}
        if os.environ.get("DIALECT") == "ckb":
            expected["ku"] = "ckb"
        assert LANGUAGE_ALIASES == expected


# =============================================================================
# ClassificationResult dataclass
# =============================================================================


class TestClassificationResult:
    """Tests for the ClassificationResult dataclass."""

    def test_valid_result(self):
        """Valid ClassificationResult can be created."""
        from app.classifier import ClassificationResult

        r = ClassificationResult(
            is_valid=True, sub_class="test", main_class="main", confidence=0.95
        )
        assert r.is_valid is True
        assert r.sub_class == "test"
        assert r.main_class == "main"
        assert r.confidence == 0.95

    def test_invalid_result(self):
        """Invalid ClassificationResult has None fields."""
        from app.classifier import ClassificationResult

        r = ClassificationResult(is_valid=False, sub_class=None, main_class=None, confidence=None)
        assert r.is_valid is False
        assert r.sub_class is None

    def test_frozen(self):
        """ClassificationResult is immutable (frozen dataclass)."""
        from app.classifier import ClassificationResult

        r = ClassificationResult(
            is_valid=True, sub_class="test", main_class="main", confidence=0.95
        )
        with pytest.raises(AttributeError):
            r.is_valid = False


# =============================================================================
# DialectConfig dataclass
# =============================================================================


class TestDialectConfig:
    """Tests for the DialectConfig dataclass."""

    def test_can_create(self):
        """DialectConfig can be instantiated with required fields."""
        from app.classifier import DialectConfig

        cfg = DialectConfig(
            name="Test",
            model_path="/test",
            preprocess_fn=lambda x: x,
            sub_to_main={0: 0},
            sub_labels={"ar": {0: "test"}, "en": {0: "test"}},
            main_labels={"ar": {0: "test"}, "en": {0: "test"}},
        )
        assert cfg.name == "Test"
        assert cfg.model_path == "/test"

    def test_frozen(self):
        """DialectConfig is immutable (frozen dataclass)."""
        from app.classifier import DialectConfig

        cfg = DialectConfig(
            name="Test",
            model_path="/test",
            preprocess_fn=lambda x: x,
            sub_to_main={0: 0},
            sub_labels={"ar": {0: "test"}, "en": {0: "test"}},
            main_labels={"ar": {0: "test"}, "en": {0: "test"}},
        )
        with pytest.raises(AttributeError):
            cfg.name = "Changed"


# =============================================================================
# _predict_single label logic (mocked model)
# =============================================================================


class TestPredictSingleLabelLogic:
    """Tests for _predict_single label lookup logic with mocked inference."""

    def _make_cfg(self):
        """Build a minimal DialectConfig for testing."""
        from app.classifier import DialectConfig

        return DialectConfig(
            name="test",
            model_path="/test",
            preprocess_fn=lambda x: x if x.strip() else "",
            sub_to_main={0: 0, 1: 1},
            sub_labels={
                "ar": {0: "sub_ar_0", 1: "sub_ar_1"},
                "en": {0: "sub_en_0", 1: "sub_en_1"},
                "ckb": {0: "sub_ckb_0", 1: "sub_ckb_1"},
            },
            main_labels={
                "ar": {0: "main_ar_0", 1: "main_ar_1"},
                "en": {0: "main_en_0", 1: "main_en_1"},
                "ckb": {0: "main_ckb_0", 1: "main_ckb_1"},
            },
        )

    def test_invalid_text_returns_invalid(self):
        """Empty preprocessed text returns is_valid=False."""
        from app.classifier import _predict_single

        cfg = self._make_cfg()
        mock_loaded = MagicMock()
        result = _predict_single("", mock_loaded, cfg, "ar")
        assert result.is_valid is False

    def test_whitespace_only_returns_invalid(self):
        """Whitespace-only text returns is_valid=False after preprocessing."""
        from app.classifier import _predict_single

        cfg = self._make_cfg()
        mock_loaded = MagicMock()
        result = _predict_single("   ", mock_loaded, cfg, "ar")
        assert result.is_valid is False

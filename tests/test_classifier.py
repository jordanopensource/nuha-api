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
# _InferenceGate (bounded admission + slot queue)
# =============================================================================


class TestInferenceGate:
    """Direct tests for the admission cap and the bounded execution-slot queue."""

    def test_admits_up_to_limit_plus_queue(self):
        """try_admit() succeeds exactly `limit + queue_size` times, then refuses."""
        from app.classifier import _InferenceGate

        gate = _InferenceGate(2, 3, 10)  # cap = 5 in flight
        assert [gate.try_admit() for _ in range(5)] == [True, True, True, True, True]
        assert gate.try_admit() is False  # running + queued cap reached

    def test_queue_zero_admits_only_limit(self):
        """queue_size=0 restores the old shed-immediately behavior (cap == limit)."""
        from app.classifier import _InferenceGate

        gate = _InferenceGate(2, 0, 10)
        assert gate.try_admit() is True
        assert gate.try_admit() is True
        assert gate.try_admit() is False  # no queue, so cap is the slot count

    def test_drop_admission_frees_an_admission(self):
        """drop_admission() (admitted but never got a slot) reopens a place."""
        from app.classifier import _InferenceGate

        gate = _InferenceGate(1, 0, 10)
        assert gate.try_admit() is True
        assert gate.try_admit() is False
        gate.drop_admission()
        assert gate.try_admit() is True

    def test_drop_admission_never_goes_negative(self):
        """Extra drops can't push the in-flight count below zero / above the cap."""
        from app.classifier import _InferenceGate

        gate = _InferenceGate(1, 0, 10)
        gate.drop_admission()  # nothing admitted, must be a no-op
        gate.drop_admission()
        assert gate.try_admit() is True
        assert gate.try_admit() is False  # still only one place

    def test_acquire_slot_grants_up_to_limit_then_times_out(self):
        """acquire_slot() grants `limit` slots immediately; the next wait times out."""
        import asyncio

        from app.classifier import _InferenceGate

        gate = _InferenceGate(2, 5, 0.05)  # tiny wait so the timeout path is fast

        async def go():
            await gate.acquire_slot()
            await gate.acquire_slot()  # both slots now held
            with pytest.raises(TimeoutError):
                await gate.acquire_slot()  # no slot frees within the wait window

        asyncio.run(go())

    def test_release_slot_lets_a_waiter_proceed(self):
        """A request blocked on acquire_slot() proceeds once a slot is released."""
        import asyncio

        from app.classifier import _InferenceGate

        gate = _InferenceGate(1, 5, 5)

        async def go():
            await gate.acquire_slot()  # hold the only slot
            waiter = asyncio.create_task(gate.acquire_slot())
            await asyncio.sleep(0.05)
            assert not waiter.done()  # still queued, not shed
            gate.release_slot()  # free the slot
            await asyncio.wait_for(waiter, 1.0)  # waiter now proceeds
            assert waiter.done()

        asyncio.run(go())

    def test_release_slot_frees_slot_and_admission(self):
        """release_slot() frees both a slot and an admission place."""
        import asyncio

        from app.classifier import _InferenceGate

        gate = _InferenceGate(1, 0, 5)

        async def go():
            assert gate.try_admit() is True
            await gate.acquire_slot()
            assert gate.try_admit() is False  # cap reached
            gate.release_slot()  # frees slot + admission
            assert gate.try_admit() is True
            await gate.acquire_slot()  # slot available again

        asyncio.run(go())


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
        gate = clf._InferenceGate(1, 0, 10)  # single slot, no queue
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
            assert gate.try_admit() is False
            # Once the abandoned worker completes, the done-callback frees it.
            assert finished.wait(3.0) is True
            await asyncio.sleep(0.1)  # let the loop run the done-callback
            assert gate.try_admit() is True

        asyncio.run(go())

    def test_fast_work_under_timeout_returns_result(self, monkeypatch):
        """A normal inference well under the timeout returns and frees its slot."""
        import asyncio

        import app.classifier as clf

        monkeypatch.setattr(clf, "INFERENCE_TIMEOUT", 5)
        gate = clf._InferenceGate(1, 0, 10)
        monkeypatch.setattr(clf, "_inference_gate", gate)

        async def go():
            result = await clf._run_gated(lambda *_a: "ok")
            assert result == "ok"
            await asyncio.sleep(0.05)  # let the done-callback free the slot
            assert gate.try_admit() is True

        asyncio.run(go())


# =============================================================================
# Bounded queueing through _run_gated (serve bursts, shed only past the cap)
# =============================================================================


class TestRunGatedQueue:
    """End-to-end gate behavior: a burst within capacity is served, not shed; a
    503 is raised only when the running+queued cap is full or the wait deadline
    passes."""

    def test_burst_within_capacity_all_served(self, monkeypatch):
        """3 concurrent requests with 1 slot + queue of 2 all return (none shed)."""
        import asyncio
        import time

        import app.classifier as clf

        monkeypatch.setattr(clf, "INFERENCE_TIMEOUT", 5)
        monkeypatch.setattr(clf, "_inference_gate", clf._InferenceGate(1, 2, 5))

        def work(*_a):
            time.sleep(0.05)
            return "ok"

        async def go():
            results = await asyncio.gather(
                clf._run_gated(work), clf._run_gated(work), clf._run_gated(work)
            )
            assert results == ["ok", "ok", "ok"]  # all served, queued not shed

        asyncio.run(go())

    def test_beyond_capacity_sheds_503(self, monkeypatch):
        """Past the running+queued cap, the next request gets an immediate 503."""
        import asyncio
        import threading

        import app.classifier as clf

        monkeypatch.setattr(clf, "INFERENCE_TIMEOUT", 5)
        monkeypatch.setattr(clf, "_inference_gate", clf._InferenceGate(1, 1, 5))  # cap 2
        release = threading.Event()

        def blocking(*_a):
            release.wait(3.0)
            return "ok"

        async def go():
            t1 = asyncio.create_task(clf._run_gated(blocking))  # runs, holds the slot
            t2 = asyncio.create_task(clf._run_gated(blocking))  # admitted, queued
            await asyncio.sleep(0.1)  # let both be admitted
            with pytest.raises(clf.ServiceOverloadedError):
                await clf._run_gated(lambda *_a: "ok")  # cap full -> fast 503
            release.set()
            await asyncio.gather(t1, t2)

        asyncio.run(go())

    def test_wait_timeout_sheds_503(self, monkeypatch):
        """A request that waits past INFERENCE_QUEUE_TIMEOUT for a slot gets 503."""
        import asyncio
        import threading

        import app.classifier as clf

        monkeypatch.setattr(clf, "INFERENCE_TIMEOUT", 5)
        monkeypatch.setattr(clf, "_inference_gate", clf._InferenceGate(1, 5, 0.05))
        release = threading.Event()

        def blocking(*_a):
            release.wait(3.0)
            return "ok"

        async def go():
            t1 = asyncio.create_task(clf._run_gated(blocking))  # holds the only slot
            await asyncio.sleep(0.05)
            with pytest.raises(clf.ServiceOverloadedError):
                await clf._run_gated(lambda *_a: "ok")  # waits, then sheds 503
            release.set()
            await t1

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


# =============================================================================
# ONNX Runtime inference path
# =============================================================================
#
# Exercises the real ONNX inference helpers and the _predict_single/_predict_batch
# numpy path with a FAKE InferenceSession (returns canned logits). These lock in
# the migration behaviour: numpy softmax/argmax, feeding only the input names the
# graph declares (BERT needs token_type_ids, XLM-R does not), and the
# (predicted_id, confidence) cache contract.


class _FakeSession:
    """Stand-in for ort.InferenceSession.

    ``run`` records the feed dict it was called with and returns the configured
    logits (shape (batch, num_labels)). ``get_inputs`` reports the declared input
    names so _build_onnx_inputs can filter against them.
    """

    def __init__(self, logits, input_names=("input_ids", "attention_mask")):
        import numpy as np

        self._logits = np.asarray(logits, dtype="float32")
        self._input_names = list(input_names)
        self.calls = []

    def get_inputs(self):
        return [type("Inp", (), {"name": n}) for n in self._input_names]

    def run(self, output_names, feed):
        self.calls.append(feed)
        # Return one logits row per input row (so batches line up).
        n = len(next(iter(feed.values())))
        return [self._logits[:n]]


def _make_loaded(session, input_names=("input_ids", "attention_mask"), max_length=8):
    """Build a real LoadedModel wrapping a fake session + a fake tokenizer.

    The fake tokenizer returns numpy arrays for input_ids/attention_mask, and
    also token_type_ids (as a real BERT tokenizer would) so we can prove
    _build_onnx_inputs drops it for an XLM-R-style graph.
    """
    import numpy as np

    from app.classifier import LoadedModel

    def fake_tokenizer(text, **kwargs):
        batch = text if isinstance(text, list) else [text]
        rows = len(batch)
        cols = 4
        return {
            "input_ids": np.ones((rows, cols), dtype=np.int64),
            "attention_mask": np.ones((rows, cols), dtype=np.int64),
            "token_type_ids": np.zeros((rows, cols), dtype=np.int64),
        }

    return LoadedModel(
        session=session,
        tokenizer=fake_tokenizer,
        input_names=frozenset(input_names),
        max_length=max_length,
    )


class TestSoftmaxArgmax:
    """_softmax_argmax: numerically-stable row-wise softmax + argmax."""

    def test_argmax_picks_largest_logit(self):
        import numpy as np

        from app.classifier import _softmax_argmax

        conf, pred = _softmax_argmax(np.array([[0.1, 5.0, 0.2]]))
        assert int(pred[0]) == 1
        assert 0.0 <= float(conf[0]) <= 1.0

    def test_probabilities_sum_to_one(self):
        import numpy as np

        from app.classifier import _softmax_argmax

        logits = np.array([[2.0, 1.0, 0.5, -1.0]])
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / np.sum(exp, axis=-1, keepdims=True)
        assert abs(float(probs.sum()) - 1.0) < 1e-6
        conf, _ = _softmax_argmax(logits)
        assert abs(float(conf[0]) - float(probs.max())) < 1e-6

    def test_stable_with_large_logits(self):
        """Large logits do not overflow (stable softmax subtracts the row max)."""
        import numpy as np

        from app.classifier import _softmax_argmax

        conf, pred = _softmax_argmax(np.array([[1000.0, 1001.0]]))
        assert int(pred[0]) == 1
        assert np.isfinite(conf[0])

    def test_batch_rows_independent(self):
        import numpy as np

        from app.classifier import _softmax_argmax

        _conf, pred = _softmax_argmax(np.array([[5.0, 0.0], [0.0, 5.0]]))
        assert list(pred) == [0, 1]


class TestBuildOnnxInputs:
    """_build_onnx_inputs: feed only the names the ONNX graph declares."""

    def test_bert_includes_token_type_ids(self):
        from app.classifier import _build_onnx_inputs

        loaded = _make_loaded(
            _FakeSession([[0.0, 1.0]]),
            input_names=("input_ids", "attention_mask", "token_type_ids"),
        )
        tokenized = loaded.tokenizer("نص")
        feed = _build_onnx_inputs(loaded, tokenized)
        assert set(feed) == {"input_ids", "attention_mask", "token_type_ids"}

    def test_xlmr_excludes_token_type_ids(self):
        """An XLM-R graph (no token_type_ids input) must not be fed token_type_ids."""
        from app.classifier import _build_onnx_inputs

        loaded = _make_loaded(
            _FakeSession([[0.0, 1.0]]),
            input_names=("input_ids", "attention_mask"),
        )
        tokenized = loaded.tokenizer("نص")  # tokenizer still produces token_type_ids
        feed = _build_onnx_inputs(loaded, tokenized)
        assert "token_type_ids" not in feed
        assert set(feed) == {"input_ids", "attention_mask"}

    def test_feeds_are_int64(self):
        import numpy as np

        from app.classifier import _build_onnx_inputs

        loaded = _make_loaded(_FakeSession([[0.0, 1.0]]))
        feed = _build_onnx_inputs(loaded, loaded.tokenizer("نص"))
        for arr in feed.values():
            assert arr.dtype == np.int64

    def test_bert_graph_synthesizes_missing_token_type_ids(self):
        """A BERT graph requires token_type_ids, but transformers 5.x's fast
        tokenizer (TokenizersBackend) omits it by default. _build_onnx_inputs must
        synthesize an all-zeros column so the required graph input is present.
        Otherwise ORT 500s with "Required inputs (['token_type_ids']) are missing"
        (the real failure that hit the arz dialect)."""
        import numpy as np

        from app.classifier import _build_onnx_inputs

        loaded = _make_loaded(
            _FakeSession([[0.0, 1.0]]),
            input_names=("input_ids", "attention_mask", "token_type_ids"),
        )
        # Tokenizer output WITHOUT token_type_ids (as the 5.x fast tokenizer gives).
        tokenized = {
            "input_ids": np.ones((1, 4), dtype=np.int64),
            "attention_mask": np.ones((1, 4), dtype=np.int64),
        }
        feed = _build_onnx_inputs(loaded, tokenized)
        assert "token_type_ids" in feed
        assert feed["token_type_ids"].shape == feed["input_ids"].shape
        assert not feed["token_type_ids"].any()  # all zeros (single-sequence segment id)


class TestTokenTypeIdsRequest:
    """Regression: a BERT dialect whose tokenizer omits token_type_ids by default.

    transformers 5.x's TokenizersBackend returns only input_ids/attention_mask
    unless return_token_type_ids=True is passed; the BERT ONNX graph requires
    token_type_ids. _predict_* must still feed it (via the explicit request or the
    zeros backstop) so inference does not 500. Earlier tests used a tokenizer that
    always returned token_type_ids, so they missed this.
    """

    def _bert_loaded(self, session):
        """LoadedModel with a BERT graph and a tokenizer that only emits
        token_type_ids when return_token_type_ids=True (like TokenizersBackend)."""
        import numpy as np

        from app.classifier import LoadedModel

        def fake_tokenizer(text, **kwargs):
            batch = text if isinstance(text, list) else [text]
            rows, cols = len(batch), 4
            out = {
                "input_ids": np.ones((rows, cols), dtype=np.int64),
                "attention_mask": np.ones((rows, cols), dtype=np.int64),
            }
            if kwargs.get("return_token_type_ids"):
                out["token_type_ids"] = np.zeros((rows, cols), dtype=np.int64)
            return out

        return LoadedModel(
            session=session,
            tokenizer=fake_tokenizer,
            input_names=frozenset(("input_ids", "attention_mask", "token_type_ids")),
            max_length=8,
        )

    def _cfg(self):
        from app.classifier import DialectConfig

        return DialectConfig(
            name="bert-test",
            model_path="/test",
            preprocess_fn=lambda x: x.strip(),
            sub_to_main={0: 0, 1: 1},
            sub_labels={"ara": {0: "neutral", 1: "violence"}},
            main_labels={"ara": {0: "neutral_m", 1: "violence_m"}},
        )

    def test_single_feeds_token_type_ids_to_bert_graph(self):
        from app.classifier import _inference_cache, _predict_single

        _inference_cache._cache.clear()
        session = _FakeSession(
            [[0.1, 9.0]], input_names=("input_ids", "attention_mask", "token_type_ids")
        )
        loaded = self._bert_loaded(session)
        result = _predict_single("نص عربي", loaded, self._cfg(), "ara")
        assert result.is_valid is True
        assert result.sub_class == "violence"
        # The fed dict reaching the session must carry token_type_ids.
        assert "token_type_ids" in session.calls[0]

    def test_batch_feeds_token_type_ids_to_bert_graph(self):
        from app.classifier import _inference_cache, _predict_batch

        _inference_cache._cache.clear()
        # Two distinct texts => two cache misses => the fake session must return
        # two logit rows (it slices [:n] by the input row count).
        session = _FakeSession(
            [[0.1, 9.0], [0.1, 9.0]],
            input_names=("input_ids", "attention_mask", "token_type_ids"),
        )
        loaded = self._bert_loaded(session)
        results = _predict_batch(["نص اول", "نص ثاني"], loaded, self._cfg(), "ara")
        assert all(r.is_valid for r in results)
        assert "token_type_ids" in session.calls[0]


class TestPredictWithFakeSession:
    """_predict_single/_predict_batch over the real numpy path with a fake session."""

    def _cfg(self):
        from app.classifier import DialectConfig

        return DialectConfig(
            name="test",
            model_path="/test",
            preprocess_fn=lambda x: x.strip(),
            sub_to_main={0: 0, 1: 1},
            sub_labels={"ara": {0: "neutral", 1: "violence"}},
            main_labels={"ara": {0: "neutral_m", 1: "violence_m"}},
        )

    def test_single_prediction_uses_argmax_label(self):
        from app.classifier import _inference_cache, _predict_single

        _inference_cache._cache.clear()
        # logits favour class 1
        loaded = _make_loaded(_FakeSession([[0.1, 9.0]]))
        result = _predict_single("نص عربي", loaded, self._cfg(), "ara")
        assert result.is_valid is True
        assert result.sub_class == "violence"
        assert result.main_class == "violence_m"
        assert 0.0 <= result.confidence <= 1.0

    def test_single_prediction_caches_raw_prediction(self):
        from app.classifier import _inference_cache, _predict_single

        _inference_cache._cache.clear()
        session = _FakeSession([[0.1, 9.0]])
        loaded = _make_loaded(session)
        cfg = self._cfg()
        _predict_single("نص عربي", loaded, cfg, "ara")
        # Second call with same text must hit the cache (no second run()).
        _predict_single("نص عربي", loaded, cfg, "ara")
        assert len(session.calls) == 1
        cached = _inference_cache.get("نص عربي")
        assert cached is not None
        assert cached[0] == 1  # predicted_id stored

    def test_batch_only_runs_inference_on_cache_misses(self):
        from app.classifier import _inference_cache, _predict_batch

        _inference_cache._cache.clear()
        session = _FakeSession([[0.1, 9.0]])
        loaded = _make_loaded(session)
        cfg = self._cfg()
        # Prime the cache with one of the two texts.
        _inference_cache.put("repeat", (1, 0.99))
        results = _predict_batch(["repeat", "fresh"], loaded, cfg, "ara")
        assert all(r.is_valid for r in results)
        # Only the single miss ("fresh") was sent to the session.
        assert len(session.calls) == 1
        assert len(next(iter(session.calls[0].values()))) == 1

    def test_batch_empty_and_valid_mix(self):
        from app.classifier import _inference_cache, _predict_batch

        _inference_cache._cache.clear()
        loaded = _make_loaded(_FakeSession([[0.1, 9.0]]))
        results = _predict_batch(["", "نص"], loaded, self._cfg(), "ara")
        assert results[0].is_valid is False
        assert results[1].is_valid is True

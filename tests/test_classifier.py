"""Tests for the classification engine.

Directory parsing, executor management, the admission gate, the inference
cache, label selection, and error handling, all with mocked models (no ML
framework dependency). Per-dialect state lives on ``LoadedDialect`` bundles
now, so these tests build entries via ``_make_entry`` with the fabricated
code "tst" (no dialect file declares it, keeping the suite's
no-hardcoded-dialects rule intact).
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from tests.conftest import ALL_DIALECTS, make_model_volume


# =============================================================================
# parse_dialect_dir: a model directory becomes a complete DialectConfig
# =============================================================================


class TestParseDialectDir:
    """parse_dialect_dir assembles the full per-dialect config from a model
    directory (the registry's completeness check), for whatever dialect files
    exist; broken directories raise instead of half-loading."""

    @pytest.mark.parametrize("code", ALL_DIALECTS)
    def test_builds_complete_config(self, tmp_path, code):
        from app.classifier import parse_dialect_dir

        volume = make_model_volume(tmp_path / "models", codes=(code,))
        cfg, config, onnx_path = parse_dialect_dir(code, volume / code)
        assert config.name == cfg["name"]
        assert callable(config.preprocess_fn)
        assert config.sub_to_main
        assert config.sub_labels and config.main_labels
        for lang_map in (*config.sub_labels.values(), *config.main_labels.values()):
            assert lang_map
        assert onnx_path == volume / code / "model.onnx"

    def test_schema_problems_raise(self, tmp_path, any_dialect):
        import json

        from app.classifier import parse_dialect_dir

        volume = make_model_volume(tmp_path / "models", codes=(any_dialect,))
        path = volume / any_dialect / "dialect.json"
        cfg = json.loads(path.read_text())
        del cfg["labels"]
        path.write_text(json.dumps(cfg))
        with pytest.raises(RuntimeError, match="Invalid dialect config"):
            parse_dialect_dir(any_dialect, volume / any_dialect)

    def test_unknown_preprocessing_type_raises(self, tmp_path, any_dialect):
        """The schema deliberately does not know the preprocessor registry; the
        engine's own build step is what rejects an unknown type."""
        import json

        from app.classifier import parse_dialect_dir

        volume = make_model_volume(tmp_path / "models", codes=(any_dialect,))
        path = volume / any_dialect / "dialect.json"
        cfg = json.loads(path.read_text())
        cfg["preprocessing"] = {"type": "no-such-preprocessor"}
        path.write_text(json.dumps(cfg))
        with pytest.raises(RuntimeError, match="Unknown preprocessing type"):
            parse_dialect_dir(any_dialect, volume / any_dialect)


# =============================================================================
# Executor and semaphore
# =============================================================================


class TestExecutorManagement:
    """Tests for thread pool executor and semaphore management."""

    @pytest.fixture(autouse=True)
    def _reset_executor_global(self):
        """Null the module-global executor after each test so a live pool one of
        these tests created isn't inherited by later tests; _get_executor()
        lazily recreates from the None baseline."""
        yield
        import app.classifier as clf

        clf.shutdown_executor()

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
# Language maps
# =============================================================================


class TestBuildLanguageMaps:
    """build_language_maps derives (supported, default, aliases) from a
    dialect's languages block, checked against every shipped dialect file."""

    @pytest.mark.parametrize("code", ALL_DIALECTS)
    def test_maps_mirror_the_dialect_file(self, code):
        from app.classifier import build_language_maps
        from tests.conftest import _DIALECT_FILES

        languages = _DIALECT_FILES[code]["languages"]
        supported, default, aliases = build_language_maps(languages)
        assert supported == frozenset(languages)
        assert default == sorted(languages)[0]
        for canonical, meta in languages.items():
            for alias in meta.get("aliases", []):
                assert aliases[alias] == canonical

    def test_canonical_codes_not_in_alias_map(self):
        """Canonical codes pass through lookups unchanged because they are NOT
        alias keys; an unknown code passes through too, so the caller still
        validates against the supported set."""
        from app.classifier import build_language_maps

        _supported, _default, aliases = build_language_maps(
            {"aaa": {"name": "A", "aliases": ["a"]}, "bbb": {"name": "B"}}
        )
        assert aliases == {"a": "aaa"}
        assert aliases.get("aaa", "aaa") == "aaa"
        assert aliases.get("zz", "zz") == "zz"


# =============================================================================
# Result/config dataclasses
# =============================================================================


class TestDataclassImmutability:
    """ClassificationResult and DialectConfig are frozen: results and the shared
    config are handed across threads, so accidental mutation must fail loudly."""

    def test_classification_result_frozen(self):
        from app.classifier import ClassificationResult

        r = ClassificationResult(
            is_valid=True, sub_class="test", main_class="main", confidence=0.95
        )
        with pytest.raises(AttributeError):
            r.is_valid = False

    def test_dialect_config_frozen(self):
        from app.classifier import DialectConfig

        cfg = DialectConfig(
            name="Test",
            preprocess_fn=lambda x: x,
            sub_to_main={0: 0},
            sub_labels={"ar": {0: "test"}},
            main_labels={"ar": {0: "test"}},
        )
        with pytest.raises(AttributeError):
            cfg.name = "Changed"


# =============================================================================
# _predict_single label logic (mocked model)
# =============================================================================


class TestPredictSingleLabelLogic:
    """_predict_single's invalid-input short circuit (empty/whitespace input
    never reaches the model). The label-lookup happy path is covered with a
    fake session in TestPredictWithFakeSession."""

    def _make_cfg(self):
        """Build a minimal DialectConfig for testing."""
        from app.classifier import DialectConfig

        return DialectConfig(
            name="test",
            preprocess_fn=lambda x: x if x.strip() else "",
            sub_to_main={0: 0, 1: 1},
            sub_labels={
                "ar": {0: "sub_ar_0", 1: "sub_ar_1"},
                "en": {0: "sub_en_0", 1: "sub_en_1"},
                "xx": {0: "sub_xx_0", 1: "sub_xx_1"},
            },
            main_labels={
                "ar": {0: "main_ar_0", 1: "main_ar_1"},
                "en": {0: "main_en_0", 1: "main_en_1"},
                "xx": {0: "main_xx_0", 1: "main_xx_1"},
            },
        )

    def test_invalid_text_returns_invalid(self):
        """Empty preprocessed text returns is_valid=False."""
        from app.classifier import _predict_single

        entry = _make_entry(self._make_cfg(), MagicMock())
        result = _predict_single("", entry, "ar")
        assert result.is_valid is False

    def test_whitespace_only_returns_invalid(self):
        """Whitespace-only text returns is_valid=False after preprocessing."""
        from app.classifier import _predict_single

        entry = _make_entry(self._make_cfg(), MagicMock())
        result = _predict_single("   ", entry, "ar")
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


def _make_entry(cfg, loaded, code="tst"):
    """Wrap a config + loaded model into a LoadedDialect bundle with a fresh
    per-entry cache and tokenizer lock. "tst" is a fabricated code no dialect
    file declares, keeping the suite's no-hardcoded-dialects rule intact."""
    from pathlib import Path

    from app.classifier import InferenceCache, LoadedDialect

    supported = frozenset(cfg.sub_labels)
    return LoadedDialect(
        code=code,
        path=Path("/nonexistent"),
        config=cfg,
        loaded=loaded,
        supported_languages=supported,
        default_language=sorted(supported)[0],
        aliases={},
        cache=InferenceCache(64),
        tokenizer_lock=threading.Lock(),
    )


class TestFindOnnxFile:
    """_find_onnx_file: locate the single ONNX graph in a model dir. Pure
    filesystem logic; reached in production only via the patched-out load_model,
    so its selection + error paths are pinned directly here."""

    def test_prefers_model_onnx(self, tmp_path):
        from app.classifier import _find_onnx_file

        (tmp_path / "model.onnx").write_bytes(b"")
        (tmp_path / "other_export.onnx").write_bytes(b"")
        assert _find_onnx_file(tmp_path) == tmp_path / "model.onnx"

    def test_falls_back_to_single_onnx(self, tmp_path):
        from app.classifier import _find_onnx_file

        (tmp_path / "export.onnx").write_bytes(b"")
        assert _find_onnx_file(tmp_path) == tmp_path / "export.onnx"

    def test_raises_when_none(self, tmp_path):
        import pytest

        from app.classifier import _find_onnx_file

        with pytest.raises(RuntimeError, match="No ONNX model"):
            _find_onnx_file(tmp_path)

    def test_raises_when_ambiguous(self, tmp_path):
        import pytest

        from app.classifier import _find_onnx_file

        (tmp_path / "a.onnx").write_bytes(b"")
        (tmp_path / "b.onnx").write_bytes(b"")
        with pytest.raises(RuntimeError, match="Multiple"):
            _find_onnx_file(tmp_path)


class TestSoftmaxArgmax:
    """_softmax_argmax: numerically-stable row-wise softmax + argmax."""

    def test_argmax_picks_largest_logit(self):
        import numpy as np

        from app.classifier import _softmax_argmax

        conf, pred = _softmax_argmax(np.array([[0.1, 5.0, 0.2]]))
        assert int(pred[0]) == 1
        assert 0.0 <= float(conf[0]) <= 1.0

    def test_top_probability_matches_hand_computed_value(self):
        """Independent oracle: the top softmax probability is a constant computed
        by hand, so a bug in the function's own softmax lines (wrong axis, no
        max-subtraction) cannot silently reproduce the expected value and pass.
        For logits [2, 1, 0.5, -1] the denominator is
        exp(0)+exp(-1)+exp(-1.5)+exp(-3) = 1.640797, so the top prob (class 0)
        is 1/1.640797 = 0.609460."""
        import numpy as np

        from app.classifier import _softmax_argmax

        conf, pred = _softmax_argmax(np.array([[2.0, 1.0, 0.5, -1.0]]))
        assert int(pred[0]) == 0
        assert abs(float(conf[0]) - 0.609460) < 1e-5

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
        (the real failure this reproduces for BERT-family graphs)."""
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
            preprocess_fn=lambda x: x.strip(),
            sub_to_main={0: 0, 1: 1},
            sub_labels={"ara": {0: "neutral", 1: "violence"}},
            main_labels={"ara": {0: "neutral_m", 1: "violence_m"}},
        )

    def test_single_feeds_token_type_ids_to_bert_graph(self):
        from app.classifier import _predict_single

        session = _FakeSession(
            [[0.1, 9.0]], input_names=("input_ids", "attention_mask", "token_type_ids")
        )
        entry = _make_entry(self._cfg(), self._bert_loaded(session))
        result = _predict_single("نص عربي", entry, "ara")
        assert result.is_valid is True
        assert result.sub_class == "violence"
        # The fed dict reaching the session must carry token_type_ids.
        assert "token_type_ids" in session.calls[0]

    def test_batch_feeds_token_type_ids_to_bert_graph(self):
        from app.classifier import _predict_batch

        # Two distinct texts => two cache misses => the fake session must return
        # two logit rows (it slices [:n] by the input row count).
        session = _FakeSession(
            [[0.1, 9.0], [0.1, 9.0]],
            input_names=("input_ids", "attention_mask", "token_type_ids"),
        )
        entry = _make_entry(self._cfg(), self._bert_loaded(session))
        results = _predict_batch(["نص اول", "نص ثاني"], entry, "ara")
        assert all(r.is_valid for r in results)
        assert "token_type_ids" in session.calls[0]


class TestPredictWithFakeSession:
    """_predict_single/_predict_batch over the real numpy path with a fake session."""

    def _cfg(self):
        from app.classifier import DialectConfig

        return DialectConfig(
            name="test",
            preprocess_fn=lambda x: x.strip(),
            sub_to_main={0: 0, 1: 1},
            sub_labels={"ara": {0: "neutral", 1: "violence"}},
            main_labels={"ara": {0: "neutral_m", 1: "violence_m"}},
        )

    def test_single_prediction_uses_argmax_label(self):
        from app.classifier import _predict_single

        # logits favour class 1
        entry = _make_entry(self._cfg(), _make_loaded(_FakeSession([[0.1, 9.0]])))
        result = _predict_single("نص عربي", entry, "ara")
        assert result.is_valid is True
        assert result.sub_class == "violence"
        assert result.main_class == "violence_m"
        assert 0.0 <= result.confidence <= 1.0

    def test_single_prediction_caches_raw_prediction(self):
        from app.classifier import _predict_single

        session = _FakeSession([[0.1, 9.0]])
        entry = _make_entry(self._cfg(), _make_loaded(session))
        _predict_single("نص عربي", entry, "ara")
        # Second call with same text must hit the entry's cache (no second run()).
        _predict_single("نص عربي", entry, "ara")
        assert len(session.calls) == 1
        cached = entry.cache.get("نص عربي")
        assert cached is not None
        assert cached[0] == 1  # predicted_id stored

    def test_batch_only_runs_inference_on_cache_misses(self):
        from app.classifier import _predict_batch

        session = _FakeSession([[0.1, 9.0]])
        entry = _make_entry(self._cfg(), _make_loaded(session))
        # Prime the entry's cache with one of the two texts.
        entry.cache.put("repeat", (1, 0.99))
        results = _predict_batch(["repeat", "fresh"], entry, "ara")
        assert all(r.is_valid for r in results)
        # Only the single miss ("fresh") was sent to the session.
        assert len(session.calls) == 1
        assert len(next(iter(session.calls[0].values()))) == 1

    def test_batch_empty_and_valid_mix(self):
        from app.classifier import _predict_batch

        entry = _make_entry(self._cfg(), _make_loaded(_FakeSession([[0.1, 9.0]])))
        results = _predict_batch(["", "نص"], entry, "ara")
        assert results[0].is_valid is False
        assert results[1].is_valid is True

    def test_same_text_on_two_entries_never_shares_a_cache(self):
        """Behavioral isolation: caching a text on one dialect must not answer
        another dialect's request for the same text (keys are preprocessed
        text only, so a shared cache would cross-serve predictions)."""
        from app.classifier import _predict_single

        session_a, session_b = _FakeSession([[0.1, 9.0]]), _FakeSession([[9.0, 0.1]])
        entry_a = _make_entry(self._cfg(), _make_loaded(session_a), code="tst")
        entry_b = _make_entry(self._cfg(), _make_loaded(session_b), code="tsu")
        _predict_single("نص مشترك", entry_a, "ara")
        result_b = _predict_single("نص مشترك", entry_b, "ara")
        # B ran its OWN inference (no cross-entry cache hit) and got its own
        # session's answer, not A's cached one.
        assert len(session_a.calls) == 1
        assert len(session_b.calls) == 1
        assert result_b.sub_class == "neutral"


class TestTokenizerSerialization:
    """Regression: tokenizer calls are serialized across worker threads.

    The HF fast tokenizer is one shared Rust object whose truncation/padding
    state is MUTATED per call; the single path (padding=False) and the batch
    path (padding=True) calling it from two pool threads at once is the
    "RuntimeError: Already borrowed" race (huggingface/tokenizers#537).
    _predict_single/_predict_batch must never be inside the tokenizer
    concurrently (the per-dialect tokenizer_lock on LoadedDialect). Inference
    itself stays parallel; only tokenization is serialized.
    """

    def test_concurrent_single_and_batch_never_overlap_in_tokenizer(self):
        import threading
        import time

        import numpy as np

        from app.classifier import (
            DialectConfig,
            LoadedModel,
            _predict_batch,
            _predict_single,
        )

        counter_lock = threading.Lock()
        in_tokenizer = 0
        max_concurrent = 0

        def tracking_tokenizer(text, **kwargs):
            """Counts concurrent entries; sleeps to widen any race window."""
            nonlocal in_tokenizer, max_concurrent
            with counter_lock:
                in_tokenizer += 1
                max_concurrent = max(max_concurrent, in_tokenizer)
            time.sleep(0.03)  # a real tokenize is µs; exaggerate the window
            with counter_lock:
                in_tokenizer -= 1
            batch = text if isinstance(text, list) else [text]
            rows = len(batch)
            return {
                "input_ids": np.ones((rows, 4), dtype=np.int64),
                "attention_mask": np.ones((rows, 4), dtype=np.int64),
            }

        loaded = LoadedModel(
            session=_FakeSession([[0.1, 9.0], [0.1, 9.0]]),
            tokenizer=tracking_tokenizer,
            input_names=frozenset(("input_ids", "attention_mask")),
            max_length=8,
        )
        cfg = DialectConfig(
            name="test",
            preprocess_fn=lambda x: x.strip(),
            sub_to_main={0: 0, 1: 1},
            sub_labels={"ara": {0: "neutral", 1: "violence"}},
            main_labels={"ara": {0: "neutral_m", 1: "violence_m"}},
        )

        entry = _make_entry(cfg, loaded)
        errors = []

        def run(fn, arg):
            try:
                fn(arg, entry, "ara")
            except Exception as e:  # the race would surface as an exception here
                errors.append(e)

        # Distinct texts so nothing is served from the cache. Mirrors real mixed
        # load: singles and batches tokenizing from separate worker threads.
        threads = [
            threading.Thread(target=run, args=(_predict_single, "نص اول")),
            threading.Thread(target=run, args=(_predict_batch, ["نص ثاني", "نص ثالث"])),
            threading.Thread(target=run, args=(_predict_single, "نص رابع")),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"prediction raised under concurrency: {errors}"
        assert max_concurrent == 1, (
            f"{max_concurrent} threads were inside the tokenizer at once; "
            f"tokenization must be serialized (see LoadedDialect.tokenizer_lock)"
        )


# =============================================================================
# check_label_consistency: refuse a mis-packaged model directory at load
# =============================================================================


class TestLabelConsistencyCheck:
    """check_label_consistency fails a dialect's load when the ONNX graph's
    output width disagrees with its declared label taxonomy, catching a
    mis-packaged model directory before a single silently-wrong label."""

    def _loaded_with_width(self, width):
        """A LoadedModel-shaped mock whose ONNX graph declares `width` output
        classes (or a symbolic dim if width is a str)."""
        out = MagicMock()
        out.shape = [None, width]
        loaded = MagicMock()
        loaded.session.get_outputs.return_value = [out]
        return loaded

    def test_matching_width_passes(self):
        from app.classifier import check_label_consistency

        # Must not raise when the graph width equals the declared taxonomy size.
        check_label_consistency(self._loaded_with_width(6), 6, "tst")

    def test_mismatched_width_fails_fast(self):
        from app.classifier import check_label_consistency

        with pytest.raises(RuntimeError, match="mis-packaged"):
            check_label_consistency(self._loaded_with_width(7), 6, "tst")

    def test_symbolic_width_is_skipped(self):
        """When the graph leaves the last logits dim symbolic, the check can't
        compare and must NOT raise (some exports are dynamic)."""
        from app.classifier import check_label_consistency

        check_label_consistency(self._loaded_with_width("num_labels"), 6, "tst")


# =============================================================================
# The async entry points, executed for real (gate + executor + predict)
# =============================================================================


class TestAsyncEntryPoints:
    """get_classification / get_classifications_batch run the full path: the
    admission gate, the thread pool, and the real predict functions."""

    @pytest.fixture(autouse=True)
    def _reset_executor_global(self):
        yield
        import app.classifier as clf

        clf.shutdown_executor()

    def _entry(self, logits):
        cfg_holder = TestPredictWithFakeSession()
        return _make_entry(cfg_holder._cfg(), _make_loaded(_FakeSession(logits)))

    def test_single_returns_a_labeled_result(self):
        import asyncio

        from app.classifier import get_classification

        entry = self._entry([[0.1, 9.0]])
        result = asyncio.run(get_classification(entry, "نص عربي", "ara"))
        assert result.is_valid is True
        assert result.sub_class == "violence"
        assert 0.0 <= result.confidence <= 1.0

    def test_batch_returns_results_in_order(self):
        import asyncio

        from app.classifier import get_classifications_batch

        entry = self._entry([[0.1, 9.0], [0.1, 9.0]])
        results = asyncio.run(get_classifications_batch(entry, ["نص اول", "", "نص ثاني"], "ara"))
        assert [r.is_valid for r in results] == [True, False, True]


# =============================================================================
# _load_model metadata handling: the snapshot's max_length is clamped
# =============================================================================


class TestLoadModelMetadata:
    """training_config.json is model-repo DATA: its max_length is honored only
    when it is a sane integer, so a hostile or corrupt snapshot cannot inflate
    per-request tokenizer allocations."""

    def _load(self, tmp_path, training_config=None):
        import json as _json

        from app.classifier import _load_model

        (tmp_path / "model.onnx").write_bytes(b"")
        if training_config is not None:
            (tmp_path / "training_config.json").write_text(_json.dumps(training_config))
        return _load_model(tmp_path, tmp_path / "model.onnx")

    def test_missing_config_defaults_to_128(self, tmp_path):
        assert self._load(tmp_path).max_length == 128

    def test_sane_value_is_honored(self, tmp_path):
        assert self._load(tmp_path, {"max_length": 256}).max_length == 256

    @pytest.mark.parametrize("bad", [10**9, 0, -5, "512", True, None, 4097])
    def test_out_of_bounds_or_non_int_falls_back_to_128(self, tmp_path, bad):
        assert self._load(tmp_path, {"max_length": bad}).max_length == 128

    def test_upper_bound_is_inclusive(self, tmp_path):
        from app.classifier import _MAX_TOKENIZER_LEN

        assert self._load(tmp_path, {"max_length": _MAX_TOKENIZER_LEN}).max_length == (
            _MAX_TOKENIZER_LEN
        )

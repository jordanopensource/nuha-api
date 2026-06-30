"""Shared fixtures for the Nuha API test suite.

app/classifier.py imports onnxruntime (for inference) and AutoTokenizer from
transformers. The test suite runs without the real onnxruntime wheel or any
model files, so we inject mocks for the ML stack BEFORE any app code is imported
(onnxruntime is always mocked; transformers' deep import chain can fail on some
hosts due to missing native libs like libprotobuf, so we mock the breaking
sub-modules and the tokenizer/model classes too).
"""

import contextlib
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "app"
DIALECTS_DIR = APP_DIR / "dialects"

ALL_DIALECTS = ("arz", "acm", "ckb")


def _dialect_file(code: str) -> Path:
    """Path to a single dialect's config file."""
    return DIALECTS_DIR / f"{code}.json"


# ---------------------------------------------------------------------------
# Mock the ML stack before app code is imported
# ---------------------------------------------------------------------------
# app/classifier.py imports `onnxruntime` (inference) and, from transformers,
# `AutoTokenizer`. Tests must run without the real onnxruntime wheel or any model
# files, so we inject a MagicMock for `onnxruntime` here, before classifier is
# imported. (numpy is a real, lightweight dependency and is used as-is.)
#
# transformers' top-level package uses lazy imports (__getattr__). When
# classifier.py does `from transformers import AutoTokenizer`, it can trigger a
# deep import chain (modeling_auto -> auto_factory -> generation -> sklearn ->
# pyarrow -> libprotobuf.so) that fails on some hosts. We pre-inject mock modules
# for that chain so the real import never reaches pyarrow, and make AutoTokenizer
# resolve to a MagicMock. If transformers is not installed at all, we mock the
# top-level package too so the suite still runs (it never does real inference).

# onnxruntime is always mocked: no native runtime needed for contract/logic tests.
_saved_onnxruntime = sys.modules.get("onnxruntime")
sys.modules["onnxruntime"] = MagicMock()

_MODULES_TO_MOCK = [
    "transformers.models.auto.modeling_auto",
    "transformers.models.auto.auto_factory",
    "transformers.generation",
    "transformers.generation.utils",
    "transformers.generation.candidate_generator",
]

_saved_modules: dict[str, ModuleType | None] = {}

for _mod_name in _MODULES_TO_MOCK:
    _saved_modules[_mod_name] = sys.modules.get(_mod_name)
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()

# Make `from transformers import AutoTokenizer` work. Prefer the real lazy
# package (so its __getattr__ can be patched); if it isn't installed, fall back
# to a fully mocked top-level module.
try:
    import transformers as _tf
except Exception:  # transformers not installed on this host
    _tf = MagicMock()
    sys.modules["transformers"] = _tf

if not getattr(_tf, "_test_patched", False):
    _orig_getattr = getattr(type(_tf), "__getattr__", None)

    def _safe_getattr(self, name):
        """Return a MagicMock for tokenizer/model classes instead of triggering deep imports."""
        if name in (
            "AutoTokenizer",
            "PreTrainedModel",
            "PreTrainedTokenizer",
            "PreTrainedTokenizerFast",
        ):
            return MagicMock()
        if _orig_getattr is not None:
            return _orig_getattr(self, name)
        raise AttributeError(name)

    # MagicMock instances accept attribute assignment but have no settable
    # __getattr__ on the type; guard so the real-package path still patches.
    with contextlib.suppress(TypeError, AttributeError):
        type(_tf).__getattr__ = _safe_getattr
    _tf._test_patched = True

# ---------------------------------------------------------------------------
# Set DIALECT before importing app code
# ---------------------------------------------------------------------------

os.environ.setdefault("DIALECT", "arz")

# ---------------------------------------------------------------------------
# Raw data fixtures (no import of app needed)
# ---------------------------------------------------------------------------


def _load_all_dialect_files() -> dict:
    """Load every app/dialects/<code>.json keyed by dialect code."""
    return {
        p.stem: json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(DIALECTS_DIR.glob("*.json"))
    }


@pytest.fixture(scope="session")
def labels_data() -> dict:
    """Labels keyed by dialect: {code: {sub, main, sub_to_main}}.

    Reconstructed from the per-dialect files' ``labels`` blocks so existing
    tests that expect the flat label shape keep working.
    """
    return {code: cfg["labels"] for code, cfg in _load_all_dialect_files().items()}


@pytest.fixture(scope="session")
def dialects_data() -> dict:
    """Per-dialect config (minus labels), keyed by code under "dialects".

    Reconstructed from the per-dialect files so structure tests keep working
    against one fixture.
    """
    dialects = {
        code: {k: v for k, v in cfg.items() if k != "labels"}
        for code, cfg in _load_all_dialect_files().items()
    }
    return {"dialects": dialects}


# ---------------------------------------------------------------------------
# Sample texts for preprocessing tests
# ---------------------------------------------------------------------------

SAMPLE_TEXTS = {
    "arabic_simple": "مرحبا بالعالم",
    "arabic_long": " ".join(["كلمة"] * 50),
    "arabic_over_50": " ".join(["كلمة"] * 51),
    "arabic_with_emoji": "مرحبا 😊",
    "emoji_only": "😊😂🔥",
    "english_only": "Hello world",
    "mixed_arabic_latin": "مرحبا hello بالعالم",
    "empty": "",
    "whitespace": "   ",
    "url_text": "هذا http://example.com نص",
    "mention_text": "هذا @user نص",
    "hashtag_text": "#تحية مرحبا",
    "photo_text": "[[photo]] مرحبا",
    "leetspeak_text": "3rbi m7shi",
    "repeated_chars": "هههههههه مرحبا",
    "alef_variants": "إبراهيم أحمد آدم ٱلله",
    "alef_maqsura": "على مصطفى",
    "with_diacritics": "بِسْمِ اللَّهِ الرَّحْمَنِ",
    "non_arabic_non_latin": "مرحبا 你好 بالعالم",
    "short_arabic": "ا",
    "pure_numbers": "12345",
}


# ---------------------------------------------------------------------------
# Mock model fixture
# ---------------------------------------------------------------------------


def _make_mock_loaded_model():
    """Create a mock LoadedModel matching the ONNX-runtime field shape.

    The real LoadedModel holds an ORT ``session`` plus the tokenizer, the set of
    input names the graph expects, and ``max_length``. Prediction is mocked at a
    higher level (``_make_mock_predict_single``), so the session never actually
    runs here; this just mirrors the dataclass fields.
    """
    loaded = MagicMock()
    loaded.session = MagicMock()
    loaded.tokenizer = MagicMock()
    loaded.input_names = frozenset({"input_ids", "attention_mask"})
    loaded.max_length = 128
    return loaded


@pytest.fixture
def mock_loaded_model():
    """Provide a mock LoadedModel for tests that need it."""
    return _make_mock_loaded_model()


# ---------------------------------------------------------------------------
# FastAPI TestClient fixture
# ---------------------------------------------------------------------------


def _make_mock_predict_single(dialect: str, labels_data: dict):
    """Build a mock _predict_single that uses real label data."""
    from app.classifier import ClassificationResult

    entry = labels_data[dialect]
    # Labels are keyed by canonical language code (ara/eng/ckb). The API resolves
    # any alias to its canonical code before calling, so lang is always a key here.
    sub_maps = {lg: {int(k): v for k, v in m.items()} for lg, m in entry["sub"].items()}
    main_maps = {lg: {int(k): v for k, v in m.items()} for lg, m in entry["main"].items()}
    sub_to_main = {int(k): v for k, v in entry["sub_to_main"].items()}

    def mock_predict(text, loaded, cfg, lang):
        cleaned = cfg.preprocess_fn(text)
        if not cleaned:
            return ClassificationResult(
                is_valid=False, sub_class=None, main_class=None, confidence=None
            )
        pred_id = 0
        conf = 0.95
        sub_l, main_l = sub_maps[lang], main_maps[lang]
        return ClassificationResult(
            is_valid=True,
            sub_class=sub_l[pred_id],
            main_class=main_l[sub_to_main[pred_id]],
            confidence=round(conf, 4),
        )

    return mock_predict


def _make_mock_predict_batch(mock_single):
    """Build a mock _predict_batch from a mock _predict_single."""

    def mock_batch(texts, loaded, cfg, lang):
        return [mock_single(t, loaded, cfg, lang) for t in texts]

    return mock_batch


@pytest.fixture
def test_client(labels_data):
    """Create a FastAPI TestClient with mocked model inference.

    Uses the DIALECT env var that must already be set.
    """
    dialect = os.environ.get("DIALECT", "arz")

    mock_single = _make_mock_predict_single(dialect, labels_data)
    mock_batch = _make_mock_predict_batch(mock_single)

    import app.classifier as clf

    async def mock_get_classification(text, lang="ara"):
        loaded = _make_mock_loaded_model()
        return mock_single(text, loaded, clf.ACTIVE_CONFIG, lang)

    async def mock_get_classifications_batch(texts, lang="ara"):
        loaded = _make_mock_loaded_model()
        return mock_batch(texts, loaded, clf.ACTIVE_CONFIG, lang)

    with (
        patch("app.classifier.load_model", return_value=_make_mock_loaded_model()),
        patch("app.main.load_model", return_value=_make_mock_loaded_model()),
        patch("app.main.get_classification", side_effect=mock_get_classification),
        patch(
            "app.main.get_classifications_batch",
            side_effect=mock_get_classifications_batch,
        ),
    ):
        from fastapi.testclient import TestClient

        from app.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            yield client

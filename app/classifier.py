"""
Text classification module.

This module provides the interface between the API and the ML model.
Each container serves a single dialect, configured via the DIALECT env var.
"""

import asyncio
import json
import logging
import os
import re
import threading
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import emoji
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# -----------------------------------------------------------------------------
# Dialect config (one self-contained file per dialect in app/dialects/)
# -----------------------------------------------------------------------------
#
# app/dialects/<code>.json is the single source of truth for a dialect: name,
# hf_repo, languages (each with a display name and aliases), preprocessing, and
# labels (sub/main per language + sub_to_main). The dialect code is the filename
# stem. Adding a dialect is a one-file change: the loader globs the directory,
# the Dockerfile reads the same files to decide which models to download, and
# nginx renders its routing from them at startup.

_DIALECTS_DIR = Path(__file__).parent / "dialects"


def _load_json(path: Path) -> dict:
    """Load and parse a JSON file, failing fast with a clear error."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise RuntimeError(f"Config file not found: {path}") from None
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in {path}: {e}") from e


def _load_dialects_config() -> dict:
    """Load every app/dialects/<code>.json into a {code: config} mapping."""
    if not _DIALECTS_DIR.is_dir():
        raise RuntimeError(f"Dialects directory not found at {_DIALECTS_DIR}")
    config: dict[str, dict] = {}
    for path in sorted(_DIALECTS_DIR.glob("*.json")):
        config[path.stem] = _load_json(path)
    if not config:
        raise RuntimeError(f"No dialect files found in {_DIALECTS_DIR}")
    return config


_DIALECTS_CONFIG = _load_dialects_config()

VALID_DIALECTS: frozenset[str] = frozenset(_DIALECTS_CONFIG)
# code -> human-readable dialect name, for logs and API documentation
DIALECT_NAMES: dict[str, str] = {code: cfg["name"] for code, cfg in _DIALECTS_CONFIG.items()}

# -----------------------------------------------------------------------------
# Configuration from environment variables
# -----------------------------------------------------------------------------

DIALECT = os.getenv("DIALECT", "")
if DIALECT not in VALID_DIALECTS:
    raise RuntimeError(
        f"DIALECT env var must be one of {sorted(VALID_DIALECTS)}, got {DIALECT!r}. "
        f"Set it in your environment or compose file."
    )

MODEL_PATH = os.getenv("MODEL_PATH") or f"./models/{DIALECT}"


def _parse_bounded_int(name: str, default: int, lo: int, hi: int) -> int:
    """Parse an integer env var, requiring it to fall within [lo, hi] (raises if not)."""
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from None
    if value < lo or value > hi:
        raise RuntimeError(f"{name} must be between {lo} and {hi}, got {value}")
    return value


CLASSIFIER_WORKERS = _parse_bounded_int("CLASSIFIER_WORKERS", 2, 1, 32)
CACHE_SIZE = _parse_bounded_int("CACHE_SIZE", 1024, 0, 100000)

# Per-request inference timeout (seconds). A safety backstop, not a tuning knob:
# set it well above the worst-case legitimate batch time so it only fires when an
# inference is genuinely stuck. A full MAX_BATCH_SIZE batch is one inference, so
# size this above how long that takes on your hardware. Default 120s.
INFERENCE_TIMEOUT = _parse_bounded_int("INFERENCE_TIMEOUT", 120, 1, 3600)

_VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
_raw_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
if _raw_log_level not in _VALID_LOG_LEVELS:
    logging.getLogger(__name__).warning(
        "Invalid LOG_LEVEL %r, falling back to INFO. Valid: %s",
        _raw_log_level,
        sorted(_VALID_LOG_LEVELS),
    )
    _raw_log_level = "INFO"
LOG_LEVEL = _raw_log_level
LOG_FORMAT = os.getenv("LOG_FORMAT", "text").lower()

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    """Configure logging based on environment variables."""
    level = getattr(logging, LOG_LEVEL, logging.INFO)

    if LOG_FORMAT == "json":
        import json as json_lib

        class JsonFormatter(logging.Formatter):
            def format(self, record):
                return json_lib.dumps(
                    {
                        "timestamp": self.formatTime(record),
                        "level": record.levelname,
                        "logger": record.name,
                        "message": record.getMessage(),
                    }
                )

        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)


_setup_logging()

# -----------------------------------------------------------------------------
# Thread pool for async inference
# -----------------------------------------------------------------------------

_executor: ThreadPoolExecutor | None = None


class ServiceOverloadedError(Exception):
    """Raised when all inference workers are busy."""


class InferenceTimeoutError(Exception):
    """Raised when a single inference exceeds INFERENCE_TIMEOUT seconds."""


class _InferenceGate:
    """Non-blocking concurrency limiter for inference.

    ``try_acquire()`` atomically checks capacity and claims a slot in a single
    synchronous step, then ``release()`` frees it. Unlike a check-then-acquire
    pattern on ``asyncio.Semaphore``, there is no ``await`` between the capacity
    check and the claim, so the count cannot drift under concurrency. Callers
    that find every slot taken get a fast 503 instead of queuing (H-1).

    Safe under asyncio's single-threaded cooperative scheduling: neither method
    awaits, so the read-modify-write is never interleaved with another task.
    """

    __slots__ = ("_in_use", "_limit")

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._in_use = 0

    def try_acquire(self) -> bool:
        """Claim a slot if one is free. Returns False if at capacity."""
        if self._in_use >= self._limit:
            return False
        self._in_use += 1
        return True

    def release(self) -> None:
        """Release a previously-claimed slot."""
        if self._in_use > 0:
            self._in_use -= 1


def _get_executor() -> ThreadPoolExecutor:
    """Get or create the thread pool executor."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=CLASSIFIER_WORKERS)
    return _executor


def shutdown_executor() -> None:
    """Shutdown the thread pool executor gracefully."""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=True)
        _executor = None
        logger.info("Classifier executor shut down")


# -----------------------------------------------------------------------------
# Inference cache
# -----------------------------------------------------------------------------


class InferenceCache:
    """Thread-safe LRU cache for raw model predictions.

    Caches (predicted_id, confidence) keyed by preprocessed text,
    so the same input with different `lang` values is a cache hit.
    """

    def __init__(self, maxsize: int) -> None:
        self._cache: OrderedDict[str, tuple[int, float]] = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> tuple[int, float] | None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, key: str, value: tuple[int, float]) -> None:
        if self._maxsize <= 0:
            return
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._maxsize:
                    self._cache.popitem(last=False)
            self._cache[key] = value

    @property
    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "maxsize": self._maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
            }


_inference_cache = InferenceCache(CACHE_SIZE)


def get_cache_stats() -> dict:
    """Return inference cache statistics (public API for health endpoint)."""
    return _inference_cache.stats


# -----------------------------------------------------------------------------
# Language support
# -----------------------------------------------------------------------------
#
# Each dialect file declares its languages keyed by canonical ISO 639-3 code,
# each with a display name and aliases (e.g. the ISO 639-1 two-letter code). The
# config, labels, and docs use the canonical code; aliases just let the API
# accept a familiar short code. This container serves one dialect, so the maps
# below come from the active dialect's languages.

_ACTIVE_LANGUAGES: dict[str, dict] = _DIALECTS_CONFIG[DIALECT]["languages"]
SUPPORTED_LANGUAGES: frozenset[str] = frozenset(_ACTIVE_LANGUAGES)
# canonical code -> display name
LANGUAGE_NAMES: dict[str, str] = {code: meta["name"] for code, meta in _ACTIVE_LANGUAGES.items()}
# alias code -> canonical code (e.g. "ar" -> "ara")
LANGUAGE_ALIASES: dict[str, str] = {
    alias: code for code, meta in _ACTIVE_LANGUAGES.items() for alias in meta.get("aliases", [])
}


def normalize_lang(lang: str) -> str:
    """Resolve a language code to its canonical form.

    Maps a known alias (e.g. ISO 639-1 'ar') to its canonical ISO 639-3 code
    ('ara'). A canonical or unknown code passes through unchanged, so the caller
    still validates it against the dialect's supported languages.
    """
    return LANGUAGE_ALIASES.get(lang, lang)


# -----------------------------------------------------------------------------
# Label consistency
# -----------------------------------------------------------------------------


def _validate_dialect_labels() -> None:
    """Validate that each dialect file's labels cover its declared languages.

    For every language a dialect declares, its labels block must have both sub
    and main entries. Raises RuntimeError on inconsistency for fast startup
    failure (catches a typo or a missing translation before the first request).
    """
    for dialect_code, dialect_cfg in _DIALECTS_CONFIG.items():
        labels = dialect_cfg.get("labels", {})
        for lang in dialect_cfg["languages"]:
            if lang not in labels.get("sub", {}):
                raise RuntimeError(
                    f"Dialect '{dialect_code}' declares language '{lang}' "
                    f"but its labels have no sub entries for it"
                )
            if lang not in labels.get("main", {}):
                raise RuntimeError(
                    f"Dialect '{dialect_code}' declares language '{lang}' "
                    f"but its labels have no main entries for it"
                )


_validate_dialect_labels()

# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """Result of a text classification."""

    is_valid: bool
    sub_class: str | None
    main_class: str | None
    confidence: float | None


@dataclass
class LoadedModel:
    """Container for loaded model components."""

    model: AutoModelForSequenceClassification
    tokenizer: AutoTokenizer
    device: torch.device
    max_length: int


@dataclass(frozen=True)
class DialectConfig:
    name: str  # human-readable, for logs
    model_path: str  # resolved from env var
    preprocess_fn: Callable[[str], str]  # text -> cleaned text (or "" if invalid)
    sub_to_main: dict[int, int]  # sub_id -> main_id
    sub_labels: dict[str, dict[int, str]]  # lang -> {sub_id: label}
    main_labels: dict[str, dict[int, str]]  # lang -> {main_id: label}


# -----------------------------------------------------------------------------
# Text preprocessing
# -----------------------------------------------------------------------------

# Used by Iraqi and Kurdish preprocessing
ARABIC_SCRIPT_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")

# Iraqi-specific leetspeak substitution map
_LEETSPEAK_MAP = {
    "ch": "تش",
    "gh": "غ",
    "kh": "خ",
    "sh": "ش",
    "th": "ث",
    "dh": "ذ",
    "2": "ء",
    "3": "ع",
    "4": "ذ",
    "5": "خ",
    "6": "ط",
    "7": "ح",
    "8": "ق",
    "9": "ص",
}


def _preprocess_nuha(text: str) -> str:
    """Egyptian Arabic: keep Arabic chars (U+0600-U+06FF), spaces, raw emojis.
    Reject texts over 50 words or consisting entirely of emojis."""
    if len(text.split()) > 50:
        return ""
    filtered = "".join(
        ch for ch in text if "\u0600" <= ch <= "\u06ff" or ch == " " or emoji.is_emoji(ch)
    )
    filtered = " ".join(filtered.split()).strip()
    if filtered and all(emoji.is_emoji(ch) for ch in filtered.replace(" ", "")):
        return ""
    return filtered


def _preprocess_safa(text: str, *, leetspeak: bool = False, alef_maqsura: bool = False) -> str:
    """Shared preprocessing for Iraqi Arabic and Sorani Kurdish.

    Iraqi uses leetspeak decoding and ى→ي; Kurdish does not.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\[\[photo\]\]|photo scraps?", "", text, flags=re.IGNORECASE)
    if leetspeak:
        tokens = []
        for token in text.split():
            if re.search(r"[0-9]", token) and re.search(r"[a-zA-Z]", token):
                decoded = token.lower()
                for k, v in sorted(_LEETSPEAK_MAP.items(), key=lambda x: -len(x[0])):
                    decoded = decoded.replace(k, v)
                tokens.append(decoded)
            else:
                tokens.append(token)
        text = " ".join(tokens)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"[إأآٱ]", "ا", text)
    if alef_maqsura:
        text = re.sub(r"ى", "ي", text)
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = re.sub(r"[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFFa-zA-Z_\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not ARABIC_SCRIPT_RE.search(text) or len(text.strip()) < 2:
        return ""
    return text


# -----------------------------------------------------------------------------
# Active dialect configuration
# -----------------------------------------------------------------------------

_PREPROCESS_REGISTRY: dict[str, Callable[..., str]] = {
    "nuha": _preprocess_nuha,
    "safa": _preprocess_safa,
}


def _build_preprocess_fn(prep_cfg: dict) -> Callable[[str], str]:
    """Build a preprocessing function from dialect config.

    The ``type`` key selects a preprocessor from the registry; every other key
    in the config is passed through as a keyword argument. This keeps the
    dispatch generic: a new preprocessing family needs a registry entry, not
    a branch here.
    """
    prep_type = prep_cfg["type"]
    fn = _PREPROCESS_REGISTRY.get(prep_type)
    if fn is None:
        raise RuntimeError(
            f"Unknown preprocessing type '{prep_type}'. Known types: {sorted(_PREPROCESS_REGISTRY)}"
        )
    kwargs = {k: v for k, v in prep_cfg.items() if k != "type"}
    if not kwargs:
        return fn
    return lambda text: fn(text, **kwargs)


def _parse_dialect(d: str) -> dict:
    """Parse label dicts for a dialect, converting JSON string keys to int.

    Labels are grouped by language so adding a language is a JSON-only change
    (no new dataclass fields). Only languages actually present in the dialect
    file appear, and startup validation guarantees they cover its languages.
    """
    entry = _DIALECTS_CONFIG[d]["labels"]
    return {
        "sub_to_main": {int(k): v for k, v in entry["sub_to_main"].items()},
        "sub_labels": {
            lang: {int(k): v for k, v in mapping.items()} for lang, mapping in entry["sub"].items()
        },
        "main_labels": {
            lang: {int(k): v for k, v in mapping.items()} for lang, mapping in entry["main"].items()
        },
    }


_active_dialect_cfg = _DIALECTS_CONFIG[DIALECT]

ACTIVE_CONFIG = DialectConfig(
    name=_active_dialect_cfg["name"],
    model_path=MODEL_PATH,
    preprocess_fn=_build_preprocess_fn(_active_dialect_cfg["preprocessing"]),
    **_parse_dialect(DIALECT),
)


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_model() -> LoadedModel:
    """Load the model for the active dialect, cached."""
    path = Path(ACTIVE_CONFIG.model_path)
    if not path.exists():
        raise RuntimeError(
            f"Model not found at {path}. "
            f"Set MODEL_PATH environment variable or ensure the model is present."
        )
    logger.info("Loading model from %s", path)

    max_length = 128
    config_path = path / "training_config.json"
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                max_length = json.load(f).get("max_length", 128)
        except json.JSONDecodeError:
            logger.warning("Could not parse %s; using max_length=128", config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
    else:
        logger.info("Using CPU for inference")

    try:
        tokenizer = AutoTokenizer.from_pretrained(ACTIVE_CONFIG.model_path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(ACTIVE_CONFIG.model_path, use_fast=False)

    model = AutoModelForSequenceClassification.from_pretrained(ACTIVE_CONFIG.model_path)
    model.to(device)
    model.eval()

    logger.info("Model loaded | max_length=%s | device=%s", max_length, device)
    return LoadedModel(model=model, tokenizer=tokenizer, device=device, max_length=max_length)


# -----------------------------------------------------------------------------
# Classification functions
# -----------------------------------------------------------------------------


def _predict_single(
    text: str, loaded: LoadedModel, cfg: DialectConfig, lang: str
) -> ClassificationResult:
    """Synchronous single prediction."""
    cleaned = cfg.preprocess_fn(text)

    if not cleaned:
        return ClassificationResult(
            is_valid=False, sub_class=None, main_class=None, confidence=None
        )

    # Check cache for raw prediction (predicted_id, confidence)
    cached = _inference_cache.get(cleaned)
    if cached is not None:
        predicted_id, confidence_val = cached
    else:
        inputs = loaded.tokenizer(
            cleaned,
            truncation=True,
            max_length=loaded.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        inputs = {k: v.to(loaded.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = loaded.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            confidence, pred = torch.max(probs, dim=-1)

        predicted_id = pred.item()
        confidence_val = confidence.item()
        _inference_cache.put(cleaned, (predicted_id, confidence_val))

    # Label lookup (always runs; it depends on lang, which is not cached).
    # lang is validated against SUPPORTED_LANGUAGES upstream, so it's present.
    sub_labels = cfg.sub_labels[lang]
    main_labels = cfg.main_labels[lang]
    try:
        sub_class = sub_labels[predicted_id]
        main_class = main_labels[cfg.sub_to_main[predicted_id]]
    except KeyError:
        logger.error("Model predicted unknown class ID %d for dialect '%s'", predicted_id, cfg.name)
        return ClassificationResult(
            is_valid=False, sub_class=None, main_class=None, confidence=None
        )

    return ClassificationResult(
        is_valid=True,
        sub_class=sub_class,
        main_class=main_class,
        confidence=round(confidence_val, 4),
    )


def _predict_batch(
    texts: list[str], loaded: LoadedModel, cfg: DialectConfig, lang: str
) -> list[ClassificationResult]:
    """
    Synchronous batch prediction.

    Uses the inference cache to skip model inference for previously-seen texts.
    Only cache misses are batched together for model prediction.
    """
    cleaned = [cfg.preprocess_fn(t) for t in texts]

    valid_indices = [i for i, c in enumerate(cleaned) if c]

    results: list[ClassificationResult] = [
        ClassificationResult(is_valid=False, sub_class=None, main_class=None, confidence=None)
        for _ in texts
    ]

    if not valid_indices:
        return results

    # Split valid texts into cache hits and misses
    # predictions[i] will hold (predicted_id, confidence) for each valid index
    predictions: dict[int, tuple[int, float]] = {}
    miss_indices: list[int] = []  # indices into valid_indices
    miss_texts: list[str] = []

    for vi, idx in enumerate(valid_indices):
        cached = _inference_cache.get(cleaned[idx])
        if cached is not None:
            predictions[idx] = cached
        else:
            miss_indices.append(vi)
            miss_texts.append(cleaned[idx])

    # Run inference only on cache misses
    if miss_texts:
        inputs = loaded.tokenizer(
            miss_texts,
            truncation=True,
            max_length=loaded.max_length,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(loaded.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = loaded.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            confidences, predicted_ids = torch.max(probs, dim=-1)

        for j, vi in enumerate(miss_indices):
            idx = valid_indices[vi]
            pred_id = predicted_ids[j].item()
            conf = confidences[j].item()
            predictions[idx] = (pred_id, conf)
            _inference_cache.put(cleaned[idx], (pred_id, conf))

    # Label lookup for all valid texts.
    # lang is validated against SUPPORTED_LANGUAGES upstream, so it's present.
    sub_labels = cfg.sub_labels[lang]
    main_labels = cfg.main_labels[lang]

    for idx in valid_indices:
        pred_id, conf = predictions[idx]
        try:
            results[idx] = ClassificationResult(
                is_valid=True,
                sub_class=sub_labels[pred_id],
                main_class=main_labels[cfg.sub_to_main[pred_id]],
                confidence=round(conf, 4),
            )
        except KeyError:
            logger.error("Model predicted unknown class ID %d for dialect '%s'", pred_id, cfg.name)

    return results


# -----------------------------------------------------------------------------
# Async API
# -----------------------------------------------------------------------------


_inference_gate = _InferenceGate(CLASSIFIER_WORKERS)


async def _run_gated(fn: Callable, *args):
    """Run a synchronous inference ``fn`` in the thread pool under the concurrency
    gate and the per-request inference timeout.

    Two things make the slot accounting correct under timeout:

    1. The slot is released by a done-callback when the worker thread *actually*
       finishes, not when we stop awaiting it. A torch op can't be cancelled
       mid-flight, so on timeout the thread keeps running to completion; releasing
       only then keeps the gate's in-flight count honest (a slow inference can't
       make it under-count and over-admit work).
    2. We ``shield`` the future so ``wait_for`` cancels only our wait, never the
       running inference, and the callback retrieves any exception so asyncio
       doesn't warn about it going unobserved on the timeout path.

    On timeout the caller gets ``InferenceTimeoutError`` (surfaced as 504).
    """
    if not _inference_gate.try_acquire():
        raise ServiceOverloadedError("All inference workers are busy")
    try:
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(_get_executor(), fn, *args)
    except BaseException:
        # Acquired but never scheduled the work, so don't leak the slot.
        _inference_gate.release()
        raise

    def _release_slot(fut: asyncio.Future) -> None:
        _inference_gate.release()
        if not fut.cancelled():
            fut.exception()  # observe result/exception so asyncio stays quiet

    future.add_done_callback(_release_slot)

    try:
        return await asyncio.wait_for(asyncio.shield(future), timeout=INFERENCE_TIMEOUT)
    except TimeoutError:
        raise InferenceTimeoutError(
            f"Inference did not complete within {INFERENCE_TIMEOUT}s"
        ) from None


async def get_classification(text: str, lang: str = "ara") -> ClassificationResult:
    """Async single classification."""
    loaded = load_model()
    return await _run_gated(_predict_single, text, loaded, ACTIVE_CONFIG, lang)


async def get_classifications_batch(
    texts: list[str], lang: str = "ara"
) -> list[ClassificationResult]:
    """Async batch classification. Uses true batching, not sequential calls."""
    loaded = load_model()
    return await _run_gated(_predict_batch, texts, loaded, ACTIVE_CONFIG, lang)

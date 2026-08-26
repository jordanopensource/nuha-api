"""
Text classification engine.

Everything about running ONE dialect's model: preprocessing, the ONNX session,
the inference cache, the admission gate, the tokenizer lock, and label lookup.
A dialect arrives as a directory (dialect.json + an ONNX snapshot) and
``load_dialect_dir`` turns it into a self-contained ``LoadedDialect`` bundle;
app/registry.py discovers those directories on the models volume at startup and
app/main.py routes requests to the bundles. The gate and the executor are
process-global on purpose: CLASSIFIER_WORKERS is the process's inference
capacity, shared by every loaded dialect.
"""

import asyncio
import json
import logging
import re
import threading
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import emoji
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from app.common.config import parse_bounded_int
from app.common.dialect_schema import validate_dialect_config
from app.common.logging import setup_logging


# -----------------------------------------------------------------------------
# Configuration from environment variables
# -----------------------------------------------------------------------------

# Process-wide inference capacity, shared by every loaded dialect: the thread
# pool holds CLASSIFIER_WORKERS threads and the gate admits at most that many
# concurrent session.run() calls (plus the queue).
CLASSIFIER_WORKERS = parse_bounded_int("CLASSIFIER_WORKERS", 2, 1, 32)
# Per-dialect cache capacity: every loaded dialect gets its own InferenceCache
# of this size (keys are preprocessed text, which two dialects could share, so
# the caches must not be pooled). Memory scales with the loaded dialect count.
CACHE_SIZE = parse_bounded_int("CACHE_SIZE", 1024, 0, 100000)

# Per-request inference timeout (seconds). A safety backstop, not a tuning knob:
# set it well above the worst-case legitimate batch time so it only fires when an
# inference is genuinely stuck. A full MAX_BATCH_SIZE batch is one inference, so
# size this above how long that takes on your hardware. Default 120s.
INFERENCE_TIMEOUT = parse_bounded_int("INFERENCE_TIMEOUT", 120, 1, 3600)

# ONNX Runtime threads PER inference. Default 1 so each of the CLASSIFIER_WORKERS
# concurrent session.run() calls uses ~1 core, keeping the "one inference per CPU,
# CLASSIFIER_WORKERS ~= API_CPU_LIMIT" tuning model. This replaces torch's
# OMP/MKL thread vars and avoids the CFS-throttle trap they had (ORT would
# otherwise default intra_op threads to the HOST core count, ignoring the Docker
# cpu cap). To favour fewer, faster (multi-threaded) inferences over concurrency,
# raise this and lower CLASSIFIER_WORKERS to keep their product near the cpu cap.
ORT_INTRA_OP_THREADS = parse_bounded_int("ORT_INTRA_OP_THREADS", 1, 1, 32)

# Admission queue in front of the CLASSIFIER_WORKERS slots: a request that finds
# every slot busy waits (up to INFERENCE_QUEUE_TIMEOUT) instead of being shed
# immediately, so a short burst is served rather than 503'd. At most
# CLASSIFIER_WORKERS + INFERENCE_QUEUE_SIZE are in flight; beyond that it sheds a
# fast 503. It smooths bursts, does not add throughput. 0 = shed immediately.
INFERENCE_QUEUE_SIZE = parse_bounded_int("INFERENCE_QUEUE_SIZE", 32, 0, 10000)
# Max seconds a request waits for a slot before shedding 503. Part of the request
# latency budget: the compose stop_grace_period must stay above
# INFERENCE_QUEUE_TIMEOUT + INFERENCE_TIMEOUT so an admitted request can always
# finish (with its own 503/504) before the container is killed on shutdown.
INFERENCE_QUEUE_TIMEOUT = parse_bounded_int("INFERENCE_QUEUE_TIMEOUT", 30, 1, 600)

# Logging is configured once here (this module is imported exactly once per
# process, before any request); setup_logging() is idempotent.
setup_logging()
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Thread pool for async inference
# -----------------------------------------------------------------------------

_executor: ThreadPoolExecutor | None = None


class ServiceOverloadedError(Exception):
    """Raised when all inference workers are busy."""


class InferenceTimeoutError(Exception):
    """Raised when a single inference exceeds INFERENCE_TIMEOUT seconds."""


class _InferenceGate:
    """Bounded admission controller for inference.

    Two layers:

    * **Admission**: ``try_admit()`` atomically checks the in-flight count
      (running + waiting) against ``limit + queue_size`` and claims a place in a
      single synchronous step. There is no ``await`` between the check and the
      bump, so the count cannot drift under asyncio's cooperative scheduling.
      Over the cap, callers get a fast 503 instead of piling up without bound.
    * **Execution slots**: at most ``limit`` inferences run at once (one per
      CPU under the tuning model), governed by an ``asyncio.Semaphore``. An
      admitted request waits on ``acquire_slot()`` up to ``wait_timeout`` seconds
      for a slot; this is the queue that lets a short burst be served instead of
      shed the instant every worker is busy.

    With ``queue_size == 0`` admission only ever succeeds when a slot is already
    free, so ``acquire_slot()`` never waits: the original no-queue,
    shed-immediately behavior. The slot is released when the worker thread
    *actually* finishes (see ``_run_gated``), keeping the count honest under the
    inference-timeout path.
    """

    __slots__ = ("_in_flight", "_limit", "_max_in_flight", "_slots", "_wait_timeout")

    def __init__(self, limit: int, queue_size: int, wait_timeout: int) -> None:
        self._limit = limit
        self._max_in_flight = limit + queue_size
        self._wait_timeout = wait_timeout
        self._in_flight = 0
        # Constructed at import (no running loop). asyncio.Semaphore binds to the
        # loop lazily on first await (Python >= 3.10), so this is safe.
        self._slots = asyncio.Semaphore(limit)

    def try_admit(self) -> bool:
        """Claim an in-flight place if under the running+queued cap. Sync, no await."""
        if self._in_flight >= self._max_in_flight:
            return False
        self._in_flight += 1
        return True

    def drop_admission(self) -> None:
        """Release an admission that never acquired a slot (e.g. wait timed out)."""
        if self._in_flight > 0:
            self._in_flight -= 1

    async def acquire_slot(self) -> None:
        """Wait (bounded) for an execution slot. Raises TimeoutError past the deadline.

        Relies on Python >= 3.11 ``wait_for`` cancelling a not-yet-granted
        ``Semaphore.acquire()`` without consuming a permit, so a timed-out wait
        never leaks a slot (the runtime image is python:3.12).
        """
        await asyncio.wait_for(self._slots.acquire(), self._wait_timeout)

    def release_slot(self) -> None:
        """Free a slot held by a finished inference and drop its admission."""
        self._slots.release()
        if self._in_flight > 0:
            self._in_flight -= 1


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
    Instantiated once per loaded dialect: the key is only the text, so a
    process-wide cache would let two dialects poison each other's entries.
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
    """Container for loaded model components.

    Holds a single shared ONNX Runtime ``InferenceSession``. ORT sessions are
    thread-safe for concurrent ``run()`` calls, so one session backs all
    CLASSIFIER_WORKERS pool threads (the GIL is released during ``run()``).
    """

    session: ort.InferenceSession
    tokenizer: AutoTokenizer
    input_names: frozenset[str]  # input names the ONNX graph actually expects
    max_length: int


@dataclass(frozen=True)
class DialectConfig:
    name: str  # human-readable, for logs
    preprocess_fn: Callable[[str], str]  # text -> cleaned text (or "" if invalid)
    sub_to_main: dict[int, int]  # sub_id -> main_id
    sub_labels: dict[str, dict[int, str]]  # lang -> {sub_id: label}
    main_labels: dict[str, dict[int, str]]  # lang -> {main_id: label}


@dataclass(frozen=True)
class LoadedDialect:
    """One dialect, fully loaded and ready to serve.

    The whole per-dialect world lives here so the registry can swap dialects in
    and out as opaque units: config and labels, the loaded model, the language
    maps, an inference cache of its own (keys are preprocessed text, which two
    dialects could share), and a tokenizer lock of its own (the fast-tokenizer
    borrow race is per Rust object, so per-dialect locks avoid cross-dialect
    contention). A request handler snapshots one of these by reference; the
    bundle stays fully usable even if the registry has since dropped it.
    """

    code: str
    path: Path
    config: DialectConfig
    loaded: LoadedModel
    supported_languages: frozenset[str]
    default_language: str  # first declared canonical code, alphabetically
    aliases: dict[str, str]  # alias code -> canonical code (e.g. "ar" -> "ara")
    cache: InferenceCache
    tokenizer_lock: threading.Lock


def build_language_maps(languages_cfg: dict) -> tuple[frozenset[str], str, dict[str, str]]:
    """Build (supported, default, aliases) from a dialect's ``languages`` block.

    Each dialect declares its languages keyed by canonical ISO 639-3 code, each
    with a display name and aliases (e.g. the ISO 639-1 two-letter code). The
    default response language is the first canonical code alphabetically:
    derived, not hardcoded, so a dialect that doesn't serve Arabic still has a
    working default. Schema validation guarantees every declared language has
    labels, so the default is always serviceable. (For the shipped dialects
    this resolves to "ara".)
    """
    supported = frozenset(languages_cfg)
    default = sorted(supported)[0]
    aliases = {
        alias: code for code, meta in languages_cfg.items() for alias in meta.get("aliases", [])
    }
    return supported, default, aliases


# -----------------------------------------------------------------------------
# Text preprocessing
# -----------------------------------------------------------------------------

# Used by Iraqi and Kurdish preprocessing
ARABIC_SCRIPT_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")

# Max words per input, shared by both preprocessing families. An input over this
# is rejected up front (returns "") to bound per-request CPU, so a large
# many-token body cannot tie up an inference slot.
_MAX_WORDS = 50

# Upper bound for a model's declared tokenizer max_length. The shipped models
# use 128; anything a snapshot declares above this is treated as bad data (see
# _load_model), keeping a hostile training_config.json from inflating
# per-request tokenizer allocations.
_MAX_TOKENIZER_LEN = 4096

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
    Reject texts over the word cap or consisting entirely of emojis."""
    if len(text.split()) > _MAX_WORDS:
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
    # Reject overly long inputs up front, before any regex/leetspeak/demojize
    # work, mirroring _preprocess_nuha's word-cap guard. Bounds per-request CPU:
    # without this, a single request packed to the body-size cap with many
    # short leetspeak tokens could hold an inference slot for tens of seconds
    # (the leetspeak loop is per-token), and INFERENCE_TIMEOUT would not free it
    # (the slot releases only when the worker actually finishes).
    if len(text.split()) > _MAX_WORDS:
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


def _parse_dialect(cfg: dict) -> dict:
    """Parse a dialect config's label dicts, converting JSON string keys to int.

    Labels are grouped by language so adding a language is a JSON-only change
    (no new dataclass fields). Only languages actually present in the dialect
    file appear, and schema validation guarantees they cover its languages.
    """
    entry = cfg["labels"]
    return {
        "sub_to_main": {int(k): v for k, v in entry["sub_to_main"].items()},
        "sub_labels": {
            lang: {int(k): v for k, v in mapping.items()} for lang, mapping in entry["sub"].items()
        },
        "main_labels": {
            lang: {int(k): v for k, v in mapping.items()} for lang, mapping in entry["main"].items()
        },
    }


# -----------------------------------------------------------------------------
# Loading a dialect directory
# -----------------------------------------------------------------------------


def _load_json(path: Path) -> dict:
    """Load and parse a JSON file, failing fast with a clear error."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise RuntimeError(f"Config file not found: {path}") from None
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in {path}: {e}") from e


def _find_onnx_file(path: Path) -> Path:
    """Locate the ONNX graph inside a model directory.

    Prefers the conventional ``model.onnx`` produced by Optimum/transformers
    export; otherwise falls back to the single ``*.onnx`` file present. Raises if
    none or more than one ambiguous candidate is found.
    """
    preferred = path / "model.onnx"
    if preferred.is_file():
        return preferred
    candidates = sorted(path.glob("*.onnx"))
    if not candidates:
        raise RuntimeError(
            f"No ONNX model (.onnx) found in {path}. "
            f"Ensure the dialect was installed by the fetch command."
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple .onnx files in {path}: {[c.name for c in candidates]}. "
            f"Expected a single 'model.onnx'."
        )
    return candidates[0]


def parse_dialect_dir(code: str, path: Path) -> tuple[dict, DialectConfig, Path]:
    """Parse and structurally validate one model directory, without touching ML.

    This is the completeness check the registry relies on: dialect.json must
    parse, pass the shared schema validation, and the directory must hold
    exactly one resolvable ONNX graph. Returns the raw config (for the language
    maps), the built ``DialectConfig``, and the ONNX path. Raises RuntimeError
    with every problem listed when the directory is not loadable.
    """
    cfg = _load_json(path / "dialect.json")
    problems = validate_dialect_config(code, cfg)
    if problems:
        raise RuntimeError(f"Invalid dialect config in {path}: " + "; ".join(problems))
    config = DialectConfig(
        name=cfg["name"],
        preprocess_fn=_build_preprocess_fn(cfg["preprocessing"]),
        **_parse_dialect(cfg),
    )
    onnx_path = _find_onnx_file(path)
    return cfg, config, onnx_path


def _load_model(path: Path, onnx_path: Path) -> LoadedModel:
    """Load the ONNX Runtime session + tokenizer for one model directory.

    A single ``InferenceSession`` is shared across all inference threads (ORT's
    ``run()`` is thread-safe). Intra-op threads are pinned to ORT_INTRA_OP_THREADS
    (default 1) and inter-op is fixed at 1, so each concurrent inference uses ~1
    core, matching the CLASSIFIER_WORKERS-per-CPU tuning model.
    """
    logger.info("Loading model from %s", path)

    max_length = 128
    config_path = path / "training_config.json"
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                raw_len = json.load(f).get("max_length", 128)
        except json.JSONDecodeError:
            logger.warning("Could not parse %s; using max_length=128", config_path)
        else:
            # training_config.json arrives with the downloaded snapshot, so its
            # values are model-repo data, not operator config: clamp before it
            # can size tokenizer buffers (a huge value would amplify per-request
            # memory across every text in a batch).
            if (
                isinstance(raw_len, int)
                and not isinstance(raw_len, bool)
                and 1 <= raw_len <= _MAX_TOKENIZER_LEN
            ):
                max_length = raw_len
            else:
                logger.warning(
                    "Ignoring max_length=%r in %s (must be an integer in [1, %d]); using 128",
                    raw_len,
                    config_path,
                    _MAX_TOKENIZER_LEN,
                )

    tokenizer = AutoTokenizer.from_pretrained(str(path))

    # Pin ORT threading so each concurrent session.run() stays ~single-core. ORT
    # otherwise sizes intra_op threads to the host core count, ignoring the Docker
    # cpu cap and re-introducing the CFS-throttle slowdown the torch build had.
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = ORT_INTRA_OP_THREADS
    sess_options.inter_op_num_threads = 1
    # Apply every graph optimization (constant folding, node fusions, CPU layout
    # opts). This is ORT's own default, but pinning it makes the intent explicit
    # and keeps it stable if a future ORT release changes that default.
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    input_names = frozenset(i.name for i in session.get_inputs())

    logger.info(
        "Model loaded | max_length=%s | provider=CPUExecutionProvider | inputs=%s "
        "| intra_op_threads=%s",
        max_length,
        sorted(input_names),
        ORT_INTRA_OP_THREADS,
    )
    return LoadedModel(
        session=session,
        tokenizer=tokenizer,
        input_names=frozenset(input_names),
        max_length=max_length,
    )


def _output_num_labels(loaded: LoadedModel) -> int | None:
    """The classifier output width the ONNX graph declares, if statically known.

    Returns None when the graph leaves the logits' last dimension symbolic (some
    exports do), in which case the load check below can't compare and skips.
    """
    outputs = loaded.session.get_outputs()
    if not outputs:
        return None
    shape = outputs[0].shape or []
    last = shape[-1] if shape else None
    return last if isinstance(last, int) else None


def check_label_consistency(loaded: LoadedModel, expected_num_labels: int, code: str) -> None:
    """Fail a dialect's load if the model's output width != its declared labels.

    Catches a mis-packaged model directory (a model built for one taxonomy
    installed with a dialect file for another) BEFORE it can serve a single
    silently-wrong label. Skips only when the graph's output width is not
    statically declared.
    """
    actual = _output_num_labels(loaded)
    if actual is not None and actual != expected_num_labels:
        raise RuntimeError(
            f"Model/label mismatch for dialect '{code}': the ONNX graph outputs "
            f"{actual} classes but the dialect file declares {expected_num_labels} "
            f"(len(sub_to_main)). This model directory is mis-packaged; refusing to load it."
        )


def _self_test(loaded: LoadedModel) -> None:
    """Run one tiny inference through a freshly loaded session.

    A corrupt-but-parseable ONNX file passes session construction and only
    explodes on the first ``run()``; doing that run here (milliseconds) keeps a
    broken artifact from ever being registered to serve traffic. The load path
    is single-threaded, so no tokenizer lock is needed.
    """
    tokenized = loaded.tokenizer(
        "test",
        truncation=True,
        max_length=loaded.max_length,
        padding=False,
        return_tensors="np",
        return_token_type_ids="token_type_ids" in loaded.input_names,
    )
    loaded.session.run(None, _build_onnx_inputs(loaded, tokenized))


def load_dialect_dir(code: str, path: Path) -> LoadedDialect:
    """Load one model directory into a ready-to-serve ``LoadedDialect``.

    Parse and validate the directory, load the model, check the label/output
    width, self-test the session, and assemble the per-dialect bundle. Any
    failure raises; the registry treats that as "this dialect stays out" and
    the other dialects are unaffected.
    """
    cfg, config, onnx_path = parse_dialect_dir(code, path)
    loaded = _load_model(path, onnx_path)
    check_label_consistency(loaded, len(config.sub_to_main), code)
    _self_test(loaded)
    supported, default, aliases = build_language_maps(cfg["languages"])
    return LoadedDialect(
        code=code,
        path=path,
        config=config,
        loaded=loaded,
        supported_languages=supported,
        default_language=default,
        aliases=aliases,
        cache=InferenceCache(CACHE_SIZE),
        tokenizer_lock=threading.Lock(),
    )


# -----------------------------------------------------------------------------
# Inference internals
# -----------------------------------------------------------------------------


def _softmax_argmax(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Numerically-stable row-wise softmax, returning (max_prob, argmax) per row.

    ``logits`` is the ORT classifier output of shape (batch, num_labels). Returns
    the top probability and its class id for each row, as numpy arrays.
    """
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp, axis=-1, keepdims=True)
    predicted_ids = np.argmax(probs, axis=-1)
    confidences = np.max(probs, axis=-1)
    return confidences, predicted_ids


def _build_onnx_inputs(loaded: LoadedModel, tokenized: dict) -> dict[str, np.ndarray]:
    """Build the feed the ONNX graph expects from tokenizer output, as int64 arrays.

    We feed exactly the names the session declares: a BERT graph wants
    ``token_type_ids``, an XLM-R graph does not, and feeding a name the graph
    doesn't declare would break ``run()``.

    BERT graphs *require* ``token_type_ids``, but not every tokenizer emits it:
    transformers 5.x's fast ``TokenizersBackend`` omits it by default (the slow
    ``BertTokenizer`` still includes it). We pass ``return_token_type_ids=True``
    at tokenization to coax it out, and as a backstop synthesize it here as
    all-zeros (the correct single-sequence segment id) shaped like ``input_ids``
    if the graph needs it but the tokenizer still didn't produce it. Without this
    a BERT dialect on a token_type_ids-less tokenizer 500s on every request.
    """
    feed: dict[str, np.ndarray] = {}
    for name in ("input_ids", "attention_mask"):
        if name in loaded.input_names and name in tokenized:
            feed[name] = np.asarray(tokenized[name], dtype=np.int64)
    if "token_type_ids" in loaded.input_names:
        if "token_type_ids" in tokenized:
            feed["token_type_ids"] = np.asarray(tokenized["token_type_ids"], dtype=np.int64)
        elif "input_ids" in feed:
            # Single-sequence inputs are all segment 0; the tokenizer just didn't
            # emit the column. Fill it so the required graph input is present.
            feed["token_type_ids"] = np.zeros_like(feed["input_ids"])
    return feed


# -----------------------------------------------------------------------------
# Classification functions
# -----------------------------------------------------------------------------

# Tokenizer calls are serialized per dialect (LoadedDialect.tokenizer_lock). The
# HF fast tokenizer wraps ONE Rust object and applies per-call truncation and
# padding by MUTATING its state before encoding; the single path uses
# padding=False and the batch path padding=True, so two concurrent calls on the
# same tokenizer can hit the classic "RuntimeError: Already borrowed" race
# (huggingface/tokenizers#537), causing sporadic 500s under mixed single+batch
# load. Tokenization is microseconds against an inference of tens of
# milliseconds, so serializing it costs nothing observable; ONLY the tokenizer
# call is under the lock; session.run() stays fully parallel.


def _predict_single(text: str, d: LoadedDialect, lang: str) -> ClassificationResult:
    """Synchronous single prediction."""
    cfg = d.config
    loaded = d.loaded
    cleaned = cfg.preprocess_fn(text)

    if not cleaned:
        return ClassificationResult(
            is_valid=False, sub_class=None, main_class=None, confidence=None
        )

    # Check cache for raw prediction (predicted_id, confidence)
    cached = d.cache.get(cleaned)
    if cached is not None:
        predicted_id, confidence_val = cached
    else:
        with d.tokenizer_lock:
            tokenized = loaded.tokenizer(
                cleaned,
                truncation=True,
                max_length=loaded.max_length,
                # No padding for a single sequence: pad to nothing, so a short text
                # costs only its real token count instead of a fixed max_length (128)
                # forward pass. The exported graphs have a dynamic sequence axis (the
                # batch path relies on the same), so variable length is fine and this
                # is the dominant single-classify latency win for short text.
                padding=False,
                return_tensors="np",
                # BERT graphs require token_type_ids; ask for it explicitly so the
                # transformers 5.x fast tokenizer (which omits it by default) emits it.
                return_token_type_ids="token_type_ids" in loaded.input_names,
            )
        feed = _build_onnx_inputs(loaded, tokenized)
        logits = loaded.session.run(None, feed)[0]
        confidences, predicted_ids = _softmax_argmax(logits)

        predicted_id = int(predicted_ids[0])
        confidence_val = float(confidences[0])
        d.cache.put(cleaned, (predicted_id, confidence_val))

    # Label lookup (always runs; it depends on lang, which is not cached).
    # lang is validated against the dialect's supported languages by the caller.
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


def _predict_batch(texts: list[str], d: LoadedDialect, lang: str) -> list[ClassificationResult]:
    """
    Synchronous batch prediction.

    Uses the dialect's inference cache to skip model inference for
    previously-seen texts. Only cache misses are batched together for model
    prediction.
    """
    cfg = d.config
    loaded = d.loaded
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
        cached = d.cache.get(cleaned[idx])
        if cached is not None:
            predictions[idx] = cached
        else:
            miss_indices.append(vi)
            miss_texts.append(cleaned[idx])

    # Run inference only on cache misses
    if miss_texts:
        with d.tokenizer_lock:
            tokenized = loaded.tokenizer(
                miss_texts,
                truncation=True,
                max_length=loaded.max_length,
                padding=True,
                return_tensors="np",
                # BERT graphs require token_type_ids; ask for it explicitly so the
                # transformers 5.x fast tokenizer (which omits it by default) emits it.
                return_token_type_ids="token_type_ids" in loaded.input_names,
            )
        feed = _build_onnx_inputs(loaded, tokenized)
        logits = loaded.session.run(None, feed)[0]
        confidences, predicted_ids = _softmax_argmax(logits)

        for j, vi in enumerate(miss_indices):
            idx = valid_indices[vi]
            pred_id = int(predicted_ids[j])
            conf = float(confidences[j])
            predictions[idx] = (pred_id, conf)
            d.cache.put(cleaned[idx], (pred_id, conf))

    # Label lookup for all valid texts.
    # lang is validated against the dialect's supported languages by the caller.
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


_inference_gate = _InferenceGate(CLASSIFIER_WORKERS, INFERENCE_QUEUE_SIZE, INFERENCE_QUEUE_TIMEOUT)


async def _run_gated(fn: Callable, *args):
    """Run a synchronous inference ``fn`` in the thread pool under the admission
    gate, the bounded slot wait, and the per-request inference timeout.

    Flow: admit (fast 503 if the running+queued cap is full) → wait up to the
    queue timeout for an execution slot (fast 503 if the deadline passes) → run
    the inference off the event loop with a hard INFERENCE_TIMEOUT backstop (504).

    Two things make the slot accounting correct under timeout:

    1. The slot is released by a done-callback when the worker thread *actually*
       finishes, not when we stop awaiting it. A native ONNX Runtime ``run()``
       can't be cancelled mid-flight, so on timeout the thread keeps running to
       completion; releasing only then keeps the gate's in-flight count honest (a
       slow inference can't make it under-count and over-admit work).
    2. We ``shield`` the future so ``wait_for`` cancels only our wait, never the
       running inference, and the callback retrieves any exception so asyncio
       doesn't warn about it going unobserved on the timeout path.

    On overload the caller gets ``ServiceOverloadedError`` (503); on a stuck
    inference, ``InferenceTimeoutError`` (504).
    """
    if not _inference_gate.try_admit():
        raise ServiceOverloadedError("Inference queue is full")

    # Hold an admission place. Until a slot is acquired and the done-callback is
    # attached, this coroutine owns the admission and must drop it on every exit.
    try:
        await _inference_gate.acquire_slot()
    except TimeoutError:
        _inference_gate.drop_admission()
        raise ServiceOverloadedError("Timed out waiting for an inference slot") from None
    except BaseException:
        # e.g. the request was cancelled (client disconnect) while queued.
        _inference_gate.drop_admission()
        raise

    # Slot acquired. From the moment the callback is attached it owns releasing
    # both the slot and the admission when the worker thread finishes.
    try:
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(_get_executor(), fn, *args)
    except BaseException:
        _inference_gate.release_slot()
        raise

    def _release_slot(fut: asyncio.Future) -> None:
        _inference_gate.release_slot()
        if not fut.cancelled():
            fut.exception()  # observe result/exception so asyncio stays quiet

    future.add_done_callback(_release_slot)

    try:
        return await asyncio.wait_for(asyncio.shield(future), timeout=INFERENCE_TIMEOUT)
    except TimeoutError:
        raise InferenceTimeoutError(
            f"Inference did not complete within {INFERENCE_TIMEOUT}s"
        ) from None


async def get_classification(d: LoadedDialect, text: str, lang: str) -> ClassificationResult:
    """Async single classification against one loaded dialect."""
    return await _run_gated(_predict_single, text, d, lang)


async def get_classifications_batch(
    d: LoadedDialect, texts: list[str], lang: str
) -> list[ClassificationResult]:
    """Async batch classification. Uses true batching, not sequential calls."""
    return await _run_gated(_predict_batch, texts, d, lang)

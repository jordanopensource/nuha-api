"""Root fixtures for the Nuha test suite.

One service, one suite, one pytest run: the ML mocks install here at import,
before anything touches ``app`` (the suite runs with no onnxruntime wheel and
no model files). The suite discovers dialects from ``app/dialects/*.json`` the
same way the fetch command does. No dialect code or dialect-specific value is
hardcoded anywhere, and the app-level fixtures build a throwaway models VOLUME
from those same files, so the tests exercise the real startup scan against real
configs with only the model load mocked out.
"""

import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests._ml_mock import install_ml_mocks


# Mock the ML stack BEFORE any app import below.
install_ml_mocks()

from app.common.dialect_schema import validate_dialect_config  # noqa: E402,F401


# The one place the dialect-file schema is defined; tests assert through it
# (re-exported so test modules import it from one seam).


# ---------------------------------------------------------------------------
# Paths and discovery
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "app"
DIALECTS_DIR = APP_DIR / "dialects"


def _dialect_file(code: str) -> Path:
    """Path to a single dialect's config file."""
    return DIALECTS_DIR / f"{code}.json"


def _load_all_dialect_files() -> dict:
    """Load every app/dialects/<code>.json keyed by dialect code."""
    return {
        p.stem: json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(DIALECTS_DIR.glob("*.json"))
    }


# A new app/dialects/<code>.json is picked up by every parametrized test with no
# test edit; the tests validate structure and wiring for WHATEVER files exist.
_DIALECT_FILES = _load_all_dialect_files()
ALL_DIALECTS = tuple(_DIALECT_FILES)


# ---------------------------------------------------------------------------
# Raw data fixtures (no app import needed)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def labels_data() -> dict:
    """Labels keyed by dialect: {code: {sub, main, sub_to_main}}, from each
    per-dialect file's ``labels`` block."""
    return {code: cfg["labels"] for code, cfg in _DIALECT_FILES.items()}


# ---------------------------------------------------------------------------
# Model-volume builder (what the fetch command produces, minus the real model)
# ---------------------------------------------------------------------------


def _write_model_unit(unit: Path) -> None:
    """Write the file-level shape of one installed model directory: a dummy ONNX
    graph and a training_config.json. The real load is mocked in these tests, so
    the bytes never matter, only that the completeness rules see a graph."""
    unit.mkdir(parents=True, exist_ok=True)
    (unit / "model.onnx").write_bytes(b"not a real onnx graph")
    (unit / "training_config.json").write_text(json.dumps({"max_length": 128}))


def write_dialect_dir(root: Path, code: str, cfg: dict, member_names=None) -> Path:
    """Materialize one dialect directory the way the fetch command would, from a
    config dict: dialect.json at the top, and the model snapshot below it. Pass
    ``member_names`` to build the ensemble layout (a model unit per name under
    members/<name>/); omit it for a single flat model unit. The dialect file
    itself carries only an ``hf_repo``, so which layout to build is the caller's
    choice, mirroring that a repo's snapshot decides it in production.
    """
    from app.common.dialect_schema import MEMBERS_DIRNAME

    d = root / code
    d.mkdir(parents=True, exist_ok=True)
    (d / "dialect.json").write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    if member_names:
        for name in member_names:
            _write_model_unit(d / MEMBERS_DIRNAME / name)
    else:
        _write_model_unit(d)
    return d


def make_model_volume(root: Path, codes=ALL_DIALECTS) -> Path:
    """Build a models directory shaped like a fetch-populated volume, one flat
    model directory per code, from the real dialect files. The ensemble layout
    (a members/ tree) is exercised explicitly where it matters, via
    write_dialect_dir(member_names=...).
    """
    root.mkdir(parents=True, exist_ok=True)
    for code in codes:
        write_dialect_dir(root, code, _DIALECT_FILES[code])
    return root


def with_aggregation(cfg: dict, *, bias=None, method=None) -> dict:
    """A copy of ``cfg`` with an aggregation block attached (combine method and/or
    per-sub-class bias). Everything else, including the single ``hf_repo``, is
    preserved; the ensemble-ness of a dialect is its on-volume members/ layout,
    not anything in the file, so this only tunes how the members combine.
    """
    out = dict(cfg)
    aggregation: dict = {}
    if method is not None:
        aggregation["method"] = method
    if bias is not None:
        aggregation["bias"] = list(bias)
    if aggregation:
        out["aggregation"] = aggregation
    return out


def mock_loaded_model():
    """A mock LoadedModel matching the ONNX-runtime field shape."""
    loaded = MagicMock()
    loaded.session = MagicMock()
    loaded.tokenizer = MagicMock()
    loaded.input_names = frozenset({"input_ids", "attention_mask"})
    loaded.max_length = 128
    return loaded


@pytest.fixture
def patched_model_load(monkeypatch):
    """Replace only the ML half of a dialect load with a mock.

    ``parse_dialect_dir`` (dialect.json parsing, schema validation, ONNX-file
    resolution) stays REAL, so the registry tests exercise the actual
    completeness rules; ``_load_model`` (tokenizer + ORT session) is the only
    mocked seam.
    """
    import app.classifier as clf

    monkeypatch.setattr(clf, "_load_model", lambda path, onnx_path: mock_loaded_model())


@pytest.fixture(autouse=True)
def registry_reset():
    """Every test starts and ends with an empty registry."""
    from app import registry

    registry.reset()
    yield
    registry.reset()


# ---------------------------------------------------------------------------
# The app harness: the real app booted against a throwaway volume
# ---------------------------------------------------------------------------


class ApiHarness:
    """The booted app plus the seams the tests poke.

    ``calls`` records every (dialect code, payload, lang) the mocked engine
    received, so routing tests can assert WHICH dialect served a request
    without hardcoding label strings. ``fail(exc)`` makes the next engine call
    raise, for the 503/504/500 surface.
    """

    def __init__(self, client) -> None:
        self.client = client
        self.calls: list[tuple[str, object, str]] = []
        self._raise: Exception | None = None

    def fail(self, exc: Exception) -> None:
        self._raise = exc


@pytest.fixture
def api_factory(tmp_path, monkeypatch, labels_data):
    """Boot the real app (lifespan + startup scan) against a built volume.

    A context manager factory so restart-pickup tests can boot, change the
    volume, and boot again: two boots ARE the operator's restart. Only
    ``_load_model`` and the two engine entry points are mocked; the scan,
    schema validation, lang resolution, and routing all run for real.
    """
    from fastapi.testclient import TestClient

    import app.classifier as clf
    import app.main as app_main
    from app import registry
    from app.classifier import ClassificationResult

    def _result_for(harness: ApiHarness, entry, text: str, lang: str):
        if harness._raise is not None:
            raise harness._raise
        cleaned = entry.config.preprocess_fn(text)
        if not cleaned:
            return ClassificationResult(
                is_valid=False, sub_class=None, main_class=None, confidence=None
            )
        # Real label tables for the entry's dialect; the route resolved lang to
        # a canonical code, so it is always a key here.
        labels = labels_data[entry.code]
        sub = {int(k): v for k, v in labels["sub"][lang].items()}
        main = {int(k): v for k, v in labels["main"][lang].items()}
        s2m = {int(k): v for k, v in labels["sub_to_main"].items()}
        return ClassificationResult(
            is_valid=True, sub_class=sub[0], main_class=main[s2m[0]], confidence=0.95
        )

    @contextmanager
    def boot(codes=ALL_DIALECTS, volume: Path | None = None):
        vol = volume if volume is not None else make_model_volume(tmp_path / "models", codes)
        monkeypatch.setattr(registry, "MODELS_DIR", vol)
        monkeypatch.setattr(clf, "_load_model", lambda path, onnx_path: mock_loaded_model())

        harness_box: list[ApiHarness] = []

        async def fake_single(entry, text, lang):
            harness_box[0].calls.append((entry.code, text, lang))
            return _result_for(harness_box[0], entry, text, lang)

        async def fake_batch(entry, texts, lang):
            harness_box[0].calls.append((entry.code, list(texts), lang))
            return [_result_for(harness_box[0], entry, t, lang) for t in texts]

        monkeypatch.setattr(app_main, "get_classification", fake_single)
        monkeypatch.setattr(app_main, "get_classifications_batch", fake_batch)

        with TestClient(app_main.app, raise_server_exceptions=False) as client:
            harness = ApiHarness(client)
            harness_box.append(harness)
            yield harness

    return boot


@pytest.fixture
def api(api_factory):
    """The default harness: every repo dialect installed on the volume."""
    with api_factory() as harness:
        yield harness


@pytest.fixture
def any_dialect() -> str:
    """One valid dialect code, chosen positionally (never hardcoded)."""
    return sorted(ALL_DIALECTS)[0]

"""The startup scan: discovery, completeness rules, and isolation.

Everything drives ``registry.scan_once`` directly against a throwaway volume
built from the real dialect files (see tests/conftest.py); only ``_load_model``
is mocked, so dialect.json parsing, schema validation, and ONNX-file resolution
run for real. No sleeps, no restarts: a second ``scan_once`` IS what a restart
does to the registry.
"""

import json
import shutil

import pytest

from app import registry
from app.common.config import MAX_LANG_LEN
from app.common.dialect_schema import RESERVED_CODES
from tests.conftest import ALL_DIALECTS, make_model_volume


pytestmark = pytest.mark.usefixtures("patched_model_load")


# =============================================================================
# Discovery: which directories count as candidates
# =============================================================================


class TestScanDiscovery:
    def test_scan_loads_every_complete_dir(self, tmp_path):
        volume = make_model_volume(tmp_path / "models")
        result = registry.scan_once(volume)
        assert sorted(result.loaded) == sorted(ALL_DIALECTS)
        assert result.failed == []
        assert registry.codes() == frozenset(ALL_DIALECTS)

    def test_scan_ignores_dotdirs(self, tmp_path):
        """The fetch command stages installs in dot-prefixed dirs; a scan that
        raced a fetch must never see the partial install."""
        volume = make_model_volume(tmp_path / "models")
        staging = volume / f".staging-{ALL_DIALECTS[0]}"
        staging.mkdir()
        (staging / "dialect.json").write_text("{not even json")
        result = registry.scan_once(volume)
        assert result.failed == []
        assert registry.codes() == frozenset(ALL_DIALECTS)

    @pytest.mark.parametrize(
        "name",
        [
            "UPPER",
            "with-hyphen",
            "digits123",
            "a" * (MAX_LANG_LEN + 1),
            *sorted(RESERVED_CODES),
        ],
    )
    def test_scan_ignores_invalid_names(self, tmp_path, name):
        """Only plausible dialect codes (the schema validator's own charset and
        length rules, minus the reserved names) are considered at all."""
        volume = make_model_volume(tmp_path / "models", codes=())
        bad = volume / name
        bad.mkdir()
        (bad / "dialect.json").write_text("{}")
        result = registry.scan_once(volume)
        assert result.loaded == [] and result.failed == []
        assert registry.codes() == frozenset()

    def test_scan_ignores_plain_files(self, tmp_path):
        volume = make_model_volume(tmp_path / "models", codes=())
        (volume / "stray.txt").write_text("not a dialect")
        result = registry.scan_once(volume)
        assert result.loaded == [] and result.failed == []

    def test_missing_models_dir_yields_empty_registry(self, tmp_path):
        result = registry.scan_once(tmp_path / "does-not-exist")
        assert result.loaded == [] and result.failed == []
        assert registry.codes() == frozenset()

    def test_empty_models_dir_yields_zero_dialects(self, tmp_path):
        volume = make_model_volume(tmp_path / "models", codes=())
        registry.scan_once(volume)
        assert registry.codes() == frozenset()


# =============================================================================
# Completeness: what a directory must hold to be loadable
# =============================================================================


class TestScanCompleteness:
    def test_dir_without_dialect_json_not_registered(self, tmp_path):
        volume = make_model_volume(tmp_path / "models")
        code = sorted(ALL_DIALECTS)[0]
        (volume / code / "dialect.json").unlink()
        result = registry.scan_once(volume)
        assert code in result.failed
        assert code not in registry.codes()

    def test_invalid_json_not_registered(self, tmp_path):
        volume = make_model_volume(tmp_path / "models")
        code = sorted(ALL_DIALECTS)[0]
        (volume / code / "dialect.json").write_text("{broken")
        result = registry.scan_once(volume)
        assert code in result.failed
        assert code not in registry.codes()

    def test_schema_invalid_config_not_registered(self, tmp_path):
        """A parseable dialect.json that fails the shared schema stays out."""
        volume = make_model_volume(tmp_path / "models")
        code = sorted(ALL_DIALECTS)[0]
        cfg_path = volume / code / "dialect.json"
        cfg = json.loads(cfg_path.read_text())
        del cfg["labels"]
        cfg_path.write_text(json.dumps(cfg))
        result = registry.scan_once(volume)
        assert code in result.failed
        assert code not in registry.codes()

    def test_dir_without_onnx_not_registered(self, tmp_path):
        volume = make_model_volume(tmp_path / "models")
        code = sorted(ALL_DIALECTS)[0]
        (volume / code / "model.onnx").unlink()
        result = registry.scan_once(volume)
        assert code in result.failed

    def test_dir_with_two_onnx_not_registered(self, tmp_path):
        """Two graphs and neither named model.onnx is ambiguous, so it stays out."""
        volume = make_model_volume(tmp_path / "models")
        code = sorted(ALL_DIALECTS)[0]
        (volume / code / "model.onnx").rename(volume / code / "graph_a.onnx")
        (volume / code / "graph_b.onnx").write_bytes(b"another")
        result = registry.scan_once(volume)
        assert code in result.failed

    def test_broken_dir_does_not_affect_siblings(self, tmp_path):
        """The 1.x guarantee, now at scan level: a broken dialect fails alone."""
        volume = make_model_volume(tmp_path / "models")
        codes = sorted(ALL_DIALECTS)
        broken = codes[0]
        (volume / broken / "dialect.json").write_text("{broken")
        result = registry.scan_once(volume)
        assert result.failed == [broken]
        assert registry.codes() == frozenset(codes[1:])


# =============================================================================
# Replacement semantics: a rescan rebuilds the registry from disk
# =============================================================================


class TestScanReplace:
    def test_second_scan_drops_removed_dialects(self, tmp_path):
        """What the operator's remove + restart does."""
        volume = make_model_volume(tmp_path / "models")
        registry.scan_once(volume)
        removed = sorted(ALL_DIALECTS)[-1]
        shutil.rmtree(volume / removed)
        registry.scan_once(volume)
        assert removed not in registry.codes()
        assert registry.codes() == frozenset(ALL_DIALECTS) - {removed}

    def test_second_scan_picks_up_added_dialects(self, tmp_path):
        """What the operator's fetch + restart does."""
        codes = sorted(ALL_DIALECTS)
        volume = make_model_volume(tmp_path / "models", codes=codes[:-1])
        registry.scan_once(volume)
        assert codes[-1] not in registry.codes()
        make_model_volume(volume, codes=codes[-1:])
        registry.scan_once(volume)
        assert registry.codes() == frozenset(codes)

    def test_held_entry_survives_a_replacing_scan(self, tmp_path):
        """A request that snapshotted its entry keeps a usable bundle even if
        the registry has since dropped the dialect (in-flight safety)."""
        volume = make_model_volume(tmp_path / "models")
        code = sorted(ALL_DIALECTS)[0]
        registry.scan_once(volume)
        entry = registry.get(code)
        shutil.rmtree(volume / code)
        registry.scan_once(volume)
        assert registry.get(code) is None
        assert entry.code == code
        assert callable(entry.config.preprocess_fn)
        assert entry.default_language in entry.supported_languages


# =============================================================================
# The loaded bundle reflects its dialect file
# =============================================================================


class TestLoadedEntry:
    @pytest.mark.parametrize("code", ALL_DIALECTS)
    def test_entry_mirrors_its_dialect_file(self, tmp_path, code):
        from tests.conftest import _DIALECT_FILES

        volume = make_model_volume(tmp_path / "models", codes=(code,))
        registry.scan_once(volume)
        entry = registry.get(code)
        cfg = _DIALECT_FILES[code]
        assert entry.supported_languages == frozenset(cfg["languages"])
        assert entry.default_language == sorted(cfg["languages"])[0]
        declared_aliases = {
            alias: lang
            for lang, meta in cfg["languages"].items()
            for alias in meta.get("aliases", [])
        }
        assert entry.aliases == declared_aliases
        assert entry.config.name == cfg["name"]
        assert set(entry.config.sub_to_main) == {int(k) for k in cfg["labels"]["sub_to_main"]}

    def test_each_entry_has_its_own_cache_and_lock(self, tmp_path):
        """Per-dialect caches keep same-text keys from colliding across
        dialects; per-dialect locks keep tokenizer serialization local."""
        if len(ALL_DIALECTS) < 2:
            pytest.skip("needs at least two dialects to compare")
        volume = make_model_volume(tmp_path / "models")
        registry.scan_once(volume)
        a, b = (registry.get(c) for c in sorted(ALL_DIALECTS)[:2])
        assert a.cache is not b.cache
        assert a.tokenizer_lock is not b.tokenizer_lock

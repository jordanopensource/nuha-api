"""The fetch command: validated installs, atomic activation, clean removal.

Loads scripts/fetch_models.py by path (scripts/ is not a package) and mocks
only ``huggingface_hub.snapshot_download`` (writing a dummy snapshot), so the
staging/rename/trash mechanics, the validate-before-download gate, and the
CLI surface all run for real against a tmp models directory.
"""

import importlib.util
import json
import sys
from unittest.mock import MagicMock

import pytest

from app.common.dialect_schema import RESERVED_CODES
from tests.conftest import ALL_DIALECTS, PROJECT_ROOT, _dialect_file


def _load_fetch_module():
    path = PROJECT_ROOT / "scripts" / "fetch_models.py"
    spec = importlib.util.spec_from_file_location("fetch_models_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fetch = _load_fetch_module()


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    """A writable models volume the module under test points at."""
    volume = tmp_path / "models"
    volume.mkdir()
    monkeypatch.setenv("MODELS_DIR", str(volume))
    return volume


@pytest.fixture
def hub(monkeypatch):
    """Mock huggingface_hub with a snapshot_download that writes a dummy model
    snapshot (one model.onnx + a tokenizer file) into local_dir."""

    def fake_snapshot_download(repo_id, local_dir, revision=None):
        from pathlib import Path

        target = Path(local_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.onnx").write_bytes(b"dummy graph")
        (target / "tokenizer.json").write_text("{}")
        return str(target)

    module = MagicMock()
    module.snapshot_download = MagicMock(side_effect=fake_snapshot_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return module


class TestAdd:
    def test_add_installs_via_staging_and_rename(self, models_dir, hub, any_dialect):
        fetch.main(["add", any_dialect])
        target = models_dir / any_dialect
        assert (target / "model.onnx").is_file()
        assert (target / "dialect.json").is_file()
        # No staging or trash residue survives a successful install.
        assert not list(models_dir.glob(".staging-*"))
        assert not list(models_dir.glob(".trash-*"))

    def test_add_writes_the_validated_config_as_dialect_json(self, models_dir, hub, any_dialect):
        fetch.main(["add", any_dialect])
        installed = json.loads((models_dir / any_dialect / "dialect.json").read_text())
        shipped = json.loads(_dialect_file(any_dialect).read_text(encoding="utf-8"))
        assert installed == shipped

    def test_add_validates_before_download(self, models_dir, hub, tmp_path, any_dialect):
        """A structurally broken config is refused before any network call."""
        cfg = json.loads(_dialect_file(any_dialect).read_text(encoding="utf-8"))
        del cfg["labels"]
        broken = tmp_path / "broken.json"
        broken.write_text(json.dumps(cfg))
        with pytest.raises(SystemExit, match="fails the schema"):
            fetch.main(["add", "xyz", "--file", str(broken)])
        hub.snapshot_download.assert_not_called()
        assert not (models_dir / "xyz").exists()

    @pytest.mark.parametrize("code", sorted(RESERVED_CODES))
    def test_add_refuses_reserved_codes(self, models_dir, hub, tmp_path, any_dialect, code):
        cfg_file = tmp_path / f"{code}.json"
        cfg_file.write_text(_dialect_file(any_dialect).read_text(encoding="utf-8"))
        with pytest.raises(SystemExit, match="fails the schema"):
            fetch.main(["add", code, "--file", str(cfg_file)])
        hub.snapshot_download.assert_not_called()

    def test_add_unknown_code_names_the_shipped_dialects(self, models_dir, hub):
        with pytest.raises(SystemExit, match="Shipped dialects"):
            fetch.main(["add", "zzz"])
        hub.snapshot_download.assert_not_called()

    def test_add_replaces_an_existing_install(self, models_dir, hub, any_dialect):
        stale = models_dir / any_dialect
        stale.mkdir()
        (stale / "leftover.bin").write_bytes(b"old")
        fetch.main(["add", any_dialect])
        target = models_dir / any_dialect
        assert not (target / "leftover.bin").exists()
        assert (target / "dialect.json").is_file()
        assert not list(models_dir.glob(".trash-*"))

    def test_failed_download_leaves_no_partial_dir(self, models_dir, hub, any_dialect):
        hub.snapshot_download.side_effect = RuntimeError("network down")
        with pytest.raises(RuntimeError, match="network down"):
            fetch.main(["add", any_dialect])
        assert not (models_dir / any_dialect).exists()
        assert not list(models_dir.glob(".staging-*"))

    def test_snapshot_without_onnx_refused(self, models_dir, hub, any_dialect):
        def no_onnx(repo_id, local_dir, revision=None):
            from pathlib import Path

            Path(local_dir).mkdir(parents=True, exist_ok=True)
            (Path(local_dir) / "tokenizer.json").write_text("{}")

        hub.snapshot_download.side_effect = no_onnx
        with pytest.raises(SystemExit, match="no single ONNX graph"):
            fetch.main(["add", any_dialect])
        assert not (models_dir / any_dialect).exists()
        assert not list(models_dir.glob(".staging-*"))

    def test_add_all_installs_every_repo_dialect(self, models_dir, hub):
        fetch.main(["add", "--all"])
        installed = sorted(p.name for p in models_dir.iterdir() if p.is_dir())
        assert installed == sorted(ALL_DIALECTS)

    def test_add_creates_a_missing_models_dir(self, monkeypatch, tmp_path, hub, any_dialect):
        """A fresh clone has no ./models; the add path creates it instead of
        refusing (in a container /models is baked into the image anyway)."""
        volume = tmp_path / "fresh" / "models"
        monkeypatch.setenv("MODELS_DIR", str(volume))
        fetch.main(["add", any_dialect])
        assert (volume / any_dialect / "dialect.json").is_file()

    def test_stale_staging_cleaned_before_install(self, models_dir, hub, any_dialect):
        stale = models_dir / ".staging-old"
        stale.mkdir()
        (stale / "junk").write_bytes(b"x")
        fetch.main(["add", any_dialect])
        assert not stale.exists()


class TestRemove:
    def test_remove_deletes_the_dialect_dir(self, models_dir, hub, any_dialect):
        fetch.main(["add", any_dialect])
        fetch.main(["remove", any_dialect])
        assert not (models_dir / any_dialect).exists()
        assert not list(models_dir.glob(".trash-*"))

    def test_remove_missing_code_errors(self, models_dir):
        with pytest.raises(SystemExit, match="not installed"):
            fetch.main(["remove", "zzz"])

    @pytest.mark.parametrize("bad", ["../escape", "UPPER", "with-hyphen", "a" * 17, "."])
    def test_remove_refuses_path_shaped_codes(self, models_dir, tmp_path, bad):
        """`remove` validates the code like the scanner does, so a path-shaped
        argument can never rename or delete anything outside the volume."""
        outside = tmp_path / "escape"
        outside.mkdir()
        with pytest.raises(SystemExit, match="invalid dialect code"):
            fetch.main(["remove", bad])
        assert outside.exists()

    def test_remove_with_missing_models_dir_errors_cleanly(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MODELS_DIR", str(tmp_path / "never-created"))
        with pytest.raises(SystemExit, match="not installed"):
            fetch.main(["remove", "zzz"])


class TestList:
    def test_list_reports_installed_and_shipped(self, models_dir, hub, any_dialect, capsys):
        fetch.main(["add", any_dialect])
        fetch.main(["list"])
        out = capsys.readouterr().out
        assert any_dialect in out
        assert "shipped configs" in out

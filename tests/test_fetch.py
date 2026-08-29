"""The fetch command: validated installs, atomic activation, clean removal.

Loads scripts/fetch_models.py by path (scripts/ is not a package) and mocks
``huggingface_hub`` as a whole: snapshot_download writes a dummy snapshot and
HfApi resolves a fake head revision (move it via ``hub.head``). The
staging/rename/trash mechanics, the validate-before-download gate, the
revision records, and the CLI surface all run for real against a tmp models
directory.
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
    """Mock huggingface_hub: snapshot_download writes a dummy model snapshot
    (one model.onnx + a tokenizer file) into local_dir, and HfApi resolves a
    stable fake head revision (set hub.head to move the fake repo head)."""

    def fake_snapshot_download(repo_id, local_dir, revision=None):
        from pathlib import Path

        target = Path(local_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.onnx").write_bytes(b"dummy graph")
        (target / "tokenizer.json").write_text("{}")
        return str(target)

    module = MagicMock()
    module.snapshot_download = MagicMock(side_effect=fake_snapshot_download)
    module.head = "shaaaa000head"

    def fake_model_info(repo_id, revision=None):
        info = MagicMock()
        info.sha = revision or module.head
        return info

    module.HfApi = MagicMock(
        return_value=MagicMock(model_info=MagicMock(side_effect=fake_model_info))
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return module


class TestAdd:
    def test_add_installs_via_staging_and_rename(self, models_dir, hub, any_dialect):
        fetch.main(["add", any_dialect])
        target = models_dir / any_dialect
        assert (target / "model.onnx").is_file()
        assert (target / "dialect.json").is_file()
        record = json.loads((target / ".install.json").read_text())
        assert record["revision"] == hub.head
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

    def test_stale_staging_cleaned_only_for_this_runs_codes(self, models_dir, hub, any_dialect):
        """Cleanup is scoped: this run's own stale staging goes, another
        code's in-flight staging survives (a concurrent fetch elsewhere)."""
        mine = models_dir / f".staging-{any_dialect}"
        mine.mkdir()
        (mine / "junk").write_bytes(b"x")
        other = models_dir / ".staging-zzz"
        other.mkdir()
        (other / "inflight").write_bytes(b"x")
        fetch.main(["add", any_dialect])
        assert not mine.exists()
        assert other.exists()


class TestEnsure:
    def test_missing_dialect_is_installed(self, models_dir, hub, any_dialect):
        fetch.main(["ensure", any_dialect])
        assert (models_dir / any_dialect / "dialect.json").is_file()
        assert hub.snapshot_download.call_count == 1

    def test_complete_install_is_verified_without_network(self, models_dir, hub, any_dialect):
        fetch.main(["add", any_dialect])
        hub.snapshot_download.reset_mock()
        fetch.main(["ensure", any_dialect])
        hub.snapshot_download.assert_not_called()

    def test_incomplete_install_is_healed(self, models_dir, hub, any_dialect):
        fetch.main(["add", any_dialect])
        (models_dir / any_dialect / "model.onnx").unlink()
        hub.snapshot_download.reset_mock()
        fetch.main(["ensure", any_dialect])
        assert (models_dir / any_dialect / "model.onnx").is_file()
        assert hub.snapshot_download.call_count == 1

    def test_config_only_drift_rewrites_in_place(self, models_dir, hub, any_dialect, tmp_path):
        """A changed dialect config over an intact snapshot (a label tweak in
        an image upgrade) is fixed by rewriting dialect.json, with no
        download: models-init must not need hub egress for it."""
        fetch.main(["add", any_dialect])
        cfg = json.loads(_dialect_file(any_dialect).read_text(encoding="utf-8"))
        cfg["name"] = cfg["name"] + " v2"
        changed = tmp_path / "changed.json"
        changed.write_text(json.dumps(cfg, ensure_ascii=False))
        hub.snapshot_download.reset_mock()
        fetch.main(["ensure", any_dialect, "--file", str(changed)])
        hub.snapshot_download.assert_not_called()
        installed = json.loads((models_dir / any_dialect / "dialect.json").read_text())
        assert installed["name"].endswith("v2")

    def test_update_skips_when_head_matches_record(self, models_dir, hub, any_dialect):
        fetch.main(["add", any_dialect])
        hub.snapshot_download.reset_mock()
        fetch.main(["ensure", any_dialect, "--update"])
        hub.snapshot_download.assert_not_called()

    def test_update_keeps_a_verified_install_when_resolution_fails(
        self, models_dir, hub, any_dialect
    ):
        """A hub metadata outage must not turn a complete, current install
        into a re-download (or an init failure)."""
        fetch.main(["add", any_dialect])
        hub.snapshot_download.reset_mock()
        api = hub.HfApi.return_value
        api.model_info.side_effect = RuntimeError("hub metadata down")
        fetch.main(["ensure", any_dialect, "--update"])
        hub.snapshot_download.assert_not_called()

    def test_update_reinstalls_once_when_record_is_missing(self, models_dir, hub, any_dialect):
        """A legacy install without .install.json gets one refresh to
        establish the record, then verifies."""
        fetch.main(["add", any_dialect])
        (models_dir / any_dialect / ".install.json").unlink()
        hub.snapshot_download.reset_mock()
        fetch.main(["ensure", any_dialect, "--update"])
        assert hub.snapshot_download.call_count == 1
        hub.snapshot_download.reset_mock()
        fetch.main(["ensure", any_dialect, "--update"])
        hub.snapshot_download.assert_not_called()

    def test_heal_preserves_a_pinned_revision(self, models_dir, hub, any_dialect):
        """A broken install that was pinned heals at its pin, not the head."""
        fetch.main(["add", any_dialect, "--revision", "pinnedsha42"])
        (models_dir / any_dialect / "model.onnx").unlink()
        hub.head = "someotherhead"
        fetch.main(["ensure", any_dialect])
        record = json.loads((models_dir / any_dialect / ".install.json").read_text())
        assert record["revision"] == "pinnedsha42"
        assert record["pinned"] is True

    def test_update_leaves_pinned_installs_alone(self, models_dir, hub, any_dialect):
        fetch.main(["add", any_dialect, "--revision", "pinnedsha42"])
        hub.head = "someotherhead"
        hub.snapshot_download.reset_mock()
        fetch.main(["ensure", any_dialect, "--update"])
        hub.snapshot_download.assert_not_called()

    def test_ensure_revision_flag_pins(self, models_dir, hub, any_dialect):
        fetch.main(["ensure", any_dialect, "--revision", "pinnedsha42"])
        record = json.loads((models_dir / any_dialect / ".install.json").read_text())
        assert record == {
            "hf_repo": json.loads(_dialect_file(any_dialect).read_text(encoding="utf-8"))[
                "hf_repo"
            ],
            "revision": "pinnedsha42",
            "pinned": True,
        }
        hub.snapshot_download.reset_mock()
        fetch.main(["ensure", any_dialect, "--revision", "pinnedsha42"])
        hub.snapshot_download.assert_not_called()

    def test_missing_tokenizer_artifact_counts_as_incomplete(self, models_dir, hub, any_dialect):
        """A snapshot without any tokenizer file would load-fail in the api,
        so ensure heals it instead of verifying it green."""
        fetch.main(["add", any_dialect])
        (models_dir / any_dialect / "tokenizer.json").unlink()
        hub.snapshot_download.reset_mock()
        fetch.main(["ensure", any_dialect])
        assert hub.snapshot_download.call_count == 1

    def test_one_failing_dialect_does_not_block_siblings(self, models_dir, hub):
        """Per-dialect isolation on the init path: with one dialect complete
        and the hub down, ensure still verifies the healthy one and exits
        nonzero only because the missing one stayed unservable."""
        codes = sorted(ALL_DIALECTS)
        fetch.main(["ensure", codes[0]])
        hub.snapshot_download.side_effect = RuntimeError("hub down")
        api = hub.HfApi.return_value
        api.model_info.side_effect = RuntimeError("hub down")
        with pytest.raises(SystemExit, match="unservable after ensure"):
            fetch.main(["ensure", "--all"])
        assert (models_dir / codes[0] / "dialect.json").is_file()

    def test_all_servable_despite_update_failure_exits_zero(self, models_dir, hub):
        """When every install is complete, a failed update attempt is a
        warning, not an init failure: the api must still boot."""
        fetch.main(["ensure", "--all"])
        hub.snapshot_download.side_effect = RuntimeError("hub down")
        api = hub.HfApi.return_value
        api.model_info.side_effect = RuntimeError("hub down")
        fetch.main(["ensure", "--all", "--update"])

    def test_update_reinstalls_when_head_moved(self, models_dir, hub, any_dialect):
        fetch.main(["add", any_dialect])
        hub.head = "shabbbb111new"
        hub.snapshot_download.reset_mock()
        fetch.main(["ensure", any_dialect, "--update"])
        assert hub.snapshot_download.call_count == 1
        record = json.loads((models_dir / any_dialect / ".install.json").read_text())
        assert record["revision"] == "shabbbb111new"

    def test_ensure_all_covers_every_repo_dialect(self, models_dir, hub):
        fetch.main(["ensure", "--all"])
        installed = sorted(p.name for p in models_dir.iterdir() if p.is_dir())
        assert installed == sorted(ALL_DIALECTS)


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

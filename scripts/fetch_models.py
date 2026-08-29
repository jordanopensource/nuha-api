"""Install, verify, remove, and list dialect models on the models volume.

A dialect is a directory <MODELS_DIR>/<code>/ holding dialect.json plus the
model snapshot, and this command manages those directories. The api service
scans the volume once at startup, so every change here is followed by
`docker compose restart api`; the image and the repo stay untouched.

Two services run this script over the same volume (the api mounts it
read-only). `models-init` runs `ensure --all` before the api starts, the
initContainer pattern: verify each shipped dialect, download what is missing
or broken, and with --update also refresh installs whose recorded revision
differs from the repo head. `fetch` is the manual tool for add, remove, and
trial installs:

    docker compose run --rm fetch add arz          # a repo-shipped dialect
    docker compose run --rm fetch add xyz --file xyz.json   # a trial dialect
    docker compose run --rm fetch add --all
    docker compose run --rm fetch ensure --all     # verify; download only gaps
    docker compose run --rm fetch ensure --all --update    # also pull updates
    docker compose run --rm fetch remove arz
    docker compose run --rm fetch list
    docker compose restart api

Safety properties:
  - The config is validated against the shared schema before anything is
    downloaded; a broken file cannot reach the volume.
  - Downloads land in a dot-prefixed staging directory the api's scanner
    ignores, then activate with an atomic rename, so a scan cannot see a
    partial install. Cleanup of stale staging only touches the codes the
    current run is working on, so a concurrent run for another code is safe;
    still avoid two concurrent runs for the SAME code (they share a staging
    directory and can clobber each other).
  - Every install records the resolved model revision (and whether it was
    operator-pinned) in the directory's .install.json. `ensure` heals a
    broken pinned install at its pinned revision, and --update leaves pinned
    installs alone; only unpinned installs track the repo head.
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.common.config import MAX_LANG_LEN  # noqa: E402
from app.common.dialect_schema import validate_dialect_config  # noqa: E402


DIALECTS_DIR = ROOT / "app" / "dialects"


# The same code shape the registry scanner accepts; used here so `remove` can
# never be pointed outside the models directory.
_CODE_RE = re.compile(rf"[a-z]{{1,{MAX_LANG_LEN}}}")

_INSTALL_META = ".install.json"

# Files a usable snapshot's tokenizer can arrive as; at least one must exist
# for AutoTokenizer.from_pretrained to stand a chance at load time.
_TOKENIZER_FILES = ("tokenizer.json", "vocab.txt", "spiece.model", "sentencepiece.bpe.model")


def _models_dir() -> Path:
    """Where the model directories live (created on install if missing).

    `or` so an empty MODELS_DIR= line in an env file means the default too. In
    deployment this is the mounted volume (read-write for this service).
    """
    return Path(os.getenv("MODELS_DIR") or "./models")


def _check_code(code: str) -> None:
    """Refuse a code that is not a plausible dialect directory name.

    `add` and `ensure` get this via the full schema validation; `remove` needs
    it directly so a path-shaped argument (../something) cannot rename or
    delete anything outside the models directory.
    """
    if not _CODE_RE.fullmatch(code):
        raise _fail(
            f"invalid dialect code {code!r}: expected 1-{MAX_LANG_LEN} lowercase ASCII letters"
        )


def _fail(msg: str) -> "SystemExit":
    return SystemExit(f"ERROR: {msg}")


def _load_config(code: str, file: str | None) -> dict:
    """The dialect config to install: --file wins, else the repo-shipped file."""
    path = Path(file) if file else DIALECTS_DIR / f"{code}.json"
    if not path.is_file():
        shipped = ", ".join(sorted(p.stem for p in DIALECTS_DIR.glob("*.json")))
        raise _fail(
            f"no config for dialect '{code}' at {path}. "
            f"Shipped dialects: {shipped}. For a new one, pass --file."
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise _fail(f"{path} is not valid JSON: {e}") from e


def _validate_or_die(code: str, cfg: dict) -> None:
    """Refuse a structurally broken config before any network or disk work.
    The same schema definition the pre-commit hook and the runtime scan use."""
    problems = validate_dialect_config(code, cfg)
    if problems:
        listing = "\n".join(f"  - {p}" for p in problems)
        raise _fail(f"dialect config for '{code}' fails the schema:\n{listing}")


def _cleanup_stale(models_dir: Path, codes: list[str]) -> None:
    """Remove leftovers from crashed earlier runs, scoped to this run's codes.

    Trash dirs are always safe to clear (they are pid-suffixed discards).
    Staging dirs are cleared only for the codes this run is about to work on,
    so a concurrent run installing a DIFFERENT code keeps its in-flight
    download. Both kinds are dot-prefixed, invisible to the api's scanner.
    """
    for entry in models_dir.iterdir():
        if not entry.is_dir():
            continue
        if entry.name.startswith(".trash-") or (
            entry.name.startswith(".staging-") and entry.name[len(".staging-") :] in codes
        ):
            print(f"cleaning stale {entry.name}")
            shutil.rmtree(entry)


def _find_onnx(path: Path) -> Path | None:
    """The snapshot's ONNX graph: model.onnx, else a single *.onnx, else None.
    Mirrors the app's resolution rule without importing the ML stack."""
    preferred = path / "model.onnx"
    if preferred.is_file():
        return preferred
    candidates = sorted(path.glob("*.onnx"))
    return candidates[0] if len(candidates) == 1 else None


def _resolve_revision(repo: str, revision: str | None) -> str | None:
    """The exact commit sha this install will pin: the given revision resolved
    against the hub, else the repo head. None when the hub cannot answer (the
    download then proceeds unpinned, and the record stays honest about it)."""
    from huggingface_hub import HfApi

    try:
        sha = HfApi().model_info(repo, revision=revision).sha
    except Exception as e:
        print(f"note: could not resolve a revision for {repo} ({e}); installing unpinned")
        return revision
    return sha if isinstance(sha, str) and sha else revision


def _read_meta(target: Path) -> dict:
    meta = target / _INSTALL_META
    if not meta.is_file():
        return {}
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _model_files_complete(target: Path) -> bool:
    """The snapshot half of completeness: one resolvable ONNX graph and at
    least one tokenizer artifact the loader can read."""
    if _find_onnx(target) is None:
        return False
    return any((target / name).is_file() for name in _TOKENIZER_FILES)


def _installed_matches(target: Path, cfg: dict) -> bool:
    """A complete install of THIS config.

    Mirrors the file-level subset of the api's loadability rules: dialect.json
    present and equal to the config being ensured, one resolvable ONNX graph,
    and a tokenizer artifact on disk. The api additionally checks the label
    width against the graph and runs a self-test inference at load; those need
    the ML stack, so a directory can still fail there and stay out of service.
    """
    dialect_json = target / "dialect.json"
    if not dialect_json.is_file():
        return False
    try:
        installed_cfg = json.loads(dialect_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return installed_cfg == cfg and _model_files_complete(target)


def _write_config(target: Path, cfg: dict) -> None:
    """Replace an install's dialect.json atomically (temp file + rename)."""
    tmp = target / ".dialect.json.tmp"
    tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, target / "dialect.json")


def install(
    code: str,
    cfg: dict,
    revision: str | None,
    *,
    resolved: str | None = None,
    pinned: bool = False,
) -> None:
    """Download + activate one dialect: snapshot into a scanner-invisible
    staging dir, write dialect.json and the revision record, sanity-check the
    graph, then activate with an atomic rename. Pass ``resolved`` when the
    caller already resolved the revision, so the compare, the download, and
    the record share one resolution."""
    # Imported lazily: validation and `list` work without network intent, and
    # the tests mock this symbol.
    from huggingface_hub import snapshot_download

    models_dir = _models_dir()
    # Locally ./models will not exist on a fresh clone; in a container /models
    # is baked into the image, so this is a no-op there.
    models_dir.mkdir(parents=True, exist_ok=True)

    staging = models_dir / f".staging-{code}"
    if staging.exists():
        shutil.rmtree(staging)

    if resolved is None:
        resolved = _resolve_revision(cfg["hf_repo"], revision)
    print(f"downloading {cfg['hf_repo']}" + (f" @ {resolved}" if resolved else "") + " ...")
    try:
        snapshot_download(repo_id=cfg["hf_repo"], local_dir=str(staging), revision=resolved)
        (staging / "dialect.json").write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (staging / _INSTALL_META).write_text(
            json.dumps({"hf_repo": cfg["hf_repo"], "revision": resolved, "pinned": pinned}) + "\n",
            encoding="utf-8",
        )
        if _find_onnx(staging) is None:
            raise _fail(
                f"snapshot of {cfg['hf_repo']} holds no single ONNX graph; refusing to install it"
            )
    except BaseException:
        # A partial staging dir is invisible to the scanner either way; remove
        # it so disk usage stays bounded.
        shutil.rmtree(staging, ignore_errors=True)
        raise

    target = models_dir / code
    trash = models_dir / f".trash-{code}-{os.getpid()}"
    if target.exists():
        os.rename(target, trash)
    os.rename(staging, target)
    if trash.exists():
        shutil.rmtree(trash)
    print(f"installed dialect '{code}' at {target}")


def remove(code: str) -> None:
    """Deactivate + delete one dialect: rename out first (atomic, so the
    scanner cannot see a half-deleted directory), then delete."""
    _check_code(code)
    models_dir = _models_dir()
    target = models_dir / code
    if not target.is_dir():
        installed = ", ".join(
            sorted(p.name for p in models_dir.iterdir() if p.is_dir())
            if models_dir.is_dir()
            else []
        )
        raise _fail(f"dialect '{code}' is not installed. Installed: {installed or '(none)'}")
    trash = models_dir / f".trash-{code}-{os.getpid()}"
    os.rename(target, trash)
    shutil.rmtree(trash)
    print(f"removed dialect '{code}' from {models_dir}")


def ensure(code: str, cfg: dict, update: bool, revision: str | None = None) -> str:
    """Bring one dialect to a served state with the least work.

    The decision ladder, in order:
    - An explicit --revision: verified when the install already records it,
      else a pinned (re)install at exactly that revision.
    - Config-only drift (snapshot complete, same hf_repo, different
      dialect.json): rewrite the config in place, no download.
    - Missing or incomplete: reinstall, at the recorded revision when the
      previous install was pinned (a heal keeps the pin), else at the head.
    - Complete without --update: verified, no network at all.
    - Complete with --update: pinned installs are left alone; otherwise
      resolve the head and reinstall only when it differs from the recorded
      revision. When the hub cannot answer, staleness cannot be established,
      so the verified install is kept.
    Returns what happened: installed, updated, or verified.
    """
    target = _models_dir() / code
    meta = _read_meta(target)
    recorded = meta.get("revision")
    pinned = bool(meta.get("pinned"))

    if revision is not None:
        if _installed_matches(target, cfg) and recorded == revision:
            print(f"{code}: verified (pinned at {revision})")
            return "verified"
        print(f"{code}: installing pinned revision {revision}")
        install(code, cfg, revision, pinned=True)
        return "installed"

    if not _installed_matches(target, cfg):
        if _model_files_complete(target) and meta.get("hf_repo") == cfg.get("hf_repo"):
            print(f"{code}: config drifted; rewriting dialect.json in place")
            _write_config(target, cfg)
            return "updated"
        heal_rev = recorded if pinned else None
        label = f" at pinned {heal_rev[:12]}" if heal_rev else ""
        print(f"{code}: missing or incomplete; installing{label}")
        install(code, cfg, heal_rev, pinned=pinned and heal_rev is not None)
        return "installed"

    if not update:
        print(f"{code}: verified (present and complete)")
        return "verified"
    if pinned:
        print(f"{code}: verified (pinned at {recorded and recorded[:12]}; --update skips pins)")
        return "verified"
    remote = _resolve_revision(cfg["hf_repo"], None)
    if remote is None:
        print(f"{code}: verified (revision resolution unavailable; keeping the install)")
        return "verified"
    if recorded == remote:
        print(f"{code}: verified (up to date at {remote[:12]})")
        return "verified"
    print(f"{code}: recorded={recorded and recorded[:12]} remote={remote[:12]}; updating")
    install(code, cfg, None, resolved=remote)
    return "updated"


def list_dialects() -> None:
    """Installed vs shipped, a cheap operator overview."""
    models_dir = _models_dir()
    installed = (
        sorted(p.name for p in models_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
        if models_dir.is_dir()
        else []
    )
    shipped = sorted(p.stem for p in DIALECTS_DIR.glob("*.json"))
    print(f"installed ({_models_dir()}): {', '.join(installed) or '(none)'}")
    print(f"shipped configs (app/dialects): {', '.join(shipped) or '(none)'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="fetch_models.py",
        description="Manage dialect models on the models volume.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", help="download and activate a dialect")
    add.add_argument("code", nargs="?", help="dialect code (a repo-shipped config)")
    add.add_argument("--all", action="store_true", help="install every repo-shipped dialect")
    add.add_argument("--file", help="path to a dialect config JSON (for a trial dialect)")
    add.add_argument("--revision", help="pin a model revision (default: the repo head)")

    ens = sub.add_parser(
        "ensure", help="verify installs; download only what is missing (the init path)"
    )
    ens.add_argument("code", nargs="?", help="dialect code (a repo-shipped config)")
    ens.add_argument("--all", action="store_true", help="ensure every repo-shipped dialect")
    ens.add_argument("--file", help="path to a dialect config JSON (for a trial dialect)")
    ens.add_argument("--revision", help="verify or (re)install at exactly this pinned revision")
    ens.add_argument(
        "--update",
        action="store_true",
        help="also reinstall unpinned installs the head has left behind",
    )

    rem = sub.add_parser("remove", help="deactivate and delete a dialect")
    rem.add_argument("code")

    sub.add_parser("list", help="show installed and shipped dialects")

    args = parser.parse_args(argv)

    if args.command == "list":
        list_dialects()
        return
    if args.command == "remove":
        remove(args.code)
        return

    # add / ensure share the selection rules
    if args.all and (args.code or args.file):
        raise _fail("--all takes no code or --file")
    if not args.all and not args.code:
        raise _fail("pass a dialect code or --all")

    codes = sorted(p.stem for p in DIALECTS_DIR.glob("*.json")) if args.all else [args.code]
    if _models_dir().is_dir():
        _cleanup_stale(_models_dir(), codes)

    outcomes: dict[str, str] = {}
    failures: dict[str, str] = {}
    for code in codes:
        cfg = _load_config(code, None if args.all else args.file)
        _validate_or_die(code, cfg)
        if args.command == "ensure":
            # Each dialect succeeds or fails alone, matching the api's own
            # per-dialect isolation: one unreachable model must not keep an
            # otherwise healthy volume from booting the api.
            try:
                outcomes[code] = ensure(code, cfg, args.update, args.revision)
            except (Exception, SystemExit) as e:
                outcomes[code] = "failed"
                failures[code] = str(e)
                print(f"{code}: FAILED ({e})")
        else:
            install(code, cfg, args.revision, pinned=args.revision is not None)
            outcomes[code] = "installed"
    summary = ", ".join(f"{code}: {what}" for code, what in outcomes.items())
    print(f"done ({summary}).")
    if any(what not in ("verified", "failed") for what in outcomes.values()):
        print("Restart the api service to pick the change up: docker compose restart api")
    if failures:
        # Exit nonzero only when a failed dialect is not servable: a complete
        # install whose update attempt failed still serves, so the init path
        # must not block the api over it.
        unservable = [
            code
            for code in failures
            if not _installed_matches(
                _models_dir() / code, _load_config(code, None if args.all else args.file)
            )
        ]
        if unservable:
            raise _fail(f"unservable after ensure: {', '.join(unservable)}")
        print(f"warning: update failures on servable installs: {', '.join(failures)}")


if __name__ == "__main__":
    main()

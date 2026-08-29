"""Install, replace, remove, and list dialect models on the models volume.

The operator's control surface for the running stack: a dialect is a directory
<MODELS_DIR>/<code>/ holding dialect.json plus the model snapshot, and this
command manages those directories. The api service scans the volume once at
startup, so the flow is fetch (or remove), then restart the api service. No
image rebuild, no compose edit, no repo change.

In deployment it runs as the compose `fetch` service, the volume's only
writer (the api mounts it read-only):

    docker compose run --rm fetch add arz          # a repo-shipped dialect
    docker compose run --rm fetch add xyz --file xyz.json   # a trial dialect
    docker compose run --rm fetch add --all
    docker compose run --rm fetch remove arz
    docker compose run --rm fetch list
    docker compose restart api

Safety properties:
  - The config is validated against the shared schema BEFORE anything is
    downloaded; a broken file never reaches the volume.
  - Run ONE fetch at a time: two concurrent runs for the same code share a
    staging directory and can clobber each other. The api is safe either way
    (it only ever sees complete, atomically renamed directories), but the
    losing fetch run can fail or be undone.
  - Downloads land in a dot-prefixed staging directory the api's scanner
    ignores, then activate with an atomic rename, so a scan can never see a
    partial install.
  - Model revisions are deliberately unpinned by default (a re-fetch picks up
    the HF repo's head, the same trade-off the old image bake had); pass
    --revision to pin one.
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


def _models_dir() -> Path:
    """Where the model directories live (created on install if missing).

    `or` so an empty MODELS_DIR= line in an env file means the default too. In
    deployment this is the mounted volume (read-write for this service).
    """
    return Path(os.getenv("MODELS_DIR") or "./models")


def _check_code(code: str) -> None:
    """Refuse a code that is not a plausible dialect directory name.

    `add` gets this via the full schema validation; `remove` needs it directly
    so a path-shaped argument (../something) can never rename or delete
    anything outside the models directory.
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


def _cleanup_stale(models_dir: Path) -> None:
    """Remove staging/trash leftovers from a crashed earlier run. They are
    dot-prefixed, so the api's scanner never saw them; this is just hygiene."""
    for entry in models_dir.iterdir():
        if entry.is_dir() and (
            entry.name.startswith(".staging-") or entry.name.startswith(".trash-")
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


def install(code: str, cfg: dict, revision: str | None) -> None:
    """Download + activate one dialect: snapshot into a scanner-invisible
    staging dir, write dialect.json, sanity-check the graph, atomic rename."""
    # Imported lazily: validation and `list` must work without network intent,
    # and the tests mock this symbol.
    from huggingface_hub import snapshot_download

    models_dir = _models_dir()
    # Locally ./models will not exist on a fresh clone; in a container /models
    # is baked into the image, so this is a no-op there.
    models_dir.mkdir(parents=True, exist_ok=True)

    staging = models_dir / f".staging-{code}"
    if staging.exists():
        shutil.rmtree(staging)

    print(f"downloading {cfg['hf_repo']}" + (f" @ {revision}" if revision else "") + " ...")
    try:
        snapshot_download(repo_id=cfg["hf_repo"], local_dir=str(staging), revision=revision)
        (staging / "dialect.json").write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if _find_onnx(staging) is None:
            raise _fail(
                f"snapshot of {cfg['hf_repo']} holds no single ONNX graph; refusing to install it"
            )
    except BaseException:
        # Never leave a partial staging dir behind (disk hygiene; the scanner
        # could not see it either way).
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
    scanner never sees a half-deleted directory), then delete."""
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

    # add
    if args.all and (args.code or args.file):
        raise _fail("--all takes no code or --file")
    if not args.all and not args.code:
        raise _fail("pass a dialect code or --all")

    if _models_dir().is_dir():
        _cleanup_stale(_models_dir())

    codes = sorted(p.stem for p in DIALECTS_DIR.glob("*.json")) if args.all else [args.code]
    for code in codes:
        cfg = _load_config(code, None if args.all else args.file)
        _validate_or_die(code, cfg)
        install(code, cfg, args.revision)
    print("done. Restart the api service to pick the change up: docker compose restart api")


if __name__ == "__main__":
    main()

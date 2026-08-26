"""
Runtime model registry: which dialects this process serves.

The models directory (MODELS_DIR, a Docker volume in deployment) holds one
subdirectory per dialect: <code>/dialect.json plus the model snapshot. The
lifespan scans it ONCE at startup; operators add or remove a dialect by
changing the volume (scripts/fetch_models.py) and restarting the api service.
No repo change, no image rebuild.

A scan builds a fresh {code: LoadedDialect} dict and rebinds the module global
in one step, so readers only ever see a complete registry. Each dialect loads
inside its own try/except: a broken directory logs a warning and stays out
while its siblings serve. Directories whose names are not plausible dialect
codes (dotdirs like the fetch command's .staging-*/.trash-*, reserved names,
anything outside ascii lowercase) are ignored, so a partial install is never
visible here.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.classifier import LoadedDialect, load_dialect_dir
from app.common.config import MAX_LANG_LEN
from app.common.dialect_schema import RESERVED_CODES


logger = logging.getLogger(__name__)

# Where the model directories live. Deployment mounts the models volume here
# (compose sets MODELS_DIR=/models); local development defaults to ./models.
# `or` so an empty MODELS_DIR= line in an env file means the default too,
# never "scan the current directory".
MODELS_DIR = Path(os.getenv("MODELS_DIR") or "./models")

# A candidate directory name must look like a dialect code: the same charset
# and length bound the schema validator enforces on file stems.
_CODE_RE = re.compile(rf"[a-z]{{1,{MAX_LANG_LEN}}}")

_registry: dict[str, LoadedDialect] = {}


@dataclass(frozen=True)
class ScanResult:
    """What one scan did, for logs and tests."""

    loaded: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


def get(code: str) -> LoadedDialect | None:
    """The loaded dialect for a code, or None. One dict read; request-path hot."""
    return _registry.get(code)


def codes() -> frozenset[str]:
    """The currently loaded dialect codes (drives the 400 detail and /health)."""
    return frozenset(_registry)


def reset() -> None:
    """Drop every loaded dialect (test isolation helper)."""
    _registry.clear()


def _candidate_dirs(models_dir: Path) -> dict[str, Path]:
    """Immediate subdirectories that could hold a dialect, keyed by code.

    Only names that fully match the dialect-code shape count; dotfiles and
    dotdirs (including in-progress fetch staging), plain files, reserved names,
    and anything with an unexpected charset are skipped without logging (the
    fetch command's staging/trash dirs land here by design).
    """
    candidates: dict[str, Path] = {}
    for entry in sorted(models_dir.iterdir()):
        if not entry.is_dir():
            continue
        code = entry.name
        if not _CODE_RE.fullmatch(code) or code in RESERVED_CODES:
            continue
        candidates[code] = entry
    return candidates


def scan_once(models_dir: Path | None = None) -> ScanResult:
    """Build the registry from the models directory, replacing the current one.

    Called synchronously from the lifespan before the server accepts requests.
    Every dialect loads in isolation: one broken directory cannot block its
    siblings and never raises out of the scan. A missing models directory
    yields an empty registry with a warning (the bootstrap flow: bring the
    stack up, run fetch, restart).
    """
    global _registry
    root = models_dir if models_dir is not None else MODELS_DIR
    result = ScanResult()

    if not root.is_dir():
        logger.warning(
            "Models directory %s does not exist; serving zero dialects. "
            "Install models with scripts/fetch_models.py and restart.",
            root,
        )
        _registry = {}
        return result

    fresh: dict[str, LoadedDialect] = {}
    for code, path in _candidate_dirs(root).items():
        try:
            fresh[code] = load_dialect_dir(code, path)
            result.loaded.append(code)
            logger.info("Loaded dialect '%s' from %s", code, path)
        except Exception:
            result.failed.append(code)
            logger.warning(
                "Dialect '%s' failed to load and stays out of service; "
                "its siblings are unaffected (dir: %s)",
                code,
                path,
                exc_info=True,
            )

    _registry = fresh
    if not fresh:
        logger.warning(
            "No dialects loaded from %s; every classify request will get 400. "
            "Install models with scripts/fetch_models.py and restart.",
            root,
        )
    return result

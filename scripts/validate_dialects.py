"""Validate every dialect config file against the shared schema.

Thin CLI over app/common/dialect_schema.py (the single schema definition, also
used by the runtime registry, the fetch command, and the tests). Wired as the
validate-dialects pre-commit hook so a structurally broken (but valid-JSON)
dialect file is rejected at commit time. Stdlib-only imports, no network.

Run from the repo root:  python scripts/validate_dialects.py
"""

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.common.dialect_schema import validate_dialect_config  # noqa: E402


DIALECTS_DIR = ROOT / "app" / "dialects"


def main() -> None:
    if not DIALECTS_DIR.is_dir():
        raise SystemExit(f"Dialects directory not found: {DIALECTS_DIR}")
    paths = sorted(DIALECTS_DIR.glob("*.json"))
    if not paths:
        raise SystemExit(f"No dialect files found in {DIALECTS_DIR}")

    problems: list[str] = []
    configs: dict[str, dict] = {}
    for path in paths:
        try:
            configs[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            problems.append(f"{path.name}: invalid JSON: {e}")
    for code, cfg in configs.items():
        problems.extend(validate_dialect_config(code, cfg))

    if problems:
        print("Dialect config problems:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Validated dialects: {', '.join(sorted(configs))}")


if __name__ == "__main__":
    main()

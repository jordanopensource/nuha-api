#!/usr/bin/env python3
"""Verify requirements.lock is in sync with requirements.txt.

I regenerate the lock into a temp file and compare it against the committed one.
The regenerate is seeded with the current lock, so pip-compile keeps the existing
pins and only changes something when requirements.txt forces it. That way this
fails when someone edits requirements.txt without regenerating the lock, but does
not nag just because a new upstream release landed within an allowed range.

The compare ignores comments and blank lines, because pip-compile rewrites the
header and the `# via` notes while the meaningful content (the index URL, the
pinned versions, and the hashes) is what must match.

Run it the same way locally: `python scripts/check_lock.py` (needs pip-tools).
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = ROOT / "requirements.txt"
LOCK = ROOT / "requirements.lock"


def substantive(path):
    lines = (line.strip() for line in path.read_text().splitlines())
    return [s for s in lines if s and not s.startswith("#")]


def main():
    with tempfile.TemporaryDirectory() as tmp:
        regenerated = Path(tmp) / "requirements.lock"
        shutil.copy(LOCK, regenerated)  # seed with current pins
        subprocess.run(
            [
                "pip-compile",
                "--quiet",
                "--generate-hashes",
                "--allow-unsafe",
                "--output-file",
                str(regenerated),
                str(REQUIREMENTS),
            ],
            check=True,
        )
        if substantive(LOCK) != substantive(regenerated):
            print("ERROR: requirements.lock is out of sync with requirements.txt.")
            print("Regenerate it (the command is in the lock file header) and commit it.")
            return 1
    print("requirements.lock is in sync with requirements.txt.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

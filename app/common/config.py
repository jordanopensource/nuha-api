"""Environment-variable parsing helpers.

Deliberately dependency-free (stdlib only): imported by the dialect schema
validator, which the pre-commit hook runs without the ML stack installed.
"""

import os


# `lang` is a short ISO 639 code or a two-letter alias. This cap is a static,
# generous constant: it bounds how much of an over-long `lang` value the request
# schema sees before rejecting it (as a stripped 422, never echoed); whether the
# code is declared for the dialect is checked against the loaded dialect. The
# schema validator rejects any dialect config whose code/alias exceeds this, and
# the registry only considers model directories named within the same bound.
MAX_LANG_LEN = 16


def parse_bounded_int(name: str, default: int, lo: int, hi: int) -> int:
    """Parse an integer env var, requiring it to fall within [lo, hi] (raises if not).

    An unset or empty value takes the default; an out-of-range or non-numeric value
    is a hard error (it signals a misconfiguration).
    """
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from None
    if value < lo or value > hi:
        raise RuntimeError(f"{name} must be between {lo} and {hi}, got {value}")
    return value

"""Tests for text preprocessing.

Pure functions, no mocking. Every preprocessor is obtained exactly the way the
app builds it -- ``_build_preprocess_fn`` applied to each dialect file's
``preprocessing`` config -- so the tests exercise the configurations that
actually ship and stay correct as dialect files are added or removed. Nothing
here imports a specific preprocessor or names a dialect: the universal contract
is checked against every discovered config, and family-specific behaviour is
keyed off the ``preprocessing.type`` declared in the files (data, not a
hardcoded family).

Note: conftest.py mocks the heavy ML imports and sets DIALECT before these
imports occur.
"""

import pytest

from app.classifier import _build_preprocess_fn
from tests.conftest import _DIALECT_FILES


# --- discovery: everything derives from the shipped dialect files ------------


def _unique_preprocessors():
    """One ``pytest.param(built_fn, prep_config)`` per DISTINCT preprocessing
    config that ships, built the way the app builds it. A new dialect (or family)
    is covered automatically with no edit here."""
    out, seen = [], set()
    for code, cfg in _DIALECT_FILES.items():
        prep = cfg["preprocessing"]
        key = tuple(sorted(prep.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(pytest.param(_build_preprocess_fn(prep), prep, id=code))
    return out


_PREPROCESSORS = _unique_preprocessors()
_TYPES_PRESENT = sorted({cfg["preprocessing"]["type"] for cfg in _DIALECT_FILES.values()})


def _fn_for_type(prep_type):
    """Build the preprocessor for the first shipped dialect declaring this type."""
    for cfg in _DIALECT_FILES.values():
        if cfg["preprocessing"]["type"] == prep_type:
            return _build_preprocess_fn(cfg["preprocessing"])
    return None


def _configs_declaring(flag):
    """Params for every shipped config that exposes ``flag`` (both on and off
    values, whichever ship), or a single explicitly-skipped param if none do."""
    params = [
        pytest.param(_build_preprocess_fn(c["preprocessing"]), c["preprocessing"], id=code)
        for code, c in _DIALECT_FILES.items()
        if flag in c["preprocessing"]
    ]
    return params or [
        pytest.param(
            None, None, marks=pytest.mark.skip(reason=f"no shipped config declares {flag!r}")
        )
    ]


# =============================================================================
# Universal contract -- must hold for every preprocessor, whatever type/flags
# =============================================================================


@pytest.mark.parametrize("preprocess,prep", _PREPROCESSORS)
class TestPreprocessingContract:
    def test_returns_str(self, preprocess, prep):
        assert isinstance(preprocess("مرحبا بالعالم"), str)

    def test_empty_and_whitespace_return_empty(self, preprocess, prep):
        assert preprocess("") == ""
        assert preprocess("   ") == ""

    def test_valid_arabic_kept_and_whitespace_normalized(self, preprocess, prep):
        result = preprocess("  مرحبا    بالعالم  ")
        assert "مرحبا" in result
        assert "  " not in result
        assert result == result.strip()

    def test_over_50_words_rejected_50_accepted(self, preprocess, prep):
        assert preprocess(" ".join(["مرحبا"] * 50)) != ""
        assert preprocess(" ".join(["مرحبا"] * 51)) == ""

    def test_deterministic(self, preprocess, prep):
        text = "مرحبا بالعالم"
        assert preprocess(text) == preprocess(text)

    def test_word_guard_short_circuits_before_expensive_work(self, preprocess, prep):
        """The >50-word guard fires cheaply on a pathological many-token input
        (what an attacker would send to burn CPU) instead of running the full
        pipeline -- this bounds per-request CPU so a huge body can't tie up an
        inference slot."""
        pathological = "a3 " * 20000  # 20k arabizi-shaped tokens, far over 50 words
        # The >50-word guard rejects this pathological many-token input up front,
        # so a huge body can't burn CPU running the full per-token pipeline.
        assert preprocess(pathological) == ""


# =============================================================================
# Family-specific behaviour -- keyed off the discovered preprocessing.type
# =============================================================================
#
# Keys are matched against the `preprocessing.type` values found in the dialect
# files (data, not a hardcoded family list). A type that ships but is missing
# here fails loudly (a new family must state its expected behaviour); a type
# listed here but not shipped simply never runs. Each check is
# (label, input, predicate-on-output).

_TYPE_BEHAVIOURS = {
    "nuha": [
        ("keeps raw emoji beside arabic", "مرحبا 😊", lambda r: "😊" in r and "مرحبا" in r),
        ("emoji-only is invalid", "😊😂🔥", lambda r: r == ""),
        ("drops non-arabic non-emoji chars", "Hello مرحبا World", lambda r: r == "مرحبا"),
        (
            "keeps only arabic in mixed script",
            "أنا I am أحب love الخير good",
            lambda r: "I" not in r and "am" not in r and "أنا" in r and "أحب" in r,
        ),
        (
            "drops digits and punctuation",
            "مرحبا 123 بالعالم!.",
            lambda r: "123" not in r and "!" not in r and "." not in r and "مرحبا" in r,
        ),
    ],
    "safa": [
        (
            "strips http/www urls",
            "مرحبا http://example.com بالعالم",
            lambda r: "http" not in r and "example" not in r,
        ),
        ("strips www urls", "مرحبا www.example.com بالعالم", lambda r: "www" not in r),
        (
            "strips @mentions",
            "مرحبا @username بالعالم",
            lambda r: "@" not in r and "username" not in r,
        ),
        ("strips # but keeps the word", "#مرحبا بالعالم", lambda r: "#" not in r and "مرحبا" in r),
        (
            "strips [[photo]] artifacts",
            "[[photo]] مرحبا بالعالم",
            lambda r: "photo" not in r.lower(),
        ),
        (
            "strips 'photo scraps' artifacts",
            "photo scraps مرحبا بالعالم",
            lambda r: "photo" not in r.lower(),
        ),
        ("demojizes rather than keeping raw emoji", "مرحبا 😊 بالعالم", lambda r: "😊" not in r),
        (
            "collapses 3+ repeated chars to 2",
            "هههههههه مرحبا بالعالم",
            lambda r: "هههههههه" not in r and "هه" in r,
        ),
        (
            "normalizes alef variants to bare alef",
            "إبراهيم أحمد آدم ٱلله",
            lambda r: all(v not in r for v in "إأآٱ") and "ابراهيم" in r,
        ),
        ("strips arabic diacritics", "بِسْمِ اللَّهِ الرَّحمن", lambda r: "ِ" not in r and "ْ" not in r),
        (
            "strips non-arabic non-latin scripts",
            "مرحبا 你好 بالعالم",
            lambda r: "你" not in r and "مرحبا" in r,
        ),
        ("no arabic script is invalid", "Hello World", lambda r: r == ""),
        ("under 2 chars is invalid", "ا", lambda r: r == ""),
        ("non-string input is invalid", None, lambda r: r == ""),
        ("non-string input is invalid (int)", 123, lambda r: r == ""),
        ("non-string input is invalid (list)", [], lambda r: r == ""),
    ],
}


@pytest.mark.parametrize("prep_type", _TYPES_PRESENT)
def test_type_specific_behaviour(prep_type):
    """Each shipped preprocessing type satisfies its declared behaviour table."""
    assert prep_type in _TYPE_BEHAVIOURS, (
        f"preprocessing type {prep_type!r} ships but has no behaviour table in this test"
    )
    preprocess = _fn_for_type(prep_type)
    for label, text, ok in _TYPE_BEHAVIOURS[prep_type]:
        assert ok(preprocess(text)), f"{prep_type}: {label} (got {preprocess(text)!r})"


# =============================================================================
# Optional preprocessing flags -- exercised for whichever configs declare them
# =============================================================================


@pytest.mark.parametrize("preprocess,prep", _configs_declaring("leetspeak"))
def test_leetspeak_flag(preprocess, prep):
    """When on, arabizi tokens (digits + latin letters) decode to Arabic, and
    only mixed digit+letter tokens are touched; when off, they are left alone."""
    if prep["leetspeak"]:
        assert "ع" in preprocess("3rbi مرحبا")
        assert "ح" in preprocess("7abibi مرحبا")
        assert "تش" in preprocess("ch3b مرحبا")
        assert "hello" in preprocess("hello مرحبا بالعالم")  # pure-letter token untouched
        assert "3rbi" not in preprocess("3rbi مرحبا بالعالم")  # mixed token decoded
        assert "ء" not in preprocess("123 مرحبا بالعالم")  # pure-number token not decoded
    else:
        assert "عrbi" not in preprocess("3rbi مرحبا")  # digit not decoded
        assert "حabibi" not in preprocess("7abibi مرحبا")  # digraph not decoded


@pytest.mark.parametrize("preprocess,prep", _configs_declaring("alef_maqsura"))
def test_alef_maqsura_flag(preprocess, prep):
    """When on, ى normalizes to ي; when off, ى is preserved."""
    result = preprocess("على مرحبا")
    if prep["alef_maqsura"]:
        assert "ى" not in result and "علي" in result
    else:
        assert "ى" in result

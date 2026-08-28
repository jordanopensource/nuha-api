"""Tests for the shared dialect-schema validator.

app/common/dialect_schema.py validates every dialect config structurally: at
commit time (the validate-dialects pre-commit hook), at install time (the fetch
command refuses a broken config before downloading), and at load time (the
registry keeps a broken directory out of service). These tests pin that gate:
the real dialect files pass, and each kind of breakage is reported.

The validator is imported through conftest, so this file, test_labels.py, the
registry, and the fetch command all exercise the same single definition.
"""

import copy
import json

import pytest

from app.common.config import MAX_LANG_LEN
from app.common.dialect_schema import RESERVED_CODES
from tests.conftest import ALL_DIALECTS, _dialect_file, validate_dialect_config


validate = validate_dialect_config


def _valid_cfg() -> dict:
    """A known-good config: the first shipped dialect's file."""
    return json.loads(_dialect_file(ALL_DIALECTS[0]).read_text(encoding="utf-8"))


@pytest.mark.parametrize("dialect", ALL_DIALECTS)
def test_shipped_files_are_valid(dialect):
    """Every real dialect file passes the validator (no false positives)."""
    cfg = json.loads(_dialect_file(dialect).read_text(encoding="utf-8"))
    assert validate(dialect, cfg) == []


def test_bad_dialect_code_flagged():
    """The dialect code (the file stem) must be ASCII lowercase letters: uppercase,
    digits, punctuation, whitespace, non-ASCII, or empty are all flagged. It feeds
    the routing path segment, the models-volume directory name, and image tags, so
    this rule is load-bearing."""
    for bad in ("ABC", "a-c", "ab2", "a c", "عرب", ""):
        assert any("a-z" in p for p in validate(bad, _valid_cfg())), bad


def test_good_dialect_code_not_flagged():
    """A well-formed lowercase-ASCII code passes the code-format check."""
    assert not any("a-z" in p for p in validate("abc", _valid_cfg()))


def test_missing_required_field_flagged():
    cfg = _valid_cfg()
    del cfg["labels"]
    problems = validate("x", cfg)
    assert any("missing required field" in p and "labels" in p for p in problems)


def test_malformed_hf_repo_flagged():
    for bad in ("not-a-repo", "a/b/c", "https://hf.co/a/b", "ns/", "/name", "a b/c"):
        cfg = _valid_cfg()
        cfg["hf_repo"] = bad
        assert any("hf_repo" in p for p in validate("x", cfg)), bad


def test_empty_or_nonstring_name_flagged():
    cfg = _valid_cfg()
    cfg["name"] = "   "
    assert any("'name'" in p for p in validate("x", cfg))


def test_label_language_mismatch_flagged():
    """A declared language without labels (or vice versa) is caught."""
    cfg = _valid_cfg()
    cfg["languages"]["fra"] = {"name": "French", "aliases": ["fr"]}  # no labels for fra
    problems = validate("x", cfg)
    assert any("languages" in p and "declared" in p for p in problems)


def test_sub_to_main_dangling_target_flagged():
    cfg = _valid_cfg()
    first_sub = next(iter(cfg["labels"]["sub_to_main"]))
    cfg["labels"]["sub_to_main"][first_sub] = 9999
    assert any("no matching main label" in p for p in validate("x", cfg))


def test_sub_to_main_orphan_sub_flagged():
    """A sub id with no sub_to_main entry is caught."""
    cfg = _valid_cfg()
    # Drop one sub id's mapping so it becomes an orphan.
    victim = sorted(cfg["labels"]["sub_to_main"], key=int)[-1]
    del cfg["labels"]["sub_to_main"][victim]
    assert any("no sub_to_main mapping" in p for p in validate("x", cfg))


def test_non_digit_label_key_flagged():
    cfg = _valid_cfg()
    lang = next(iter(cfg["labels"]["sub"]))
    mapping = cfg["labels"]["sub"][lang]
    mapping["notanumber"] = "x"
    assert any("digit string" in p for p in validate("x", cfg))


def test_empty_label_string_flagged():
    cfg = _valid_cfg()
    lang = next(iter(cfg["labels"]["main"]))
    key = next(iter(cfg["labels"]["main"][lang]))
    cfg["labels"]["main"][lang][key] = "  "
    assert any("non-empty string" in p for p in validate("x", cfg))


def test_valid_config_untouched_has_no_problems():
    """Deep-copying and revalidating a good config still passes (validator is
    non-mutating and deterministic)."""
    cfg = _valid_cfg()
    assert validate("x", copy.deepcopy(cfg)) == []
    assert validate("x", cfg) == []


# =============================================================================
# Reserved dialect codes + a length cap on codes/aliases.
#
# Two checks pin the code grammar: (a) reject a dialect code (stem) OR any
# language code/alias longer than MAX_LANG_LEN (16), and (b) reject the
# reserved dialect codes (they collide with the api service name and the
# image-tag grammar). The constants are imported from the validator's own
# modules, so these tests assert against the values it actually uses.
# =============================================================================


@pytest.mark.parametrize("reserved", sorted(RESERVED_CODES))
def test_reserved_dialect_code_flagged(reserved):
    """A reserved code (even with an otherwise-good config) is flagged with a
    reserved-code message; those stems collide with the api service name and
    the image-tag grammar."""
    problems = validate(reserved, _valid_cfg())
    assert any("reserved" in p.lower() for p in problems), (reserved, problems)


def test_over_long_dialect_code_flagged():
    """A dialect code longer than MAX_LANG_LEN is flagged (it feeds the routing
    path segment, the models-volume directory name, and image tags)."""
    long_code = "a" * (MAX_LANG_LEN + 1)  # 17 chars, otherwise a clean a-z code
    problems = validate(long_code, _valid_cfg())
    assert any(str(MAX_LANG_LEN) in p for p in problems), problems


def test_over_long_language_alias_flagged():
    """A language alias longer than MAX_LANG_LEN is flagged (a hypothetical
    >16-char code could never be selected: the request schema caps lang first)."""
    cfg = _valid_cfg()
    lang = next(iter(cfg["languages"]))
    over_long_alias = "a" * (MAX_LANG_LEN + 1)  # 17 chars
    cfg["languages"][lang].setdefault("aliases", []).append(over_long_alias)
    problems = validate("x", cfg)
    assert any("alias" in p and str(MAX_LANG_LEN) in p for p in problems), problems


def test_over_long_language_code_flagged():
    """A canonical language code longer than MAX_LANG_LEN is flagged too."""
    cfg = _valid_cfg()
    long_lang = "a" * (MAX_LANG_LEN + 1)
    # Give the over-long language a full, otherwise-valid label set so it is the
    # length that trips the validator, not a missing-labels problem.
    cfg["languages"][long_lang] = {"name": "Overlong", "aliases": []}
    for cat in ("sub", "main"):
        any_lang = next(iter(cfg["labels"][cat]))
        cfg["labels"][cat][long_lang] = dict(cfg["labels"][cat][any_lang])
    problems = validate("x", cfg)
    assert any("language code" in p and str(MAX_LANG_LEN) in p for p in problems), problems

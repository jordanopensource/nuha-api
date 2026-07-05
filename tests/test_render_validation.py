"""Tests for the render-config commit-time dialect validator.

scripts/render_config.py validates every dialect file structurally before it
regenerates compose/nginx/CI, so the render-config pre-commit hook rejects a
broken (but valid-JSON) dialect file at commit time instead of deferring to CI.
These tests pin that gate: the real dialect files pass, and each kind of
breakage is reported.

The validator is loaded once in conftest (by path, since scripts/ is not an
importable package) and shared, so both this file and test_labels.py exercise
the same single definition.
"""

import copy
import json

import pytest

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
    Docker service names, image tags, and the nginx path segment, so this rule is
    load-bearing."""
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


def test_bad_replicas_flagged():
    for bad in (0, -1, True, "2"):
        cfg = _valid_cfg()
        cfg["replicas"] = bad
        assert any("replicas" in p for p in validate("x", cfg)), bad


def test_bad_mem_limit_flagged():
    """A non-string or non-Docker-size mem_limit is caught at commit time rather
    than deferred to a failed `docker compose up`."""
    for bad in ("4gigs", "abc", "-4g", "", "g", 4):
        cfg = _valid_cfg()
        cfg["mem_limit"] = bad
        assert any("mem_limit" in p for p in validate("x", cfg)), bad


def test_good_mem_limit_not_flagged():
    """Valid Docker memory sizes pass: integer or decimal, an optional space, and
    single- or two-letter (IEC) units in any case."""
    for good in ("4g", "512m", "1073741824", "2G", "1gb", "1.5g", "4 g", "1024kb"):
        cfg = _valid_cfg()
        cfg["mem_limit"] = good
        assert not any("mem_limit" in p for p in validate("x", cfg)), good


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

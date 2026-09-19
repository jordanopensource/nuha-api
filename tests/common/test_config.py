"""Tests for app.common.config: dependency-free env parsing.

The module stays stdlib-only so the dialect-schema validator (and its
pre-commit hook) can import it with nothing installed. These tests exercise
the bounded-int parser every knob relies on plus the MAX_LANG_LEN constant
that caps the `lang` field.
"""

import pytest

from app.common.config import MAX_LANG_LEN, parse_bounded_int


class TestParseBoundedInt:
    def test_uses_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("SOME_KNOB", raising=False)
        assert parse_bounded_int("SOME_KNOB", 7, 0, 100) == 7

    def test_uses_default_when_blank(self, monkeypatch):
        """A SET-but-empty var takes the default rather than crashing (so
        blanking a knob in .env means the default, uniformly)."""
        monkeypatch.setenv("SOME_KNOB", "")
        assert parse_bounded_int("SOME_KNOB", 7, 0, 100) == 7

    def test_reads_valid_env(self, monkeypatch):
        monkeypatch.setenv("SOME_KNOB", "42")
        assert parse_bounded_int("SOME_KNOB", 7, 0, 100) == 42

    def test_accepts_boundaries(self, monkeypatch):
        monkeypatch.setenv("SOME_KNOB", "0")
        assert parse_bounded_int("SOME_KNOB", 7, 0, 100) == 0
        monkeypatch.setenv("SOME_KNOB", "100")
        assert parse_bounded_int("SOME_KNOB", 7, 0, 100) == 100

    def test_below_range_raises(self, monkeypatch):
        monkeypatch.setenv("SOME_KNOB", "-1")
        with pytest.raises(RuntimeError, match="between 0 and 100"):
            parse_bounded_int("SOME_KNOB", 7, 0, 100)

    def test_above_range_raises(self, monkeypatch):
        monkeypatch.setenv("SOME_KNOB", "101")
        with pytest.raises(RuntimeError, match="between 0 and 100"):
            parse_bounded_int("SOME_KNOB", 7, 0, 100)

    def test_non_integer_raises(self, monkeypatch):
        monkeypatch.setenv("SOME_KNOB", "not-an-int")
        with pytest.raises(RuntimeError, match="must be an integer"):
            parse_bounded_int("SOME_KNOB", 7, 0, 100)


class TestMaxLangLen:
    def test_is_a_generous_positive_constant(self):
        """MAX_LANG_LEN is a static, dialect-agnostic cap (16 in 2.0). It only
        bounds how much of an over-long `lang` can be seen before the schema
        rejects it; whether the code is declared for the dialect is checked by
        the app's own lang resolution."""
        assert isinstance(MAX_LANG_LEN, int)
        assert MAX_LANG_LEN >= 3  # room for the longest ISO 639-3 code

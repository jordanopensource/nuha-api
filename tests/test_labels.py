"""Integrity tests for the per-dialect config files.

The dialect-file SCHEMA (required fields, well-formed hf_repo, label/language
consistency, sub_to_main integrity, digit keys, ...) is defined in ONE place:
`validate_dialect_config` in app/common/dialect_schema.py (the commit-time,
install-time, and load-time gate). These tests assert the shipped files pass
that validator rather than re-encoding the rules a second time;
`test_dialect_schema.py` proves the validator's logic itself. What remains here
is what the validator deliberately does NOT cover: that the APP can actually
parse and load every file (_parse_dialect, _build_preprocess_fn, the
preprocessing registry) and the directory-level invariants. Everything
parametrizes over whatever files exist, so it holds for one dialect file or a
thousand.
"""

import json

import pytest

from tests.conftest import ALL_DIALECTS, DIALECTS_DIR, _dialect_file, validate_dialect_config


# =============================================================================
# Schema: assert through the single validator (no parallel re-encoding)
# =============================================================================


class TestDialectFilesValid:
    """Every shipped dialect file satisfies the schema validator."""

    @pytest.mark.parametrize("dialect", ALL_DIALECTS)
    def test_passes_schema_validator(self, dialect):
        """The file has no structural problems per validate_dialect_config,
        the same check the validate-dialects pre-commit hook runs. This covers
        required fields and types, hf_repo shape, languages, label/language
        consistency, digit keys, non-empty label strings, and sub_to_main
        integrity, all from one definition."""
        cfg = json.loads(_dialect_file(dialect).read_text(encoding="utf-8"))
        problems = validate_dialect_config(dialect, cfg)
        assert problems == [], f"Dialect '{dialect}' failed schema validation: {problems}"


# =============================================================================
# App parsing: what the validator does not cover (int-key conversion, shape)
# =============================================================================


class TestParseDialect:
    """_parse_dialect() converts any valid dialect file the way the app needs."""

    @pytest.fixture(autouse=True)
    def _import_parser(self):
        from app.classifier import _parse_dialect

        self._parse_dialect = lambda code: _parse_dialect(
            json.loads(_dialect_file(code).read_text(encoding="utf-8"))
        )

    @pytest.mark.parametrize("dialect", ALL_DIALECTS)
    def test_produces_int_keys(self, dialect):
        """JSON string keys become int keys (the schema validator checks they
        are digit strings; this checks the app actually converts them)."""
        parsed = self._parse_dialect(dialect)
        for k in parsed["sub_to_main"]:
            assert isinstance(k, int), f"Dialect '{dialect}', sub_to_main: key {k!r} is not int"
        for group in ("sub_labels", "main_labels"):
            for lang, mapping in parsed[group].items():
                for k in mapping:
                    assert isinstance(k, int), (
                        f"Dialect '{dialect}', {group}[{lang}]: key {k!r} is not int"
                    )

    @pytest.mark.parametrize("dialect", ALL_DIALECTS)
    def test_contains_expected_keys(self, dialect):
        """_parse_dialect() returns exactly the keys DialectConfig expects."""
        parsed = self._parse_dialect(dialect)
        assert set(parsed.keys()) == {"sub_to_main", "sub_labels", "main_labels"}

    @pytest.mark.parametrize("dialect", ALL_DIALECTS)
    def test_parsed_languages_match_file(self, dialect):
        """The parsed label groups carry exactly the languages the file's labels
        block carries: nothing dropped, nothing invented."""
        parsed = self._parse_dialect(dialect)
        raw = json.loads(_dialect_file(dialect).read_text(encoding="utf-8"))["labels"]
        assert set(parsed["sub_labels"]) == set(raw["sub"])
        assert set(parsed["main_labels"]) == set(raw["main"])


# =============================================================================
# App integration: the file is not just well-formed, it is loadable/servable
# =============================================================================


class TestDialectAppIntegration:
    """Beyond the schema: the app can build the preprocessor, the declared
    default language is serviceable, and the whole loader accepts every file."""

    @pytest.fixture(autouse=True)
    def _load(self, labels_data):
        self.labels = labels_data

    @pytest.mark.parametrize("dialect", ALL_DIALECTS)
    def test_preprocessing_config_is_usable(self, dialect):
        """The preprocessing block builds into a working callable: the type is
        one the classifier's registry knows AND every other key is accepted as
        a kwarg (a stray/misspelled option surfaces here, not at first request).
        This is the app-side check the schema validator deliberately omits to
        stay decoupled from the preprocessing registry."""
        from app.classifier import _build_preprocess_fn

        cfg = json.loads(_dialect_file(dialect).read_text(encoding="utf-8"))
        fn = _build_preprocess_fn(cfg["preprocessing"])
        assert callable(fn)
        assert isinstance(fn("مرحبا بالعالم"), str)

    @pytest.mark.parametrize("dialect", ALL_DIALECTS)
    def test_default_language_is_serviceable(self, dialect):
        """The app's default lang (the first declared language alphabetically,
        LoadedDialect.default_language) has both sub and main labels, so a
        request that omits lang always succeeds; no dialect must declare a
        specific language."""
        cfg = json.loads(_dialect_file(dialect).read_text(encoding="utf-8"))
        default_lang = sorted(cfg["languages"])[0]
        entry = self.labels[dialect]
        assert entry["sub"].get(default_lang) and entry["main"].get(default_lang), (
            f"Dialect '{dialect}': default lang '{default_lang}' is missing labels"
        )


# =============================================================================
# Dialects directory: invariants that hold no matter how many files exist
# =============================================================================


class TestDialectsDirectory:
    """Whole-directory invariants: the app requires at least one dialect, every
    *.json must parse, each stem must be a clean dialect code, and the app's own
    loader must accept every file. These hold for one file or a thousand,
    whoever authored them."""

    def test_directory_has_at_least_one_dialect(self):
        """The app raises at import if app/dialects/ is empty; guard that here."""
        files = list(DIALECTS_DIR.glob("*.json"))
        assert files, "no app/dialects/*.json found; the app cannot start"

    def test_every_json_file_parses_to_an_object(self):
        """Every *.json in the directory is valid JSON and a dict (a broken file
        would keep its own dialect out of the registry)."""
        for path in sorted(DIALECTS_DIR.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                raise AssertionError(f"{path.name} is not valid JSON: {e}") from e
            assert isinstance(data, dict), f"{path.name} must contain a JSON object"

    def test_discovered_dialects_match_directory(self):
        """conftest's ALL_DIALECTS is exactly the set of *.json stems, and each
        stem is a plausible dialect code (lowercase letters, no path junk) that
        routing, the models volume, and the fetch command can all use verbatim."""
        stems = {p.stem for p in DIALECTS_DIR.glob("*.json")}
        assert set(ALL_DIALECTS) == stems
        for code in stems:
            assert code and code.isascii() and code.islower() and code.isalpha(), (
                f"dialect code '{code}' is not a clean lowercase ASCII identifier"
            )

    def test_app_loads_every_dialect(self):
        """The app's own config path accepts every file on disk: the ground
        truth that a passing suite implies the startup scan loads each dialect,
        whoever added the file. (Sibling isolation, the "a broken file fails
        only its own dialect" property, is asserted at scan level in
        test_registry.py.) Mirrors the load path pieces the registry runs per
        directory (_parse_dialect + _build_preprocess_fn)."""
        from app.classifier import DialectConfig, _build_preprocess_fn, _parse_dialect

        for code in ALL_DIALECTS:
            cfg = json.loads(_dialect_file(code).read_text(encoding="utf-8"))
            DialectConfig(
                name=cfg["name"],
                preprocess_fn=_build_preprocess_fn(cfg["preprocessing"]),
                **_parse_dialect(cfg),
            )  # must construct without raising

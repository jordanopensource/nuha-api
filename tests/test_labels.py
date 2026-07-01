"""Tests for label data integrity.

These tests validate the structure and consistency of the per-dialect config
files (app/dialects/<code>.json) without any model dependency. They catch data
corruption, missing mappings, and taxonomy violations. The labels_data/
dialects_data fixtures reconstruct the legacy flat shapes from the per-dialect
files.
"""

import json

import pytest

from tests.conftest import ALL_DIALECTS, DIALECTS_DIR, _dialect_file


# =============================================================================
# label data structure tests
# =============================================================================


class TestLabelsJsonStructure:
    """Verify the structural integrity of each dialect's label data."""

    @pytest.fixture(autouse=True)
    def _load_labels(self, labels_data):
        self.labels = labels_data

    @pytest.mark.parametrize("dialect", ALL_DIALECTS)
    def test_dialect_file_parses_successfully(self, dialect):
        """Each dialect file is valid JSON with a labels block."""
        data = json.loads(_dialect_file(dialect).read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert "labels" in data

    @pytest.mark.parametrize("dialect", ALL_DIALECTS)
    def test_dialect_present(self, dialect):
        """All three dialects have label data."""
        assert dialect in self.labels, f"Dialect '{dialect}' missing label data"

    @pytest.mark.parametrize("dialect", ALL_DIALECTS)
    def test_sub_to_main_targets_exist_in_main(self, dialect):
        """Every sub_to_main target exists in main labels."""
        entry = self.labels[dialect]
        main_ids = {int(k) for k in entry["main"]["ara"]}
        for sub_id, main_id in entry["sub_to_main"].items():
            assert main_id in main_ids, (
                f"Dialect '{dialect}': sub_to_main maps sub {sub_id} to main {main_id}, "
                f"but main {main_id} does not exist in main labels"
            )

    @pytest.mark.parametrize("dialect", ALL_DIALECTS)
    def test_sub_to_main_sources_exist_in_sub(self, dialect):
        """Every sub_to_main key exists in sub labels."""
        entry = self.labels[dialect]
        sub_ids = {int(k) for k in entry["sub"]["ara"]}
        for sub_id_str in entry["sub_to_main"]:
            sub_id = int(sub_id_str)
            assert sub_id in sub_ids, (
                f"Dialect '{dialect}': sub_to_main has key {sub_id}, "
                f"but it does not exist in sub labels"
            )

    @pytest.mark.parametrize("dialect", ALL_DIALECTS)
    def test_every_sub_has_mapping(self, dialect):
        """Every sub-class ID has a mapping in sub_to_main (no orphans)."""
        entry = self.labels[dialect]
        sub_ids = {int(k) for k in entry["sub"]["ara"]}
        mapped_ids = {int(k) for k in entry["sub_to_main"]}
        orphans = sub_ids - mapped_ids
        assert not orphans, f"Dialect '{dialect}': sub IDs {orphans} have no mapping in sub_to_main"

    @pytest.mark.parametrize("dialect", ALL_DIALECTS)
    def test_ar_en_sub_labels_same_keys(self, dialect):
        """Arabic and English sub label dicts have the same keys."""
        entry = self.labels[dialect]
        ar_keys = set(entry["sub"]["ara"].keys())
        en_keys = set(entry["sub"]["eng"].keys())
        assert ar_keys == en_keys, (
            f"Dialect '{dialect}': ar sub keys {ar_keys} != en sub keys {en_keys}"
        )

    @pytest.mark.parametrize("dialect", ALL_DIALECTS)
    def test_ar_en_main_labels_same_keys(self, dialect):
        """Arabic and English main label dicts have the same keys."""
        entry = self.labels[dialect]
        ar_keys = set(entry["main"]["ara"].keys())
        en_keys = set(entry["main"]["eng"].keys())
        assert ar_keys == en_keys, (
            f"Dialect '{dialect}': ar main keys {ar_keys} != en main keys {en_keys}"
        )

    @pytest.mark.parametrize("dialect", ALL_DIALECTS)
    def test_no_empty_label_strings(self, dialect):
        """No label string is empty or whitespace-only."""
        entry = self.labels[dialect]
        for category in ("sub", "main"):
            for lang, labels in entry[category].items():
                for key, value in labels.items():
                    assert value.strip(), (
                        f"Dialect '{dialect}', {category}/{lang}, key {key}: "
                        f"label is empty or whitespace"
                    )

    @pytest.mark.parametrize("dialect", ALL_DIALECTS)
    def test_keys_are_valid_non_negative_integers(self, dialect):
        """All label keys and sub_to_main keys are valid non-negative integer strings."""
        entry = self.labels[dialect]
        for category in ("sub", "main"):
            for lang, labels in entry[category].items():
                for key in labels:
                    assert key.isdigit(), (
                        f"Dialect '{dialect}', {category}/{lang}: "
                        f"key '{key}' is not a valid non-negative integer"
                    )
        for key in entry["sub_to_main"]:
            assert key.isdigit(), (
                f"Dialect '{dialect}': sub_to_main key '{key}' is not a valid non-negative integer"
            )


# =============================================================================
# Kurdish-specific labels
# =============================================================================


class TestKurdishLabels:
    """Verify Kurdish-specific label requirements."""

    @pytest.fixture(autouse=True)
    def _load_labels(self, labels_data):
        self.labels = labels_data

    @pytest.mark.parametrize("dialect", ["acm", "ckb"])
    def test_safa_dialect_has_kurdish_sub_labels(self, dialect):
        """The SAFA dialects (acm, ckb) have 'ckb' sub labels."""
        assert "ckb" in self.labels[dialect]["sub"]
        assert len(self.labels[dialect]["sub"]["ckb"]) > 0

    @pytest.mark.parametrize("dialect", ["acm", "ckb"])
    def test_safa_dialect_has_kurdish_main_labels(self, dialect):
        """The SAFA dialects (acm, ckb) have 'ckb' main labels."""
        assert "ckb" in self.labels[dialect]["main"]
        assert len(self.labels[dialect]["main"]["ckb"]) > 0

    @pytest.mark.parametrize("dialect", ["acm", "ckb"])
    def test_safa_kurdish_labels_same_keys_as_ar(self, dialect):
        """Kurdish sub/main labels have the same keys as the Arabic labels."""
        entry = self.labels[dialect]
        assert set(entry["sub"]["ckb"].keys()) == set(entry["sub"]["ara"].keys())
        assert set(entry["main"]["ckb"].keys()) == set(entry["main"]["ara"].keys())

    def test_arz_lacks_kurdish_labels(self):
        """arz, the only non-SAFA dialect, has no 'ckb' labels (or empty dicts)."""
        entry = self.labels["arz"]
        assert len(entry["sub"].get("ckb", {})) == 0, "arz should not have Kurdish sub labels"
        assert len(entry["main"].get("ckb", {})) == 0, "arz should not have Kurdish main labels"


# =============================================================================
# SAFA taxonomy structure (Iraqi and Kurdish share the same structure)
# =============================================================================


class TestSafaTaxonomy:
    """Verify that Iraqi and Kurdish share the SAFA taxonomy structure."""

    @pytest.fixture(autouse=True)
    def _load_labels(self, labels_data):
        self.labels = labels_data

    def test_safa_same_number_of_subs(self):
        """Iraqi and Kurdish have the same number of sub-classes."""
        acm_subs = len(self.labels["acm"]["sub"]["ara"])
        ckb_subs = len(self.labels["ckb"]["sub"]["ara"])
        assert acm_subs == ckb_subs

    def test_safa_same_number_of_mains(self):
        """Iraqi and Kurdish have the same number of main classes."""
        acm_mains = len(self.labels["acm"]["main"]["ara"])
        ckb_mains = len(self.labels["ckb"]["main"]["ara"])
        assert acm_mains == ckb_mains

    def test_safa_same_sub_to_main_mapping(self):
        """Iraqi and Kurdish share the same sub_to_main mapping."""
        assert self.labels["acm"]["sub_to_main"] == self.labels["ckb"]["sub_to_main"]

    def test_safa_share_kurdish_labels(self):
        """Iraqi and Kurdish carry the identical Kurdish (ckb) label set."""
        assert self.labels["acm"]["sub"]["ckb"] == self.labels["ckb"]["sub"]["ckb"]
        assert self.labels["acm"]["main"]["ckb"] == self.labels["ckb"]["main"]["ckb"]

    def test_arz_different_structure(self):
        """Egyptian has a different structure (different sub/main counts)."""
        arz_subs = len(self.labels["arz"]["sub"]["ara"])
        acm_subs = len(self.labels["acm"]["sub"]["ara"])
        arz_mains = len(self.labels["arz"]["main"]["ara"])
        acm_mains = len(self.labels["acm"]["main"]["ara"])
        assert arz_subs != acm_subs
        assert arz_mains != acm_mains


# =============================================================================
# _parse_dialect() function
# =============================================================================


class TestParseDialect:
    """Verify that _parse_dialect() produces the correct output format."""

    @pytest.fixture(autouse=True)
    def _import_parser(self):
        from app.classifier import _parse_dialect

        self._parse_dialect = _parse_dialect

    @pytest.mark.parametrize("dialect", ALL_DIALECTS)
    def test_produces_int_keys(self, dialect):
        """_parse_dialect() converts JSON string keys to int keys."""
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
        """_parse_dialect() returns dict with all expected keys."""
        parsed = self._parse_dialect(dialect)
        assert set(parsed.keys()) == {"sub_to_main", "sub_labels", "main_labels"}

    @pytest.mark.parametrize("dialect", ["acm", "ckb"])
    def test_safa_has_nonempty_kurdish_labels(self, dialect):
        """For the SAFA dialects (acm, ckb), Kurdish sub/main labels are present and non-empty."""
        parsed = self._parse_dialect(dialect)
        assert len(parsed["sub_labels"]["ckb"]) > 0
        assert len(parsed["main_labels"]["ckb"]) > 0

    def test_arz_has_no_kurdish_labels(self):
        """For arz (the only non-SAFA dialect), no Kurdish label group is present."""
        parsed = self._parse_dialect("arz")
        assert "ckb" not in parsed["sub_labels"]
        assert "ckb" not in parsed["main_labels"]


# =============================================================================
# dialect file structure tests
# =============================================================================


class TestDialectsJsonStructure:
    """Verify the structural integrity of the per-dialect config files."""

    @pytest.fixture(autouse=True)
    def _load(self, labels_data, dialects_data):
        self.labels = labels_data
        self.dialects = dialects_data

    def test_one_file_per_dialect(self):
        """The dialects directory holds exactly one file per known dialect."""
        files = {p.stem for p in DIALECTS_DIR.glob("*.json")}
        assert files == set(ALL_DIALECTS)

    def test_languages_declared_with_names_and_aliases(self):
        """Each dialect declares its languages by canonical code with a name and aliases."""
        for code in ALL_DIALECTS:
            languages = json.loads(_dialect_file(code).read_text(encoding="utf-8"))["languages"]
            assert isinstance(languages, dict)
            assert {"ara", "eng"}.issubset(languages.keys())
            for meta in languages.values():
                assert "name" in meta
                assert isinstance(meta.get("aliases", []), list)

    def test_language_aliases(self):
        """The two-letter ISO 639-1 codes are declared as aliases of the canonical codes."""
        arz = json.loads(_dialect_file("arz").read_text(encoding="utf-8"))["languages"]
        assert "ar" in arz["ara"]["aliases"]
        assert "en" in arz["eng"]["aliases"]
        ckb = json.loads(_dialect_file("ckb").read_text(encoding="utf-8"))["languages"]
        assert "ku" in ckb["ckb"]["aliases"]

    def test_all_dialects_present(self):
        """All expected dialects are loaded."""
        for dialect in ALL_DIALECTS:
            assert dialect in self.dialects["dialects"]

    @pytest.mark.parametrize("dialect", ALL_DIALECTS)
    def test_dialect_has_labels_entry(self, dialect):
        """Every dialect file carries its own labels block."""
        assert dialect in self.labels

    @pytest.mark.parametrize("dialect", ALL_DIALECTS)
    def test_languages_have_label_entries(self, dialect):
        """All declared languages have label entries in the dialect file."""
        cfg = self.dialects["dialects"][dialect]
        for lang in cfg["languages"]:
            entry = self.labels[dialect]
            assert lang in entry["sub"], (
                f"Dialect '{dialect}': language '{lang}' declared but no sub labels in its file"
            )
            assert lang in entry["main"], (
                f"Dialect '{dialect}': language '{lang}' declared but no main labels in its file"
            )

    @pytest.mark.parametrize("dialect", ALL_DIALECTS)
    def test_preprocessing_type_valid(self, dialect):
        """Preprocessing type is a recognized type."""
        cfg = self.dialects["dialects"][dialect]
        valid_types = {"nuha", "safa"}
        prep_type = cfg["preprocessing"]["type"]
        assert prep_type in valid_types, (
            f"Dialect '{dialect}': preprocessing type '{prep_type}' "
            f"not in valid types {valid_types}"
        )

    @pytest.mark.parametrize("dialect", ALL_DIALECTS)
    def test_dialect_has_required_fields(self, dialect):
        """Each dialect config has all required fields."""
        cfg = self.dialects["dialects"][dialect]
        required = {"name", "hf_repo", "languages", "preprocessing"}
        assert required.issubset(set(cfg.keys())), (
            f"Dialect '{dialect}' missing fields: {required - set(cfg.keys())}"
        )

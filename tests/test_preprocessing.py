"""Tests for text preprocessing functions.

These are the highest-value tests: pure functions with zero external
dependencies and no mocking needed. They catch real regressions in the
dialect-specific text cleaning pipelines.

Note: conftest.py handles mocking heavy ML imports and setting DIALECT
before these imports occur.
"""

import pytest

from app.classifier import _preprocess_nuha, _preprocess_safa


# =============================================================================
# Egyptian Arabic preprocessing
# =============================================================================


class TestPreprocessNuha:
    """Tests for _preprocess_nuha -- Egyptian Arabic preprocessing."""

    def test_arabic_text_kept(self):
        """Pure Arabic text passes through unchanged (modulo whitespace normalization)."""
        result = _preprocess_nuha("مرحبا بالعالم")
        assert result == "مرحبا بالعالم"

    def test_non_arabic_stripped(self):
        """Non-Arabic, non-emoji characters are removed."""
        result = _preprocess_nuha("Hello مرحبا World")
        assert result == "مرحبا"

    def test_emojis_preserved_with_arabic(self):
        """Emojis are kept alongside Arabic text."""
        result = _preprocess_nuha("مرحبا 😊")
        assert "مرحبا" in result
        assert "😊" in result

    def test_emoji_only_returns_empty(self):
        """Text consisting entirely of emojis is considered invalid."""
        result = _preprocess_nuha("😊😂🔥")
        assert result == ""

    def test_over_50_words_returns_empty(self):
        """Text exceeding 50 words returns empty (invalid)."""
        text = " ".join(["كلمة"] * 51)
        result = _preprocess_nuha(text)
        assert result == ""

    def test_exactly_50_words_accepted(self):
        """Text with exactly 50 words is valid."""
        text = " ".join(["كلمة"] * 50)
        result = _preprocess_nuha(text)
        assert result != ""
        assert result.count("كلمة") == 50

    def test_empty_input_returns_empty(self):
        """Empty string input returns empty."""
        assert _preprocess_nuha("") == ""

    def test_whitespace_only_returns_empty(self):
        """Whitespace-only input returns empty."""
        assert _preprocess_nuha("   ") == ""

    def test_mixed_arabic_latin_only_arabic_survives(self):
        """Mixed Arabic + Latin: only Arabic characters survive."""
        result = _preprocess_nuha("أنا I am أحب love الخير good")
        assert "I" not in result
        assert "am" not in result
        assert "أنا" in result
        assert "أحب" in result

    def test_multiple_spaces_collapsed(self):
        """Multiple consecutive spaces are collapsed to single space."""
        result = _preprocess_nuha("مرحبا    بالعالم")
        assert "  " not in result
        assert "مرحبا" in result

    def test_leading_trailing_whitespace_stripped(self):
        """Leading and trailing whitespace is stripped."""
        result = _preprocess_nuha("  مرحبا  ")
        assert not result.startswith(" ")
        assert not result.endswith(" ")
        assert result == "مرحبا"

    def test_arabic_with_numbers_drops_numbers(self):
        """Numbers (ASCII digits) are not in the Arabic Unicode range and get dropped."""
        result = _preprocess_nuha("مرحبا 123 بالعالم")
        assert "123" not in result
        assert "مرحبا" in result

    def test_49_words_accepted(self):
        """Text under 50 words is accepted."""
        text = " ".join(["كلمة"] * 49)
        result = _preprocess_nuha(text)
        assert result != ""

    def test_single_arabic_word(self):
        """Single Arabic word is valid."""
        assert _preprocess_nuha("مرحبا") == "مرحبا"

    def test_punctuation_stripped(self):
        """Punctuation is not in the Arabic U+0600-U+06FF range and is dropped."""
        result = _preprocess_nuha("مرحبا! بالعالم.")
        assert "!" not in result
        assert "." not in result


# =============================================================================
# Iraqi Arabic preprocessing
# =============================================================================


class TestPreprocessIraqi:
    """Tests for Iraqi Arabic preprocessing via _preprocess_safa with Iraqi flags."""

    def _preprocess_acm(self, text: str) -> str:
        """Iraqi convenience wrapper: leetspeak=True, alef_maqsura=True."""
        return _preprocess_safa(text, leetspeak=True, alef_maqsura=True)

    def test_urls_stripped(self):
        """URLs are removed from text."""
        result = self._preprocess_acm("مرحبا http://example.com بالعالم")
        assert "http" not in result
        assert "example" not in result
        assert "مرحبا" in result

    def test_www_urls_stripped(self):
        """www-prefixed URLs are also removed."""
        result = self._preprocess_acm("مرحبا www.example.com بالعالم")
        assert "www" not in result
        assert "مرحبا" in result

    def test_mentions_stripped(self):
        """@mentions are removed."""
        result = self._preprocess_acm("مرحبا @username بالعالم")
        assert "@" not in result
        assert "username" not in result

    def test_hashtag_symbol_stripped_word_kept(self):
        """# is stripped but the hashtag word remains."""
        result = self._preprocess_acm("#مرحبا بالعالم")
        assert "#" not in result
        assert "مرحبا" in result

    def test_photo_tag_stripped(self):
        """[[photo]] placeholder is removed."""
        result = self._preprocess_acm("[[photo]] مرحبا بالعالم")
        assert "photo" not in result
        assert "مرحبا" in result

    def test_photo_scraps_stripped(self):
        """'photo scraps' text is removed."""
        result = self._preprocess_acm("photo scraps مرحبا بالعالم")
        assert "photo" not in result.lower()
        assert "مرحبا" in result

    def test_leetspeak_3_to_ain(self):
        """Leetspeak: 3 decodes to ain."""
        result = self._preprocess_acm("3rbi مرحبا")
        assert "ع" in result

    def test_leetspeak_7_to_ha(self):
        """Leetspeak: 7 decodes to ha."""
        result = self._preprocess_acm("7abibi مرحبا")
        assert "ح" in result

    def test_leetspeak_ch_to_tasheen(self):
        """Leetspeak: ch decodes to tasheen."""
        result = self._preprocess_acm("ch3b مرحبا")
        assert "تش" in result

    def test_leetspeak_only_mixed_digit_letter_tokens(self):
        """Leetspeak decoding is only applied to tokens with both digits AND letters."""
        # Pure letter token: not decoded
        result_letters = self._preprocess_acm("hello مرحبا بالعالم")
        assert "hello" in result_letters

        # Token with both: decoded
        result_mixed = self._preprocess_acm("3rbi مرحبا بالعالم")
        assert "3rbi" not in result_mixed

    def test_pure_number_tokens_not_decoded(self):
        """Pure number tokens are NOT subjected to leetspeak decoding."""
        # "123" has digits but no letters -> not decoded
        # Then the non-Arabic/non-Latin filter strips digits anyway
        result = self._preprocess_acm("123 مرحبا بالعالم")
        # The number should not produce Arabic letter substitutions
        # like ء (for 2), ع (for 3), etc.
        assert "مرحبا" in result

    def test_emoji_demojized(self):
        """Emojis are converted to text descriptions."""
        result = self._preprocess_acm("مرحبا 😊 بالعالم")
        assert "😊" not in result

    def test_repeated_chars_collapsed(self):
        """Repeated characters (3+) are collapsed to 2."""
        result = self._preprocess_acm("هههههههه مرحبا بالعالم")
        assert "هههههههه" not in result
        assert "هه" in result

    def test_alef_normalization(self):
        """Alef variants are normalized to bare alef."""
        result = self._preprocess_acm("إبراهيم أحمد آدم ٱلله")
        assert "إ" not in result
        assert "أ" not in result
        assert "آ" not in result
        assert "ٱ" not in result
        assert "ابراهيم" in result

    def test_alef_maqsura_normalized(self):
        """Alef maqsura is normalized to ya in Iraqi."""
        result = self._preprocess_acm("على مرحبا")
        assert "ى" not in result
        assert "علي" in result

    def test_diacritics_stripped(self):
        """Arabic diacritics (tashkeel) are removed."""
        result = self._preprocess_acm("بِسْمِ اللَّهِ الرَّحمن")
        assert "\u0650" not in result  # kasra
        assert "\u0652" not in result  # sukun

    def test_non_arabic_non_latin_stripped(self):
        """Characters outside Arabic/Latin ranges are stripped."""
        result = self._preprocess_acm("مرحبا 你好 بالعالم")
        assert "你" not in result
        assert "好" not in result
        assert "مرحبا" in result

    def test_no_arabic_returns_empty(self):
        """Text with no Arabic script returns empty (invalid)."""
        result = self._preprocess_acm("Hello World")
        assert result == ""

    def test_text_shorter_than_2_chars_returns_empty(self):
        """Text resulting in fewer than 2 chars is invalid."""
        result = self._preprocess_acm("ا")
        assert result == ""

    def test_non_string_input_returns_empty(self):
        """Non-string input is handled gracefully."""
        assert _preprocess_safa(None, leetspeak=True, alef_maqsura=True) == ""
        assert _preprocess_safa(123, leetspeak=True, alef_maqsura=True) == ""
        assert _preprocess_safa([], leetspeak=True, alef_maqsura=True) == ""

    def test_empty_string_returns_empty(self):
        """Empty string input returns empty."""
        assert self._preprocess_acm("") == ""

    def test_whitespace_only_returns_empty(self):
        """Whitespace-only input returns empty."""
        assert self._preprocess_acm("   ") == ""

    def test_valid_arabic_text_passes(self):
        """Normal Arabic text passes preprocessing."""
        result = self._preprocess_acm("مرحبا بالعالم العربي")
        assert result != ""
        assert "مرحبا" in result

    def test_whitespace_normalized(self):
        """Multiple whitespace characters are collapsed."""
        result = self._preprocess_acm("مرحبا    بالعالم    العربي")
        assert "  " not in result


# =============================================================================
# Kurdish (Sorani) preprocessing
# =============================================================================


class TestPreprocessKurdish:
    """Tests for Sorani Kurdish preprocessing via _preprocess_safa with Kurdish flags."""

    def _preprocess_ckb(self, text: str) -> str:
        """Kurdish convenience wrapper: leetspeak=False, alef_maqsura=False."""
        return _preprocess_safa(text, leetspeak=False, alef_maqsura=False)

    def test_no_leetspeak_decoding(self):
        """Kurdish does NOT decode leetspeak; numbers stay as numbers."""
        # Use a text without natural ain to isolate the test.
        # "3rbi" with leetspeak decoding would produce "عrbi".
        # Without it, the "3" is not replaced and the token stays as-is.
        acm_result = _preprocess_safa("3rbi مرحبا", leetspeak=True, alef_maqsura=True)
        ckb_result = self._preprocess_ckb("3rbi مرحبا")
        # Iraqi decodes: "3" -> "ع", so "عrbi" appears
        assert "عrbi" in acm_result
        # Kurdish does NOT decode, so "عrbi" must be absent
        assert "عrbi" not in ckb_result

    def test_no_alef_maqsura_normalization(self):
        """Kurdish does NOT normalize alef maqsura; it stays as-is."""
        result = self._preprocess_ckb("على مرحبا")
        assert "ى" in result

    def test_urls_stripped_same_as_iraqi(self):
        """URLs are still removed (shared behavior)."""
        result = self._preprocess_ckb("مرحبا http://example.com بالعالم")
        assert "http" not in result

    def test_mentions_stripped_same_as_iraqi(self):
        """Mentions are still removed (shared behavior)."""
        result = self._preprocess_ckb("مرحبا @user بالعالم")
        assert "@" not in result

    def test_alef_normalization_same_as_iraqi(self):
        """Alef variants are still normalized (shared behavior)."""
        result = self._preprocess_ckb("إبراهيم أحمد مرحبا")
        assert "إ" not in result
        assert "أ" not in result

    def test_repeated_chars_collapsed_same_as_iraqi(self):
        """Repeated chars are still collapsed (shared behavior)."""
        result = self._preprocess_ckb("هههههههه مرحبا بالعالم")
        assert "هههههههه" not in result
        assert "هه" in result

    def test_diacritics_stripped_same_as_iraqi(self):
        """Diacritics are still removed (shared behavior)."""
        result = self._preprocess_ckb("بِسْمِ اللهِ الرحمن")
        assert "\u0650" not in result

    def test_emoji_demojized_same_as_iraqi(self):
        """Emojis are still converted to text (shared behavior)."""
        result = self._preprocess_ckb("مرحبا 😊 بالعالم")
        assert "😊" not in result

    def test_no_arabic_returns_empty_same_as_iraqi(self):
        """Text with no Arabic script returns empty (shared behavior)."""
        result = self._preprocess_ckb("Hello World")
        assert result == ""

    def test_short_text_returns_empty_same_as_iraqi(self):
        """Text under 2 chars returns empty (shared behavior)."""
        result = self._preprocess_ckb("ا")
        assert result == ""

    def test_valid_arabic_text_passes(self):
        """Normal Arabic text passes preprocessing."""
        result = self._preprocess_ckb("مرحبا بالعالم العربي")
        assert result != ""


# =============================================================================
# Parametrized: shared behavior between Iraqi and Kurdish
# =============================================================================


@pytest.mark.parametrize(
    "leetspeak, alef_maqsura, dialect_name",
    [
        (True, True, "acm"),
        (False, False, "ckb"),
    ],
    ids=["iraqi", "kurdish"],
)
class TestSafaSharedBehavior:
    """Behaviors that are identical between Iraqi and Kurdish preprocessing."""

    def _preprocess(self, text, leetspeak, alef_maqsura):
        return _preprocess_safa(text, leetspeak=leetspeak, alef_maqsura=alef_maqsura)

    def test_url_removal(self, leetspeak, alef_maqsura, dialect_name):
        result = self._preprocess("مرحبا http://example.com بالعالم", leetspeak, alef_maqsura)
        assert "http" not in result

    def test_mention_removal(self, leetspeak, alef_maqsura, dialect_name):
        result = self._preprocess("مرحبا @user بالعالم", leetspeak, alef_maqsura)
        assert "@" not in result

    def test_hashtag_handling(self, leetspeak, alef_maqsura, dialect_name):
        result = self._preprocess("#مرحبا بالعالم", leetspeak, alef_maqsura)
        assert "#" not in result
        assert "مرحبا" in result

    def test_photo_removal(self, leetspeak, alef_maqsura, dialect_name):
        result = self._preprocess("[[photo]] مرحبا بالعالم", leetspeak, alef_maqsura)
        assert "photo" not in result

    def test_emoji_demojize(self, leetspeak, alef_maqsura, dialect_name):
        result = self._preprocess("مرحبا 😊 بالعالم", leetspeak, alef_maqsura)
        assert "😊" not in result

    def test_repeated_char_collapse(self, leetspeak, alef_maqsura, dialect_name):
        result = self._preprocess("هههههههه مرحبا بالعالم", leetspeak, alef_maqsura)
        assert "هههههههه" not in result

    def test_alef_normalization(self, leetspeak, alef_maqsura, dialect_name):
        result = self._preprocess("إبراهيم أحمد مرحبا", leetspeak, alef_maqsura)
        assert "إ" not in result
        assert "أ" not in result

    def test_diacritic_removal(self, leetspeak, alef_maqsura, dialect_name):
        result = self._preprocess("بِسْمِ اللهِ الرحمن", leetspeak, alef_maqsura)
        assert "\u0650" not in result

    def test_empty_returns_empty(self, leetspeak, alef_maqsura, dialect_name):
        assert self._preprocess("", leetspeak, alef_maqsura) == ""

    def test_whitespace_returns_empty(self, leetspeak, alef_maqsura, dialect_name):
        assert self._preprocess("   ", leetspeak, alef_maqsura) == ""

    def test_non_string_returns_empty(self, leetspeak, alef_maqsura, dialect_name):
        assert _preprocess_safa(None, leetspeak=leetspeak, alef_maqsura=alef_maqsura) == ""

    def test_no_arabic_returns_empty(self, leetspeak, alef_maqsura, dialect_name):
        assert self._preprocess("Hello World", leetspeak, alef_maqsura) == ""

    def test_under_2_chars_returns_empty(self, leetspeak, alef_maqsura, dialect_name):
        assert self._preprocess("ا", leetspeak, alef_maqsura) == ""

    def test_whitespace_normalization(self, leetspeak, alef_maqsura, dialect_name):
        result = self._preprocess("مرحبا    بالعالم    العربي", leetspeak, alef_maqsura)
        assert "  " not in result


# =============================================================================
# Parametrized: divergent behavior between Iraqi and Kurdish
# =============================================================================


class TestSafaDivergentBehavior:
    """Behaviors where Iraqi and Kurdish preprocessing differ."""

    def test_leetspeak_iraqi_decodes(self):
        """Iraqi decodes leetspeak (mixed digit+letter tokens)."""
        result = _preprocess_safa("7abibi مرحبا", leetspeak=True, alef_maqsura=True)
        assert "ح" in result

    def test_leetspeak_kurdish_does_not_decode(self):
        """Kurdish does NOT decode leetspeak."""
        result = _preprocess_safa("7abibi مرحبا", leetspeak=False, alef_maqsura=False)
        # Without leetspeak, 7abibi stays as-is then non-Arabic chars stripped
        # The key: no ha from leetspeak conversion
        assert "حabibi" not in result

    def test_alef_maqsura_iraqi_normalizes(self):
        """Iraqi normalizes alef maqsura to ya."""
        result = _preprocess_safa("على مرحبا", leetspeak=True, alef_maqsura=True)
        assert "ى" not in result

    def test_alef_maqsura_kurdish_preserves(self):
        """Kurdish preserves alef maqsura as-is."""
        result = _preprocess_safa("على مرحبا", leetspeak=False, alef_maqsura=False)
        assert "ى" in result

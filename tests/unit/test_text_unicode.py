from __future__ import annotations

import pytest

from tamil_voice.text.unicode import (
    DEFAULT_FORM,
    SUPPORTED_FORMS,
    Script,
    UnicodeError,
    is_normalized,
    normalize_text,
    script_histogram,
    script_of,
    scripts_in,
)

# U+0B94 TAMIL LETTER AU decomposes canonically to U+0B92 U+0BD7. This pair is the
# hazard GUIDE section 29 exists for: two strings that render identically and are
# different codepoint sequences, so a character tokenizer would see one symbol
# versus two. Four Tamil codepoints decompose this way (U+0B94, U+0BCA, U+0BCB,
# U+0BCC), verified against unicodedata rather than assumed.
NFC_AU = "\u0b94"
NFD_AU = "\u0b92\u0bd7"


def test_default_form_is_nfc() -> None:
    assert DEFAULT_FORM == "NFC"


def test_composed_and_decomposed_tamil_differ_before_normalizing() -> None:
    assert len(NFC_AU) == 1
    assert len(NFD_AU) == 2
    assert NFC_AU != NFD_AU


def test_normalization_collapses_the_decomposed_pair() -> None:
    assert normalize_text(NFD_AU) == NFC_AU
    assert normalize_text(NFD_AU, "NFC") == NFC_AU


def test_normalization_is_idempotent() -> None:
    once = normalize_text(NFD_AU)
    assert normalize_text(once) == once


def test_nfd_splits_the_composed_pair() -> None:
    assert normalize_text(NFC_AU, "NFD") == NFD_AU


def test_is_normalized_reports_state_without_changing_it() -> None:
    assert is_normalized(NFC_AU)
    assert not is_normalized(NFD_AU)
    assert is_normalized(NFD_AU, "NFD")


def test_plain_text_passes_through_unchanged() -> None:
    assert normalize_text("வணக்கம்") == "வணக்கம்"


@pytest.mark.parametrize("form", SUPPORTED_FORMS)
def test_every_supported_form_is_accepted(form: str) -> None:
    assert normalize_text(NFD_AU, form) in (NFC_AU, NFD_AU)


def test_unsupported_form_raises_rather_than_defaulting() -> None:
    with pytest.raises(UnicodeError, match="unsupported normalization form"):
        normalize_text("x", "NFZ")


def test_non_string_input_raises() -> None:
    with pytest.raises(UnicodeError, match="must be str"):
        normalize_text(b"bytes")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("char", "expected"),
    [
        ("க", Script.TAMIL),
        ("ா", Script.TAMIL),
        ("்", Script.TAMIL),
        # U+0BF0 TAMIL DIGIT ZERO is inside the Tamil block, and this is a script
        # classifier, so script identity wins over the Unicode Nd category.
        ("௰", Script.TAMIL),
        ("A", Script.LATIN),
        ("z", Script.LATIN),
        ("7", Script.DIGIT),
        (".", Script.PUNCTUATION),
        (",", Script.PUNCTUATION),
        (" ", Script.WHITESPACE),
        ("\n", Script.WHITESPACE),
        ("+", Script.SYMBOL),
    ],
)
def test_script_of_classifies_codepoints(char: str, expected: Script) -> None:
    assert script_of(char) is expected


def test_ascii_digits_are_digits_not_latin() -> None:
    """A digit is not a letter, even though it is ASCII."""
    assert script_of("7") is Script.DIGIT
    assert script_of("7") is not Script.LATIN


def test_script_of_covers_a_mixed_sentence() -> None:
    assert script_of("க") is Script.TAMIL
    assert script_of("A") is Script.LATIN
    assert script_of("4") is Script.DIGIT


def test_script_of_rejects_a_multi_codepoint_string() -> None:
    """Silently classifying only the first codepoint is the failure being prevented."""
    with pytest.raises(UnicodeError, match="exactly one codepoint"):
        script_of(NFD_AU)


def test_script_of_rejects_non_string() -> None:
    with pytest.raises(UnicodeError, match="must be str"):
        script_of(7)  # type: ignore[arg-type]


def test_scripts_in_reports_first_appearance_order() -> None:
    assert scripts_in("ABC") == (Script.LATIN,)
    assert scripts_in("கa1") == (Script.TAMIL, Script.LATIN, Script.DIGIT)
    assert scripts_in("1aக") == (Script.DIGIT, Script.LATIN, Script.TAMIL)


def test_scripts_in_is_empty_for_empty_text() -> None:
    assert scripts_in("") == ()


def test_scripts_in_normalizes_first() -> None:
    assert scripts_in(NFD_AU) == (Script.TAMIL,)


def test_script_histogram_counts_distinct_codepoints_not_occurrences() -> None:
    # "க்க்க" is க + ் + க + ்: two distinct codepoints, four occurrences.
    histogram = script_histogram(["க்க்க", "aab"])
    assert histogram[Script.TAMIL.value] == 2
    assert histogram[Script.LATIN.value] == 2
    assert histogram[Script.WHITESPACE.value] == 0


def test_script_histogram_reports_every_script_key() -> None:
    histogram = script_histogram(["க"])
    assert set(histogram) == {script.value for script in Script}


def test_script_histogram_over_empty_input_is_all_zero() -> None:
    assert set(script_histogram([]).values()) == {0}

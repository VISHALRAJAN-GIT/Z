"""Unicode handling for Tamil text.

GUIDE section 29 requires explicit Unicode normalization, and the reason is
specific rather than general. Tamil has composed and decomposed spellings of the
same grapheme, so counting codepoints is only meaningful once it is known which
form the text is in. Two texts that look identical can be different sequences of
codepoints, and a character tokenizer that treats them as equal-length targets
will train on inconsistent alignments.

EXP-004 measured the current corpus and found NFC changes 0 of 89401
transcripts, so the corpus is already normalized. That is a fact about
`dataset_v001`, not about Tamil, so normalization is applied unconditionally
here rather than skipped because it currently looks like a no-op.

Script classification exists for the same reason: the tokenizer must report what
it saw, so a future corpus that suddenly contains Latin letters or digits is
visible as a measurement instead of arriving as a silent distribution shift.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable
from enum import StrEnum
from typing import Literal, cast

#: The forms accepted by :func:`normalize_text`, as a Literal so the value can be
#: handed straight to :func:`unicodedata.normalize` without a cast at the call.
NormalizationForm = Literal["NFC", "NFD", "NFKC", "NFKD"]

#: The form this project stores and models text in.
DEFAULT_FORM: NormalizationForm = "NFC"

#: Forms accepted by :func:`normalize_text`.
SUPPORTED_FORMS: tuple[NormalizationForm, ...] = ("NFC", "NFD", "NFKC", "NFKD")

#: Inclusive codepoint range of the Tamil block, U+0B80..U+0BFF.
TAMIL_BLOCK_START = 0x0B80
TAMIL_BLOCK_END = 0x0BFF

#: Ranges treated as Latin, beyond ASCII. Tamil corpora in the future are
#: expected to contain code-switched English and product names.
_LATIN_RANGES: tuple[tuple[int, int], ...] = (
    (0x0041, 0x005A),
    (0x0061, 0x007A),
    (0x00C0, 0x024F),
    (0x1E00, 0x1EFF),
)


class UnicodeError(Exception):
    """Text cannot be normalized, or a form was requested that does not exist."""


class Script(StrEnum):
    """Coarse script or character class of a single codepoint."""

    TAMIL = "tamil"
    LATIN = "latin"
    DIGIT = "digit"
    PUNCTUATION = "punctuation"
    WHITESPACE = "whitespace"
    SYMBOL = "symbol"
    OTHER = "other"


def normalize_text(text: str, form: str = DEFAULT_FORM) -> str:
    """Return `text` in the requested normalization form.

    Raises:
        UnicodeError: `text` is not a string, or `form` is not one of
            :data:`SUPPORTED_FORMS`. A bad request fails here rather than
            silently falling back to NFC.
    """
    if not isinstance(text, str):
        raise UnicodeError(f"text must be str, got {type(text).__name__}")
    if form not in SUPPORTED_FORMS:
        raise UnicodeError(f"unsupported normalization form {form!r}, expected one of {SUPPORTED_FORMS}")
    # Membership in SUPPORTED_FORMS is exactly what unicodedata accepts, but the
    # signature wants a Literal, so narrow it here where it is proved.
    return unicodedata.normalize(cast(NormalizationForm, form), text)


def is_normalized(text: str, form: str = DEFAULT_FORM) -> bool:
    """True when `text` is already in `form`. Measurement, not correction."""
    return normalize_text(text, form) == text


def script_of(char: str) -> Script:
    """Classify one codepoint.

    Raises:
        UnicodeError: `char` is not exactly one character. Length is checked
            because silently classifying only the first codepoint of a
            decomposed cluster is the exact failure this module exists to stop.
    """
    if not isinstance(char, str):
        raise UnicodeError(f"char must be str, got {type(char).__name__}")
    if len(char) != 1:
        raise UnicodeError(f"expected exactly one codepoint, got {len(char)}: {char!r}")
    codepoint = ord(char)
    if char.isspace():
        return Script.WHITESPACE
    if TAMIL_BLOCK_START <= codepoint <= TAMIL_BLOCK_END:
        return Script.TAMIL
    if any(low <= codepoint <= high for low, high in _LATIN_RANGES):
        return Script.LATIN
    category = unicodedata.category(char)
    if category == "Nd":
        return Script.DIGIT
    if category.startswith("P"):
        return Script.PUNCTUATION
    if category.startswith("S"):
        return Script.SYMBOL
    return Script.OTHER


def scripts_in(text: str) -> tuple[Script, ...]:
    """Every script present in `text`, in first-appearance order.

    Order is first-appearance rather than sorted so the result reads as a
    description of the text.
    """
    seen: list[Script] = []
    for char in normalize_text(text):
        script = script_of(char)
        if script not in seen:
            seen.append(script)
    return tuple(seen)


def script_histogram(texts: Iterable[str], form: str = DEFAULT_FORM) -> dict[str, int]:
    """Count distinct codepoints per script across many strings.

    Counts distinct codepoints rather than occurrences, because the question this
    answers is "what does this corpus contain", not "how often is Tamil used".
    """
    histogram: dict[str, set[str]] = {script.value: set() for script in Script}
    for text in texts:
        for char in normalize_text(text, form):
            histogram[script_of(char).value].add(char)
    return {name: len(chars) for name, chars in histogram.items()}

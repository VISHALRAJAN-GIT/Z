"""Character-level tokenizer for Tamil ASR.

GUIDE section 28 specifies character-level as the starting point and requires
support for Tamil, English, digits, punctuation, special tokens and blank.
EXP-004 then measured the alternatives and agreed with the specification:
character-level has 0 unseen symbols on dev and test where word-level leaves
13.78 % of dev tokens out of vocabulary, and its output layer is 12593
parameters where word-level's is 35478336.

Two decisions inside this module are not settled by that experiment, so both are
made explicit here rather than buried:

**The inventory is derived from data, never hardcoded.** EXP-004 measured 48
distinct codepoints in ``dataset_v001``, but that corpus contains no ASCII
digits and no Latin letters at all, so it exercises none of the English, digit
and punctuation support GUIDE section 28 requires. A tokenizer carrying a
literal 48-symbol table would silently fail on the Phase 06 corpora. The
vocabulary is always built from the texts it will be asked to encode, and the
caller records which dataset version produced it.

**Whitespace handling is a real choice, and the default favours WER.** CTC
requires encoder frames >= labels, so every whitespace character is one more
label the model must emit. EXP-004 measured the cost: excluding whitespace
leaves 9 of 89401 utterances invalid, including it leaves 33. That is a
negligible difference. Excluding it, however, means a decoded string has no word
boundaries at all, which makes word error rate undefined and forces a separate
segmentation model. Since the project's first real WER depends on word
boundaries, :data:`TokenizerConfig` defaults to including whitespace and the
alternative is one flag away.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tamil_voice.text.unicode import DEFAULT_FORM, SUPPORTED_FORMS, normalize_text

#: CTC's no-output symbol. Fixed at 0 so a trained checkpoint stays loadable.
BLANK_TOKEN = "<blank>"

#: Emitted for any codepoint absent from the inventory.
UNK_TOKEN = "<unk>"

#: Ids reserved before the first real symbol, in this order.
BLANK_ID = 0
UNK_ID = 1

#: How many codepoints a symbol may contain. One, always, so a symbol can never
#: collide with a multi-character special token.
SYMBOL_WIDTH = 1


class TokenizerError(Exception):
    """The vocabulary cannot be built, or a request cannot be served."""


def prepare_text(text: str, config: TokenizerConfig) -> str:
    """Normalize, collapse whitespace runs, and strip the ends.

    Collapsing matters for more than tidiness: a transcript with a double space
    would otherwise contribute two target symbols to CTC for one pause, and
    those are frames the acoustic model has to earn.

    This is the single definition of text preparation. ``CharacterTokenizer.prepare``,
    :meth:`CharacterTokenizer.build` and :func:`build_from_jsonl` all route
    through it, because a vocabulary built from differently-prepared text than
    the text later encoded is a silent, total mismatch.
    """
    normalized = normalize_text(text, config.form)
    collapsed = " ".join(normalized.split())
    if not config.include_space:
        collapsed = collapsed.replace(" ", "")
    return collapsed


@dataclass(frozen=True)
class TokenizerConfig:
    """How a character vocabulary is built.

    Attributes:
        form: Unicode normalization form applied before anything else.
        include_space: keep whitespace as a target symbol. See the module
            docstring; the default exists so word boundaries survive decoding.
        min_frequency: symbols rarer than this are dropped. ``1`` keeps
            everything, which is the honest default for a closed inventory.
        max_symbols: cap on the number of real symbols. ``None`` means no cap.
    """

    form: str = DEFAULT_FORM
    include_space: bool = True
    min_frequency: int = 1
    max_symbols: int | None = None

    def __post_init__(self) -> None:
        if self.form not in SUPPORTED_FORMS:
            raise TokenizerError(f"unsupported normalization form {self.form!r}")
        if self.min_frequency < 1:
            raise TokenizerError(f"min_frequency must be >= 1, got {self.min_frequency}")
        if self.max_symbols is not None and self.max_symbols < 1:
            raise TokenizerError(f"max_symbols must be >= 1 or None, got {self.max_symbols}")


@dataclass(frozen=True)
class EncodingReport:
    """What one encode call actually did, for measurement rather than guessing."""

    text: str
    ids: tuple[int, ...]
    unknown_count: int
    unknown_symbols: tuple[str, ...]


@dataclass(frozen=True)
class CharacterTokenizer:
    """A fixed character inventory with encode and decode.

    Symbols are single codepoints. Ids are assigned as blank, unk, then real
    symbols ordered by descending corpus frequency with ties broken
    lexicographically, so the same corpus always yields byte-identical ids.
    """

    symbols: tuple[str, ...]
    config: TokenizerConfig
    counts: dict[str, int]

    def __post_init__(self) -> None:
        special = (BLANK_TOKEN, UNK_TOKEN)
        if self.symbols[: len(special)] != special:
            raise TokenizerError(
                f"symbols must start with {special}, got {self.symbols[: len(special)]}"
            )
        for symbol in self.symbols[len(special) :]:
            if len(symbol) != SYMBOL_WIDTH:
                raise TokenizerError(
                    f"symbol {symbol!r} has width {len(symbol)}, expected {SYMBOL_WIDTH}"
                )

    # ---- identity ------------------------------------------------------

    @property
    def blank_id(self) -> int:
        return BLANK_ID

    @property
    def unk_id(self) -> int:
        return UNK_ID

    @property
    def stoi(self) -> dict[str, int]:
        return {symbol: index for index, symbol in enumerate(self.symbols)}

    @property
    def real_symbols(self) -> tuple[str, ...]:
        """Every symbol except the two special ones."""
        return self.symbols[UNK_ID + 1 :]

    def __len__(self) -> int:
        return len(self.symbols)

    # ---- text preparation ----------------------------------------------

    def prepare(self, text: str) -> str:
        """Normalize, collapse whitespace runs, and strip the ends."""
        return prepare_text(text, self.config)

    # ---- encode / decode ------------------------------------------------

    def encode_with_report(self, text: str) -> EncodingReport:
        prepared = self.prepare(text)
        stoi = self.stoi
        ids: list[int] = []
        unknown: Counter[str] = Counter()
        for char in prepared:
            index = stoi.get(char)
            if index is None:
                unknown[char] += 1
                ids.append(UNK_ID)
            else:
                ids.append(index)
        return EncodingReport(
            text=prepared,
            ids=tuple(ids),
            unknown_count=sum(unknown.values()),
            unknown_symbols=tuple(sorted(unknown)),
        )

    def encode(self, text: str) -> list[int]:
        """Text to ids. Unseen codepoints become :data:`UNK_ID`, never dropped.

        Dropping them silently would shorten the target sequence and could make
        an utterance satisfy CTC's frames >= labels constraint for the wrong
        reason.
        """
        return list(self.encode_with_report(text).ids)

    def decode(self, ids: Iterable[int], *, keep_unknown: bool = False) -> str:
        """Ids back to text, dropping blank and (by default) unknown."""
        characters: list[str] = []
        for index in ids:
            resolved = self._symbol_at(index)
            if resolved == BLANK_TOKEN:
                continue
            if resolved == UNK_TOKEN and not keep_unknown:
                continue
            characters.append(resolved)
        return "".join(characters)

    def decode_words(self, ids: Iterable[int], *, keep_unknown: bool = False) -> list[str]:
        """Ids back to a word list, so word error rate is computable.

        With ``include_space`` disabled there are no boundaries to recover, so
        the whole string comes back as one word. That is a property of the
        configuration, not a bug, and it is why the default keeps whitespace.
        """
        text = self.decode(ids, keep_unknown=keep_unknown)
        if not self.config.include_space:
            return [text] if text else []
        return text.split(" ") if text else []

    def _symbol_at(self, index: int) -> str:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TokenizerError(f"ids must be ints, got {type(index).__name__}")
        if index < 0 or index >= len(self.symbols):
            raise TokenizerError(f"id {index} outside vocabulary of {len(self.symbols)}")
        return self.symbols[index]

    # ---- measurement ----------------------------------------------------

    def unknown_rate(self, texts: Iterable[str]) -> dict[str, float]:
        """Fraction of target symbols and of utterances containing an unknown.

        Reported per utterance as well as per symbol, because an utterance with
        one unknown symbol is a different problem from one that is mostly unknown.
        """
        total_symbols = 0
        total_unknown = 0
        utterances = 0
        affected = 0
        for text in texts:
            report = self.encode_with_report(text)
            total_symbols += len(report.ids)
            total_unknown += report.unknown_count
            utterances += 1
            if report.unknown_count:
                affected += 1
        return {
            "utterances": utterances,
            "symbols": total_symbols,
            "unknown_symbols": total_unknown,
            "unknown_symbol_rate": total_unknown / total_symbols if total_symbols else 0.0,
            "utterances_with_unknown_rate": affected / utterances if utterances else 0.0,
        }

    # ---- construction ----------------------------------------------------

    @classmethod
    def build(cls, texts: Iterable[str], config: TokenizerConfig | None = None) -> CharacterTokenizer:
        """Derive the inventory from `texts`.

        Ordering is frequency descending, ties lexicographic. That is a
        convention, not a requirement, but it has to be fixed or two builds over
        the same corpus would disagree on every id.
        """
        settings = config or TokenizerConfig()
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(prepare_text(text, settings))
        ordered = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        kept = [(symbol, count) for symbol, count in ordered if count >= settings.min_frequency]
        if settings.max_symbols is not None:
            kept = kept[: settings.max_symbols]
        if not kept:
            raise TokenizerError("no symbols survived min_frequency; nothing to encode")
        symbols = (BLANK_TOKEN, UNK_TOKEN) + tuple(symbol for symbol, _ in kept)
        return cls(symbols=symbols, config=settings, counts=dict(kept))

    # ---- persistence -----------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write the inventory as JSON, with the config needed to rebuild it."""
        payload: dict[str, Any] = {
            "kind": "character",
            "config": {
                "form": self.config.form,
                "include_space": self.config.include_space,
                "min_frequency": self.config.min_frequency,
                "max_symbols": self.config.max_symbols,
            },
            "blank_id": self.blank_id,
            "unk_id": self.unk_id,
            "size": len(self.symbols),
            "real_symbols": len(self.real_symbols),
            "counts": self.counts,
            "symbols": list(self.symbols),
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> CharacterTokenizer:
        """Read an inventory written by :meth:`save`.

        Validates rather than trusts: a file whose blank id is not 0, or whose
        symbols are not single codepoints, is rejected instead of producing a
        tokenizer that silently disagrees with the checkpoint that uses it.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("kind") != "character":
            raise TokenizerError(f"not a character tokenizer file: {path}")
        if payload.get("blank_id") != BLANK_ID or payload.get("unk_id") != UNK_ID:
            raise TokenizerError(
                f"blank/unk ids must be {BLANK_ID}/{UNK_ID}, got "
                f"{payload.get('blank_id')}/{payload.get('unk_id')}"
            )
        symbols = tuple(str(symbol) for symbol in payload["symbols"])
        if len(set(symbols)) != len(symbols):
            raise TokenizerError("symbols file contains duplicates")
        raw_config = payload.get("config", {})
        config = TokenizerConfig(
            form=str(raw_config.get("form", DEFAULT_FORM)),
            include_space=bool(raw_config.get("include_space", True)),
            min_frequency=int(raw_config.get("min_frequency", 1)),
            max_symbols=(
                None if raw_config.get("max_symbols") is None else int(raw_config["max_symbols"])
            ),
        )
        return cls(symbols=symbols, config=config, counts=dict(payload.get("counts", {})))


def _prepared_chars(text: str, config: TokenizerConfig) -> list[str]:
    """Characters of `text` after the same preparation `prepare` applies."""
    return list(prepare_text(text, config))


def build_from_jsonl(
    path: str | Path,
    *,
    text_field: str = "text",
    config: TokenizerConfig | None = None,
    limit: int | None = None,
) -> CharacterTokenizer:
    """Build a tokenizer from the ``text_field`` of a JSONL manifest.

    Manifests are the project's pinned record of a dataset version, so deriving
    the vocabulary from one is what ties the inventory to a version. Audio is
    never read.
    """
    settings = config or TokenizerConfig()
    texts: list[str] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if text_field not in row:
                raise TokenizerError(f"{path}: line {line_number} has no field {text_field!r}")
            texts.append(str(row[text_field]))
            if limit is not None and len(texts) >= limit:
                break
    return CharacterTokenizer.build(texts, settings)


def sequence_length_report(
    tokenizer: CharacterTokenizer, texts: Sequence[str]
) -> dict[str, float]:
    """Target-length statistics, the input CTC's frames >= labels check needs.

    Kept here so the vocabulary and the lengths it implies are measured by the
    same code that will produce the targets.
    """
    lengths = [len(tokenizer.encode(text)) for text in texts]
    if not lengths:
        return {"utterances": 0, "total": 0, "mean": 0.0, "median": 0.0, "max": 0}
    ordered = sorted(lengths)
    return {
        "utterances": len(ordered),
        "total": sum(ordered),
        "mean": sum(ordered) / len(ordered),
        "median": float(ordered[len(ordered) // 2]),
        "max": ordered[-1],
    }

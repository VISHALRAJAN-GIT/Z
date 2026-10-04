from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from tamil_voice.text.tokenizer import (
    BLANK_ID,
    BLANK_TOKEN,
    UNK_ID,
    UNK_TOKEN,
    CharacterTokenizer,
    TokenizerConfig,
    TokenizerError,
    build_from_jsonl,
    prepare_text,
    sequence_length_report,
)

CORPUS = ["வணக்கம்", "வணக்கம்", "தமிழ்", "தமிழ் தமிழ்", "hello", "42"]


def _tokenizer(**kwargs) -> CharacterTokenizer:
    return CharacterTokenizer.build(CORPUS, TokenizerConfig(**kwargs))


def test_special_ids_are_reserved_in_a_fixed_order() -> None:
    tokenizer = _tokenizer()
    assert tokenizer.blank_id == BLANK_ID == 0
    assert tokenizer.unk_id == UNK_ID == 1
    assert tokenizer.symbols[0] == BLANK_TOKEN
    assert tokenizer.symbols[1] == UNK_TOKEN


def test_inventory_is_derived_from_the_texts_not_hardcoded() -> None:
    tokenizer = _tokenizer()
    assert set(tokenizer.real_symbols) == set("வணகம்திழ் hello42")
    assert "7" not in tokenizer.real_symbols, "no digits were supplied, so none may appear"


def test_symbols_are_ordered_by_frequency_then_lexicographically() -> None:
    tokenizer = _tokenizer()
    counts = tokenizer.counts
    ordered = [tokenizer.symbols[index] for index in range(2, len(tokenizer.symbols))]
    keys = [(-counts[symbol], symbol) for symbol in ordered]
    assert keys == sorted(keys)


def test_build_is_deterministic_across_calls() -> None:
    assert _tokenizer().symbols == _tokenizer().symbols


def test_build_is_order_independent() -> None:
    """Id assignment must depend on the multiset of symbols, not the input order."""
    forward = CharacterTokenizer.build(CORPUS)
    backward = CharacterTokenizer.build(list(reversed(CORPUS)))
    assert forward.symbols == backward.symbols


def test_encode_decode_round_trip() -> None:
    tokenizer = _tokenizer()
    assert tokenizer.decode(tokenizer.encode("வணக்கம்")) == "வணக்கம்"


def test_round_trip_survives_unknown_symbols() -> None:
    tokenizer = _tokenizer()
    assert tokenizer.decode(tokenizer.encode("வணqக்கம்")) == "வணக்கம்"


def test_unknown_codepoints_become_unk_and_are_counted() -> None:
    tokenizer = _tokenizer()
    report = tokenizer.encode_with_report("வணக்கம் zq")
    assert report.unknown_count == 2
    assert report.unknown_symbols == ("q", "z")
    assert report.ids.count(UNK_ID) == 2


def test_unknown_symbols_are_not_silently_dropped() -> None:
    """Dropping would shorten the target and could satisfy CTC for the wrong reason."""
    tokenizer = _tokenizer()
    assert len(tokenizer.encode("z")) == 1


def test_unknown_rate_reports_symbols_and_utterances() -> None:
    tokenizer = _tokenizer()
    texts = ["வணக்கம்", "zq", "qqq"]
    report = tokenizer.unknown_rate(texts)
    assert report["utterances"] == 3
    assert report["unknown_symbols"] == 5
    assert report["symbols"] == sum(len(tokenizer.encode(t)) for t in texts)
    assert report["unknown_symbol_rate"] == pytest.approx(5 / report["symbols"])
    assert report["utterances_with_unknown_rate"] == pytest.approx(2 / 3)


def test_unknown_rate_over_no_text_is_zero_not_a_division_error() -> None:
    report = _tokenizer().unknown_rate([])
    assert report["unknown_symbol_rate"] == 0.0
    assert report["utterances_with_unknown_rate"] == 0.0


def test_decode_drops_blank_and_unknown_by_default() -> None:
    tokenizer = _tokenizer()
    ids = [BLANK_ID, UNK_ID] + tokenizer.encode("வணக்கம்") + [BLANK_ID]
    assert tokenizer.decode(ids) == "வணக்கம்"


def test_decode_can_keep_unknown() -> None:
    tokenizer = _tokenizer()
    assert tokenizer.decode([UNK_ID], keep_unknown=True) == UNK_TOKEN


def test_decode_words_recovers_word_boundaries() -> None:
    tokenizer = _tokenizer()
    assert tokenizer.decode_words(tokenizer.encode("தமிழ் தமிழ்")) == ["தமிழ்", "தமிழ்"]


def test_decode_words_without_spaces_returns_one_word() -> None:
    """Documented consequence of include_space=False: no boundaries to recover."""
    tokenizer = _tokenizer(include_space=False)
    assert tokenizer.decode_words(tokenizer.encode("தமிழ் தமிழ்")) == ["தமிழ்தமிழ்"]


def test_decode_words_of_blank_only_is_empty() -> None:
    assert _tokenizer().decode_words([BLANK_ID, BLANK_ID]) == []


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("தமிழ்  தமிழ்", "தமிழ் தமிழ்"),
        ("  தமிழ்  ", "தமிழ்"),
        ("தமிழ்\nதமிழ்", "தமிழ் தமிழ்"),
        ("", ""),
        ("   ", ""),
    ],
)
def test_prepare_collapses_and_strips_whitespace(raw: str, expected: str) -> None:
    assert prepare_text(raw, TokenizerConfig()) == expected


def test_prepare_normalizes_to_the_configured_form() -> None:
    decomposed = "\u0b92\u0bd7"
    assert prepare_text(decomposed, TokenizerConfig(form="NFC")) == "\u0b94"
    assert prepare_text(decomposed, TokenizerConfig(form="NFD")) == decomposed


def test_prepare_removes_spaces_when_configured() -> None:
    assert prepare_text("தமிழ் தமிழ்", TokenizerConfig(include_space=False)) == "தமிழ்தமிழ்"


def test_tokenizer_prepare_matches_the_module_function() -> None:
    tokenizer = _tokenizer()
    raw = "  வணக்கம்   தமிழ் "
    assert tokenizer.prepare(raw) == prepare_text(raw, tokenizer.config)


def test_encode_uses_prepare_so_collapsing_happens_before_encoding() -> None:
    tokenizer = _tokenizer()
    assert tokenizer.encode("தமிழ்  தமிழ்") == tokenizer.encode("தமிழ் தமிழ்")


def test_decode_rejects_out_of_range_ids() -> None:
    tokenizer = _tokenizer()
    with pytest.raises(TokenizerError, match="outside vocabulary"):
        tokenizer.decode([len(tokenizer.symbols)])
    with pytest.raises(TokenizerError, match="outside vocabulary"):
        tokenizer.decode([-1])


def test_decode_rejects_non_integer_ids() -> None:
    with pytest.raises(TokenizerError, match="ids must be ints"):
        _tokenizer().decode(["வ"])  # type: ignore[list-item]


def test_min_frequency_drops_rare_symbols() -> None:
    tokenizer = _tokenizer(min_frequency=2)
    assert "q" not in tokenizer.real_symbols
    assert "வ" in tokenizer.real_symbols


def test_max_symbols_caps_the_inventory() -> None:
    tokenizer = _tokenizer(max_symbols=3)
    assert len(tokenizer.real_symbols) == 3


def test_max_symbols_keeps_the_most_frequent() -> None:
    tokenizer = _tokenizer(max_symbols=2)
    counts = tokenizer.counts
    assert counts[tokenizer.real_symbols[0]] >= counts[tokenizer.real_symbols[1]]


def test_build_over_only_unknown_raising_text_still_succeeds() -> None:
    tokenizer = CharacterTokenizer.build(["வணக்கம்"])
    assert tokenizer.unknown_rate(["z"])["unknown_symbol_rate"] == 1.0


def test_build_over_empty_input_raises_rather_than_making_an_empty_vocabulary() -> None:
    with pytest.raises(TokenizerError, match="no symbols survived"):
        CharacterTokenizer.build([])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"form": "NFZ"},
        {"min_frequency": 0},
        {"max_symbols": 0},
    ],
)
def test_invalid_config_raises(kwargs: dict[str, object]) -> None:
    with pytest.raises(TokenizerError):
        TokenizerConfig(**kwargs)  # type: ignore[arg-type]


def test_symbols_must_start_with_the_special_tokens() -> None:
    with pytest.raises(TokenizerError, match="must start with"):
        CharacterTokenizer(symbols=("வ", UNK_TOKEN), config=TokenizerConfig(), counts={})


def test_multi_codepoint_symbols_are_rejected() -> None:
    """A two-codepoint symbol could collide with a special token at decode time."""
    with pytest.raises(TokenizerError, match="has width"):
        CharacterTokenizer(
            symbols=(BLANK_TOKEN, UNK_TOKEN, "வண"),
            config=TokenizerConfig(),
            counts={},
        )


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    tokenizer = _tokenizer(min_frequency=1)
    path = tmp_path / "tok.json"
    tokenizer.save(path)
    loaded = CharacterTokenizer.load(path)
    assert loaded.symbols == tokenizer.symbols
    assert loaded.config == tokenizer.config
    assert loaded.counts == tokenizer.counts


def test_save_writes_unicode_without_escaping(tmp_path: Path) -> None:
    path = tmp_path / "tok.json"
    _tokenizer().save(path)
    assert "வ" in path.read_text(encoding="utf-8")
    assert "\\u0b" not in path.read_text(encoding="utf-8")


def test_load_rejects_a_non_character_tokenizer_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"kind": "word"}), encoding="utf-8")
    with pytest.raises(TokenizerError, match="not a character tokenizer"):
        CharacterTokenizer.load(path)


def test_load_rejects_moved_special_ids(tmp_path: Path) -> None:
    """A file claiming blank=1 would silently disagree with a trained checkpoint."""
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps({"kind": "character", "blank_id": 1, "unk_id": 2, "symbols": []}),
        encoding="utf-8",
    )
    with pytest.raises(TokenizerError, match="blank/unk ids"):
        CharacterTokenizer.load(path)


def test_load_rejects_duplicate_symbols(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "kind": "character",
                "blank_id": 0,
                "unk_id": 1,
                "symbols": [BLANK_TOKEN, UNK_TOKEN, "வ", "வ"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(TokenizerError, match="duplicates"):
        CharacterTokenizer.load(path)


def test_build_from_jsonl_reads_the_text_field(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        "\n".join(json.dumps({"text": t}, ensure_ascii=False) for t in CORPUS) + "\n",
        encoding="utf-8",
    )
    tokenizer = build_from_jsonl(path)
    assert set(tokenizer.real_symbols) == set(CharacterTokenizer.build(CORPUS).real_symbols)


def test_build_from_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        '{"text": "வணக்கம்"}\n\n\n{"text": "தமிழ்"}\n',
        encoding="utf-8",
    )
    tokenizer = build_from_jsonl(path)
    assert tokenizer.counts["வ"] == 1
    assert tokenizer.counts["த"] == 1


def test_build_from_jsonl_honours_limit(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        "\n".join(json.dumps({"text": t}, ensure_ascii=False) for t in CORPUS) + "\n",
        encoding="utf-8",
    )
    limited = build_from_jsonl(path, limit=1)
    assert limited.counts == dict(Counter("வணக்கம்"))


def test_build_from_jsonl_reports_a_missing_field(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    path.write_text('{"utt": "x"}\n', encoding="utf-8")
    with pytest.raises(TokenizerError, match="no field"):
        build_from_jsonl(path)


def test_sequence_length_report_measures_target_lengths() -> None:
    tokenizer = _tokenizer()
    report = sequence_length_report(tokenizer, ["வணக்கம்", "தமிழ் தமிழ்"])
    assert report["utterances"] == 2
    assert report["total"] == len(tokenizer.encode("வணக்கம்")) + len(
        tokenizer.encode("தமிழ் தமிழ்")
    )
    assert report["max"] >= report["median"]


def test_sequence_length_report_over_no_text_is_zero() -> None:
    report = sequence_length_report(_tokenizer(), [])
    assert report["utterances"] == 0
    assert report["max"] == 0


# EXP-004 measured 48 distinct codepoints in dataset_v001: 47 in the Tamil block
# plus the space. Asserting it here ties the module to the experiment's number, so a
# change in what the corpus contains fails a test instead of quietly changing every
# id in a future checkpoint. Skipped when the manifest is absent, because the unit
# suite must not require the 16 GB corpus.
MANIFEST = Path("data/manifests/dataset_v001/train.jsonl")


@pytest.mark.skipif(not MANIFEST.exists(), reason="dataset_v001 manifests are not present")
def test_real_corpus_inventory_matches_exp004() -> None:
    """EXP-004 measured 48 distinct codepoints in train: 47 Tamil plus the space.

    Asserted here so a change in corpus content fails a test rather than silently
    renumbering every id in a future checkpoint. The space count is the
    arithmetic EXP-004 recorded: 6061783 characters total minus 5465563
    non-space characters.
    """
    tokenizer = build_from_jsonl(MANIFEST)
    assert len(tokenizer.real_symbols) == 48
    assert len(tokenizer.symbols) == 50  # 48 symbols + blank + unk
    assert " " in tokenizer.real_symbols
    assert sum(1 for s in tokenizer.real_symbols if 0x0B80 <= ord(s) <= 0x0BFF) == 47
    assert tokenizer.counts[" "] == 6061783 - 5465563


@pytest.mark.skipif(not MANIFEST.exists(), reason="dataset_v001 manifests are not present")
def test_real_corpus_has_no_unknown_symbols_on_dev() -> None:
    """EXP-004 measured 0 unseen codepoints on dev and test. This is that claim."""
    train = build_from_jsonl(MANIFEST)
    dev_report = train.unknown_rate(_read_texts(Path("data/manifests/dataset_v001/dev.jsonl")))
    assert dev_report["unknown_symbols"] == 0
    assert dev_report["unknown_symbol_rate"] == 0.0


def _read_texts(path: Path) -> list[str]:
    return [
        json.loads(line)["text"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

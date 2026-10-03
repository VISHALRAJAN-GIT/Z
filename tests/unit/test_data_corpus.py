from __future__ import annotations

from pathlib import Path

import pytest

from tamil_voice.data.corpus import (
    CorpusError,
    discover_iisc_mile,
    parse_iisc_mile_name,
    read_transcript,
)


def test_parse_valid_name() -> None:
    assert parse_iisc_mile_name("MILE_0000123_0000045") == ("MILE", "0000123", "0000045")
    assert parse_iisc_mile_name("ISTL_0000202_0000009") == ("ISTL", "0000202", "0000009")
    assert parse_iisc_mile_name("MICI_0000000_0000004") == ("MICI", "0000000", "0000004")


@pytest.mark.parametrize(
    "bad",
    ["", "MILE", "MILE_123", "MILE_abc_0000000", "MILE_0000000", "MILE_0000000_0000001.wav", "0000000_1_2"],
)
def test_parse_invalid_name_raises(bad: str) -> None:
    with pytest.raises(CorpusError):
        parse_iisc_mile_name(bad)


def test_read_transcript_strips_and_decodes_tamil(tmp_path: Path) -> None:
    path = tmp_path / "u.txt"
    path.write_text("\n  வணக்கம்  \n", encoding="utf-8")
    assert read_transcript(path) == "வணக்கம்"


def test_read_transcript_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(CorpusError):
        read_transcript(tmp_path / "missing.txt")


def test_read_transcript_rejects_non_utf8(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_bytes(b"\xff\xfe\x00\x00not text")
    with pytest.raises(CorpusError):
        read_transcript(path)


def _make_corpus(root: Path, entries: dict[str, list[str]]) -> None:
    for split, stems in entries.items():
        audio = root / split / "audio_files"
        text = root / split / "trans_files"
        audio.mkdir(parents=True)
        text.mkdir(parents=True)
        for stem in stems:
            (audio / f"{stem}.wav").write_bytes(b"")
            (text / f"{stem}.txt").write_text("சொல்", encoding="utf-8")


def test_discover_finds_all_and_records_metadata(tmp_path: Path) -> None:
    _make_corpus(
        tmp_path,
        {
            "train": ["MILE_0000000_0000000", "MILE_0000000_0000001"],
            "test": ["ISTL_0000202_0000009"],
        },
    )
    utterances = discover_iisc_mile(tmp_path)
    assert [u.utterance_id for u in utterances] == [
        "MILE_0000000_0000000",
        "MILE_0000000_0000001",
        "ISTL_0000202_0000009",
    ]
    by_id = {u.utterance_id: u for u in utterances}
    assert by_id["MILE_0000000_0000001"].speaker_id == "0000000"
    assert by_id["MILE_0000000_0000001"].prefix == "MILE"
    assert by_id["MILE_0000000_0000001"].shipped_split == "train"
    assert by_id["ISTL_0000202_0000009"].shipped_split == "test"


def test_discover_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(CorpusError):
        discover_iisc_mile(tmp_path / "nope")


def test_discover_missing_transcript_raises(tmp_path: Path) -> None:
    root = tmp_path
    audio = root / "train" / "audio_files"
    text = root / "train" / "trans_files"
    audio.mkdir(parents=True)
    text.mkdir(parents=True)
    (audio / "MILE_0000000_0000000.wav").write_bytes(b"")
    with pytest.raises(CorpusError):
        discover_iisc_mile(root)

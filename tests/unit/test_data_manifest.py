from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tamil_voice.data.corpus import CorpusError, Utterance
from tamil_voice.data.manifest import build_records, read_manifest, write_manifest


def _write_wav(path: Path, seconds: float = 1.0, rate: int = 16000) -> None:
    times = np.arange(int(seconds * rate), dtype=np.float32) / rate
    sf.write(path, (0.1 * np.sin(2 * np.pi * 220.0 * times)).astype(np.float32), rate, subtype="PCM_16")


def _utterance(root: Path, stem: str, speaker: str) -> Utterance:
    audio = root / "audio_files" / f"{stem}.wav"
    text = root / "trans_files" / f"{stem}.txt"
    _write_wav(audio)
    text.write_text("தமிழ் சொல்", encoding="utf-8")
    return Utterance(stem, speaker, "MILE", "train", audio, text)


def _corpus(tmp_path: Path) -> tuple[Path, Utterance]:
    data_root = tmp_path / "data"
    root = data_root / "raw" / "corpus"
    (root / "audio_files").mkdir(parents=True)
    (root / "trans_files").mkdir(parents=True)
    return data_root, _utterance(root, "MILE_0000000_0000001", "0000000")


def test_build_records_fields_and_relative_path(tmp_path: Path) -> None:
    data_root, utt = _corpus(tmp_path)
    records = build_records([utt], dataset_id="iisc_mile_ta", data_root=data_root)
    assert len(records) == 1
    record = records[0]
    assert record.utterance_id == "MILE_0000000_0000001"
    assert record.speaker_id == "0000000"
    assert record.prefix == "MILE"
    assert record.shipped_split == "train"
    assert record.sample_rate == 16000
    assert record.duration_seconds == pytest.approx(1.0, abs=0.01)
    assert record.audio_path == "raw/corpus/audio_files/MILE_0000000_0000001.wav"
    assert record.text == "தமிழ் சொல்"
    assert record.dataset_id == "iisc_mile_ta"


def test_build_records_missing_audio_raises(tmp_path: Path) -> None:
    data_root, utt = _corpus(tmp_path)
    utt.audio_path.unlink()
    with pytest.raises(CorpusError):
        build_records([utt], dataset_id="iisc_mile_ta", data_root=data_root)


def test_write_and_read_manifest_roundtrip_keeps_tamil(tmp_path: Path) -> None:
    data_root, utt = _corpus(tmp_path)
    path = tmp_path / "train.jsonl"
    records = build_records([utt], dataset_id="iisc_mile_ta", data_root=data_root)
    assert write_manifest(path, records) == 1

    raw = path.read_text(encoding="utf-8")
    assert "தமிழ்" in raw
    assert "\\u0b" not in raw

    rows = read_manifest(path)
    assert rows[0]["text"] == "தமிழ் சொல்"
    assert rows[0]["speaker_id"] == "0000000"


def test_read_manifest_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(CorpusError):
        read_manifest(tmp_path / "nope.jsonl")


def test_read_manifest_invalid_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(CorpusError):
        read_manifest(path)

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import soundfile as sf

from tamil_voice.audio.io import (
    CANONICAL_SAMPLE_RATE,
    ISSUE_CLIPPED,
    ISSUE_DC_OFFSET,
    ISSUE_EMPTY,
    ISSUE_INFINITE,
    ISSUE_LOW_AMPLITUDE,
    ISSUE_NAN,
    ISSUE_SILENT,
    ISSUE_TOO_LONG,
    ISSUE_UNSUPPORTED_SAMPLE_RATE,
    AudioData,
    AudioLoadError,
    AudioValidationError,
    ValidationLimits,
    audio_info,
    load_audio,
    validate_audio,
)

FloatArray = npt.NDArray[np.float32]


def _sine(frames: int, sample_rate: int = 16_000, frequency: float = 440.0, amp: float = 0.5) -> FloatArray:
    t = np.arange(frames, dtype=np.float64) / sample_rate
    return (amp * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def _write(path: Path, data: FloatArray, sample_rate: int = 16_000, subtype: str = "FLOAT") -> Path:
    sf.write(str(path), data, sample_rate, subtype=subtype)
    return path


def _codes(report_codes: object) -> set[str]:
    return {issue.code for issue in report_codes}  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- load


def test_loads_mono_tone(tmp_path: Path) -> None:
    path = _write(tmp_path / "tone.wav", _sine(8000), 16_000)
    audio = load_audio(path)
    assert isinstance(audio, AudioData)
    assert audio.waveform.dtype == np.float32
    assert audio.waveform.ndim == 1
    assert audio.sample_rate == 16_000
    assert audio.num_frames == 8000
    assert audio.channels == 1
    assert audio.is_mono is True
    assert audio.duration == pytest.approx(0.5, abs=1e-6)


def test_native_sample_rate_is_preserved(tmp_path: Path) -> None:
    """io.py must not silently resample; that is resampling.py's job."""
    path = _write(tmp_path / "hi.wav", _sine(48_000, 48_000), 48_000)
    audio = load_audio(path)
    assert audio.sample_rate == 48_000
    assert audio.duration == pytest.approx(1.0, abs=1e-6)


def test_canonical_constant_matches_guide() -> None:
    assert CANONICAL_SAMPLE_RATE == 16_000


def test_stereo_preserved_when_mono_false(tmp_path: Path) -> None:
    left = _sine(4000, amp=0.5)
    right = _sine(4000, frequency=220.0, amp=0.5)
    stereo = np.stack([left, right], axis=1).astype(np.float32)
    path = _write(tmp_path / "stereo.wav", stereo, 16_000)
    audio = load_audio(path, mono=False)
    assert audio.waveform.ndim == 2
    assert audio.waveform.shape == (4000, 2)
    assert audio.channels == 2
    assert audio.is_mono is False


def test_stereo_downmixed_to_mono_by_default(tmp_path: Path) -> None:
    n = 4000
    stereo = np.stack([np.ones(n, np.float32), np.zeros(n, np.float32)], axis=1).astype(np.float32)
    path = _write(tmp_path / "st.wav", stereo, 16_000)
    audio = load_audio(path)
    assert audio.waveform.ndim == 1
    assert audio.channels == 1
    assert np.allclose(audio.waveform, 0.5, atol=1e-6)


def test_missing_file_raises_load_error(tmp_path: Path) -> None:
    with pytest.raises(AudioLoadError):
        load_audio(tmp_path / "nope.wav")


def test_corrupt_file_raises_load_error(tmp_path: Path) -> None:
    bad = tmp_path / "corrupt.wav"
    bad.write_bytes(b"this is not audio at all" * 4)
    with pytest.raises(AudioLoadError):
        load_audio(bad)


def test_empty_file_treated_as_failure(tmp_path: Path) -> None:
    """A header with zero frames decodes to nothing and must not pass silently."""
    empty = tmp_path / "empty.wav"
    with wave.open(str(empty), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
    with pytest.raises(AudioValidationError) as excinfo:
        load_audio(empty)
    assert ISSUE_EMPTY in _codes(excinfo.value.report.issues)


# ---------------------------------------------------------------------- validation


def test_nan_rejected(tmp_path: Path) -> None:
    data = np.array([0.1, 0.2, np.nan, 0.3], dtype=np.float32)
    path = _write(tmp_path / "nan.wav", data)
    with pytest.raises(AudioValidationError) as excinfo:
        load_audio(path)
    assert ISSUE_NAN in _codes(excinfo.value.report.issues)


def test_infinite_rejected(tmp_path: Path) -> None:
    data = np.array([0.1, np.inf, 0.2], dtype=np.float32)
    path = _write(tmp_path / "inf.wav", data)
    with pytest.raises(AudioValidationError) as excinfo:
        load_audio(path)
    assert ISSUE_INFINITE in _codes(excinfo.value.report.issues)


def test_silent_audio_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path / "silence.wav", np.zeros(16_000, dtype=np.float32))
    with pytest.raises(AudioValidationError) as excinfo:
        load_audio(path)
    assert ISSUE_SILENT in _codes(excinfo.value.report.issues)


def test_clipping_is_a_warning_not_an_error(tmp_path: Path) -> None:
    """Clipped audio is still loadable; it must be reported, not discarded."""
    path = _write(tmp_path / "clip.wav", np.full(16_000, 1.0, dtype=np.float32))
    audio = load_audio(path)
    assert audio.report is not None
    assert ISSUE_CLIPPED in _codes(audio.report.warnings)
    assert audio.report.ok is True


def test_low_amplitude_is_a_warning(tmp_path: Path) -> None:
    path = _write(tmp_path / "quiet.wav", _sine(16_000, amp=1e-6))
    audio = load_audio(path)
    assert audio.report is not None
    assert ISSUE_LOW_AMPLITUDE in _codes(audio.report.warnings)
    assert ISSUE_SILENT not in _codes(audio.report.issues)


def test_dc_offset_is_a_warning(tmp_path: Path) -> None:
    path = _write(tmp_path / "dc.wav", np.full(16_000, 0.3, dtype=np.float32))
    audio = load_audio(path)
    assert audio.report is not None
    assert ISSUE_DC_OFFSET in _codes(audio.report.warnings)


def test_unsupported_low_sample_rate_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path / "low.wav", _sine(4000, 4000), 4000)
    with pytest.raises(AudioValidationError) as excinfo:
        load_audio(path)
    assert ISSUE_UNSUPPORTED_SAMPLE_RATE in _codes(excinfo.value.report.issues)


def test_8khz_is_supported(tmp_path: Path) -> None:
    """GUIDE section 13 requires 8 kHz input to work."""
    path = _write(tmp_path / "8k.wav", _sine(8000, 8000), 8000)
    audio = load_audio(path)
    assert audio.sample_rate == 8000


def test_excessively_long_recording_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path / "long.wav", _sine(1600), 16_000)
    limits = ValidationLimits(max_duration_seconds=0.01)
    with pytest.raises(AudioValidationError) as excinfo:
        load_audio(path, limits=limits)
    assert ISSUE_TOO_LONG in _codes(excinfo.value.report.issues)


def test_validation_can_be_disabled(tmp_path: Path) -> None:
    path = _write(tmp_path / "silence.wav", np.zeros(1600, dtype=np.float32))
    audio = load_audio(path, validate=False)
    assert audio.report is None
    assert audio.num_frames == 1600


def test_validation_error_carries_every_issue(tmp_path: Path) -> None:
    data = np.full(1600, 1.0, dtype=np.float32)
    path = _write(tmp_path / "multi.wav", data)
    with pytest.raises(AudioValidationError) as excinfo:
        load_audio(path, limits=ValidationLimits(max_duration_seconds=0.001))
    codes = _codes(excinfo.value.report.issues)
    assert ISSUE_TOO_LONG in codes
    assert ISSUE_CLIPPED in codes


# ------------------------------------------------------------------- validate_audio


def test_validate_clean_tone_reports_ok() -> None:
    report = validate_audio(_sine(16_000), 16_000)
    assert report.ok is True
    assert report.issues == ()
    assert report.metrics["peak"] == pytest.approx(0.5, abs=1e-3)
    assert report.metrics["duration"] == pytest.approx(1.0, abs=1e-6)


def test_validate_empty_array_reports_empty() -> None:
    report = validate_audio(np.zeros(0, dtype=np.float32), 16_000)
    assert report.ok is False
    assert ISSUE_EMPTY in _codes(report.issues)


def test_validate_computes_crest_factor() -> None:
    report = validate_audio(_sine(16_000), 16_000)
    assert report.metrics["crest_factor"] > 0.0


def test_validate_zero_sample_rate() -> None:
    report = validate_audio(_sine(1600), 0)
    assert report.ok is False
    assert ISSUE_UNSUPPORTED_SAMPLE_RATE in _codes(report.issues)


def test_validate_summary_lists_codes() -> None:
    report = validate_audio(np.zeros(1600, dtype=np.float32), 16_000)
    assert ISSUE_SILENT in report.summary()
    assert validate_audio(_sine(1600), 16_000).summary() == "ok"


# ------------------------------------------------------------------------- helpers


def test_to_mono_is_noop_for_mono(tmp_path: Path) -> None:
    path = _write(tmp_path / "tone.wav", _sine(1600), 16_000)
    audio = load_audio(path)
    assert audio.to_mono() is audio


def test_to_mono_downmixes(tmp_path: Path) -> None:
    n = 1600
    stereo = np.stack([np.ones(n, np.float32), np.zeros(n, np.float32)], axis=1).astype(np.float32)
    path = _write(tmp_path / "st.wav", stereo, 16_000)
    audio = load_audio(path, mono=False, validate=False)
    mono = audio.to_mono()
    assert mono.is_mono is True
    assert mono.num_frames == n


def test_audio_info_reads_header(tmp_path: Path) -> None:
    stereo = np.stack([_sine(4800, 48_000), _sine(4800, 48_000)], axis=1).astype(np.float32)
    path = _write(tmp_path / "st48.wav", stereo, 48_000)
    info = audio_info(path)
    assert info["sample_rate"] == 48_000
    assert info["channels"] == 2
    assert info["frames"] == 4800
    assert info["duration"] == pytest.approx(0.1, abs=1e-6)


def test_audio_info_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(AudioLoadError):
        audio_info(tmp_path / "nope.wav")

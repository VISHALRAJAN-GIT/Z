from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from tamil_voice.audio.io import CANONICAL_SAMPLE_RATE, AudioData
from tamil_voice.audio.resampling import (
    resample_audio,
    resample_to_canonical,
    resample_waveform,
)

FloatArray = npt.NDArray[np.float32]


def _tone(freq: float, sample_rate: int, seconds: float = 1.0, amp: float = 0.5) -> FloatArray:
    t = np.arange(int(sample_rate * seconds), dtype=np.float64) / sample_rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _dominant_frequency(waveform: FloatArray, sample_rate: int) -> float:
    spectrum = np.abs(np.fft.rfft(waveform.astype(np.float64)))
    freqs = np.fft.rfftfreq(len(waveform), d=1.0 / sample_rate)
    return float(freqs[int(np.argmax(spectrum))])


@pytest.mark.parametrize("source_rate", [8_000, 22_050, 44_100, 48_000])
def test_resamples_to_16k(source_rate: int) -> None:
    """GUIDE section 13: every one of these must convert to 16 kHz."""
    source = _tone(440.0, source_rate)
    out = resample_waveform(source, source_rate, CANONICAL_SAMPLE_RATE)
    expected = int(round(len(source) * CANONICAL_SAMPLE_RATE / source_rate))
    assert abs(len(out) - expected) <= 2
    assert out.dtype == np.float32


def test_8khz_doubles_length() -> None:
    source = _tone(300.0, 8_000, seconds=1.0)
    out = resample_waveform(source, 8_000, 16_000)
    assert abs(len(out) - 16_000) <= 2


def test_already_16k_is_not_resampled() -> None:
    """The same object must come back: no copy, no re-filtering."""
    audio = AudioData(_tone(440.0, 16_000), 16_000)
    assert resample_audio(audio) is audio
    assert resample_to_canonical(audio) is audio


def test_already_16k_waveform_is_unchanged() -> None:
    source = _tone(440.0, 16_000)
    out = resample_waveform(source, 16_000, 16_000)
    assert np.array_equal(out, source)


@pytest.mark.parametrize("source_rate", [8_000, 22_050, 44_100, 48_000])
def test_sine_frequency_is_preserved(source_rate: int) -> None:
    source = _tone(440.0, source_rate, seconds=1.0)
    out = resample_waveform(source, source_rate, 16_000)
    dominant = _dominant_frequency(out, 16_000)
    assert dominant == pytest.approx(440.0, abs=2.0)


def test_stereo_times_axis_is_preserved() -> None:
    left = _tone(440.0, 48_000, seconds=0.5)
    right = _tone(880.0, 48_000, seconds=0.5)
    stereo = np.stack([left, right], axis=1).astype(np.float32)
    out = resample_waveform(stereo, 48_000, 16_000)
    assert out.ndim == 2
    assert out.shape[1] == 2
    assert abs(out.shape[0] - len(left) // 3) <= 2


def test_resample_audio_updates_rate_and_keeps_provenance() -> None:
    audio = AudioData(_tone(440.0, 48_000), 48_000, path=None)
    out = resample_audio(audio)
    assert out.sample_rate == 16_000
    assert out.path == audio.path
    assert out is not audio


def test_resample_audio_preserves_report_reference() -> None:
    from tamil_voice.audio.io import ValidationReport

    report = ValidationReport()
    audio = AudioData(_tone(440.0, 48_000), 48_000, report=report)
    assert resample_audio(audio).report is report


def test_empty_waveform_is_returned_unchanged() -> None:
    empty = np.zeros(0, dtype=np.float32)
    out = resample_waveform(empty, 48_000, 16_000)
    assert out.size == 0


def test_non_positive_rates_rejected() -> None:
    source = _tone(440.0, 16_000)
    with pytest.raises(ValueError):
        resample_waveform(source, 0, 16_000)
    with pytest.raises(ValueError):
        resample_waveform(source, 16_000, 0)


def test_target_rate_is_configurable() -> None:
    source = _tone(440.0, 48_000, seconds=1.0)
    out = resample_audio(AudioData(source, 48_000), target_sample_rate=8_000)
    assert out.sample_rate == 8_000
    assert abs(out.num_frames - 8_000) <= 2

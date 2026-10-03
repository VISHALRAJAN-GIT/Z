from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest

from tamil_voice.audio.io import (
    ISSUE_CLIPPED,
    ISSUE_DC_OFFSET,
    ISSUE_LOW_AMPLITUDE,
    ISSUE_SILENT,
    AudioData,
    ValidationReport,
)
from tamil_voice.audio.normalization import (
    ISSUE_TOO_LOUD,
    LOUDNESS_CLIPPED,
    LOUDNESS_NORMAL,
    LOUDNESS_SILENT,
    LOUDNESS_TOO_LOUD,
    LOUDNESS_TOO_QUIET,
    LoudnessLimits,
    amplitude_to_dbfs,
    analyze_loudness,
    apply_gain,
    dbfs_to_amplitude,
    normalize_peak,
    normalize_rms,
)

FloatArray = npt.NDArray[np.float32]


def _sine(frames: int = 16_000, amp: float = 0.5, sample_rate: int = 16_000) -> FloatArray:
    t = np.arange(frames, dtype=np.float64) / sample_rate
    return (amp * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


def _codes(issues: object) -> set[str]:
    return {issue.code for issue in issues}  # type: ignore[attr-defined]


# --------------------------------------------------------------- dBFS conversions


def test_amplitude_to_dbfs_reference_points() -> None:
    assert amplitude_to_dbfs(1.0) == pytest.approx(0.0)
    assert amplitude_to_dbfs(0.5) == pytest.approx(-6.0206, abs=1e-3)
    assert amplitude_to_dbfs(0.0) == float("-inf")


def test_dbfs_to_amplitude_reference_points() -> None:
    assert dbfs_to_amplitude(0.0) == pytest.approx(1.0)
    assert dbfs_to_amplitude(-6.0206) == pytest.approx(0.5, abs=1e-4)
    assert dbfs_to_amplitude(float("-inf")) == 0.0


def test_dbfs_roundtrip() -> None:
    for amplitude in (0.001, 0.1, 0.5, 0.99):
        assert dbfs_to_amplitude(amplitude_to_dbfs(amplitude)) == pytest.approx(amplitude, rel=1e-9)


# --------------------------------------------------------------------- analysis


def test_clean_tone_is_normal() -> None:
    report = analyze_loudness(_sine(amp=0.5))
    assert report.loudness == LOUDNESS_NORMAL
    assert report.is_normal is True
    assert report.issues == ()
    assert report.peak == pytest.approx(0.5, abs=1e-3)
    assert report.rms == pytest.approx(0.3536, abs=1e-3)
    assert report.peak_dbfs == pytest.approx(-6.02, abs=0.05)
    assert report.crest_factor == pytest.approx(math.sqrt(2.0), abs=0.02)


def test_analysis_does_not_modify_input() -> None:
    data = _sine(amp=0.5)
    before = data.copy()
    analyze_loudness(data)
    assert np.array_equal(data, before)


def test_empty_is_silent() -> None:
    report = analyze_loudness(np.zeros(0, dtype=np.float32))
    assert report.loudness == LOUDNESS_SILENT
    assert report.peak_dbfs == float("-inf")
    assert report.rms_dbfs == float("-inf")


def test_digital_silence_is_reported() -> None:
    report = analyze_loudness(np.zeros(16_000, dtype=np.float32))
    assert report.loudness == LOUDNESS_SILENT
    assert ISSUE_SILENT in _codes(report.issues)


def test_quiet_tone_is_too_quiet() -> None:
    report = analyze_loudness(_sine(amp=1e-3))
    assert report.loudness == LOUDNESS_TOO_QUIET
    assert ISSUE_LOW_AMPLITUDE in _codes(report.issues)


def test_hot_tone_is_too_loud_but_not_clipped() -> None:
    report = analyze_loudness(_sine(amp=0.99))
    assert report.loudness == LOUDNESS_TOO_LOUD
    assert ISSUE_TOO_LOUD in _codes(report.issues)
    assert ISSUE_CLIPPED not in _codes(report.issues)


def test_clipped_square_is_classified_clipped() -> None:
    report = analyze_loudness(np.full(16_000, 1.0, dtype=np.float32))
    assert report.loudness == LOUDNESS_CLIPPED
    assert ISSUE_CLIPPED in _codes(report.issues)
    assert report.clipping_ratio == pytest.approx(1.0)


def test_clipping_overrides_normal_level() -> None:
    """A few ceiling samples on an otherwise normal-level signal must win."""
    data = _sine(amp=0.5)
    data[:8] = 1.0
    report = analyze_loudness(data, LoudnessLimits(clipping_ratio_warn=0.0))
    assert report.loudness == LOUDNESS_CLIPPED


def test_dc_offset_is_reported() -> None:
    report = analyze_loudness(np.full(16_000, 0.3, dtype=np.float32))
    assert ISSUE_DC_OFFSET in _codes(report.issues)


def test_non_finite_rejected() -> None:
    with pytest.raises(ValueError):
        analyze_loudness(np.array([0.1, np.nan, 0.2], dtype=np.float32))
    with pytest.raises(ValueError):
        analyze_loudness(np.array([0.1, np.inf], dtype=np.float32))


def test_limits_are_configurable() -> None:
    report = analyze_loudness(_sine(amp=0.5), LoudnessLimits(too_quiet_rms_dbfs=-1.0))
    assert report.loudness == LOUDNESS_TOO_QUIET


def test_summary_is_human_readable() -> None:
    assert analyze_loudness(_sine(amp=0.5)).summary() == LOUDNESS_NORMAL
    assert LOUDNESS_SILENT in analyze_loudness(np.zeros(1600, dtype=np.float32)).summary()


# ------------------------------------------------------------------------ gain


def test_apply_gain_doubles_amplitude_at_plus_6_db() -> None:
    audio = AudioData(_sine(amp=0.25), 16_000)
    out = apply_gain(audio, 6.0206)
    assert float(np.abs(out.waveform).max()) == pytest.approx(0.5, abs=1e-3)


def test_apply_zero_gain_is_identity() -> None:
    audio = AudioData(_sine(), 16_000)
    assert apply_gain(audio, 0.0) is audio


def test_apply_gain_drops_stale_report() -> None:
    audio = AudioData(_sine(), 16_000, report=ValidationReport())
    out = apply_gain(audio, -3.0)
    assert out.report is None


def test_apply_non_finite_gain_rejected() -> None:
    audio = AudioData(_sine(), 16_000)
    with pytest.raises(ValueError):
        apply_gain(audio, float("nan"))
    with pytest.raises(ValueError):
        apply_gain(audio, float("inf"))


# ----------------------------------------------------------------- normalize_peak


def test_normalize_peak_hits_target() -> None:
    audio = AudioData(_sine(amp=0.25), 16_000)
    out = normalize_peak(audio, target_dbfs=-1.0)
    peak = float(np.abs(out.waveform).max())
    assert amplitude_to_dbfs(peak) == pytest.approx(-1.0, abs=1e-3)


def test_normalize_peak_rejects_non_negative_target() -> None:
    audio = AudioData(_sine(), 16_000)
    with pytest.raises(ValueError):
        normalize_peak(audio, target_dbfs=0.0)
    with pytest.raises(ValueError):
        normalize_peak(audio, target_dbfs=3.0)


def test_normalize_peak_silent_rejected() -> None:
    audio = AudioData(np.zeros(1600, dtype=np.float32), 16_000)
    with pytest.raises(ValueError):
        normalize_peak(audio)


# ----------------------------------------------------------------- normalize_rms


def test_normalize_rms_hits_target() -> None:
    audio = AudioData(_sine(amp=0.5), 16_000)
    out = normalize_rms(audio, target_dbfs=-20.0)
    rms = float(np.sqrt(np.mean(np.square(out.waveform, dtype=np.float64))))
    assert amplitude_to_dbfs(rms) == pytest.approx(-20.0, abs=1e-3)


def test_normalize_rms_silent_rejected() -> None:
    audio = AudioData(np.zeros(1600, dtype=np.float32), 16_000)
    with pytest.raises(ValueError):
        normalize_rms(audio)


def test_normalize_rms_rejects_non_negative_target() -> None:
    audio = AudioData(_sine(), 16_000)
    with pytest.raises(ValueError):
        normalize_rms(audio, target_dbfs=0.0)

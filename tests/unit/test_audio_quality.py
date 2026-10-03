from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
import torch

from tamil_voice.audio.io import ISSUE_SILENT
from tamil_voice.audio.normalization import LOUDNESS_NORMAL, LOUDNESS_SILENT
from tamil_voice.audio.quality import QualityLimits, analyze_quality

FloatArray = npt.NDArray[np.float32]


def _tone(freq: float = 440.0, seconds: float = 1.0, amp: float = 0.5, sr: int = 16_000) -> FloatArray:
    t = np.arange(int(seconds * sr), dtype=np.float64) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _bursty(sr: int = 16_000) -> FloatArray:
    """Half a tone, half low-level noise: a clear signal/noise separation."""
    rng = np.random.default_rng(0)
    tone = _tone(seconds=0.5, amp=0.5, sr=sr)
    noise = (0.001 * rng.standard_normal(sr // 2)).astype(np.float32)
    return np.concatenate([tone, noise])


def test_report_shapes_and_basic_values() -> None:
    report = analyze_quality(_tone(), 16_000)
    assert report.duration == pytest.approx(1.0, abs=1e-6)
    assert report.sample_rate == 16_000
    assert report.num_samples == 16_000
    assert report.loudness == LOUDNESS_NORMAL
    assert report.peak == pytest.approx(0.5, abs=1e-3)
    assert report.rms_dbfs == pytest.approx(-9.03, abs=0.1)


def test_zero_crossing_rate_of_sine() -> None:
    report = analyze_quality(_tone(freq=440.0), 16_000)
    # Two sign changes per period.
    assert report.zero_crossing_rate == pytest.approx(2 * 440.0 / 16_000, abs=2e-3)


def test_spectral_centroid_of_sine_is_the_tone() -> None:
    report = analyze_quality(_tone(freq=440.0), 16_000)
    assert report.spectral_centroid_hz == pytest.approx(440.0, abs=60.0)
    assert report.spectral_rolloff_hz == pytest.approx(440.0, abs=360.0)


def test_tone_is_spectrally_peaky_noise_is_flat() -> None:
    rng = np.random.default_rng(1)
    noise = (0.3 * rng.standard_normal(16_000)).astype(np.float32)
    tone_report = analyze_quality(_tone(amp=0.5), 16_000)
    noise_report = analyze_quality(noise, 16_000)
    assert tone_report.spectral_flatness < 0.2
    assert noise_report.spectral_flatness > tone_report.spectral_flatness
    assert noise_report.spectral_centroid_hz > tone_report.spectral_centroid_hz


def test_silence_report() -> None:
    report = analyze_quality(np.zeros(16_000, dtype=np.float32), 16_000)
    assert report.loudness == LOUDNESS_SILENT
    assert report.silence_ratio == pytest.approx(1.0)
    assert report.estimated_snr_db == 0.0
    assert report.zero_crossing_rate == 0.0
    assert ISSUE_SILENT in {i.code for i in report.issues}


def test_silence_ratio_half_silent() -> None:
    signal = np.concatenate([_tone(seconds=0.5), np.zeros(8_000, dtype=np.float32)])
    report = analyze_quality(signal, 16_000)
    assert 0.35 < report.silence_ratio < 0.65


def test_estimated_snr_separates_signal_from_noise() -> None:
    report = analyze_quality(_bursty(), 16_000)
    assert report.estimated_snr_db > 20.0


def test_stationary_tone_has_little_estimated_snr() -> None:
    """A steady tone has no quiet frames, so the estimate collapses toward zero."""
    report = analyze_quality(_tone(), 16_000)
    assert report.estimated_snr_db < 5.0


def test_accepts_torch_tensor() -> None:
    from_array = analyze_quality(_tone(), 16_000)
    from_tensor = analyze_quality(torch.from_numpy(_tone()), 16_000)
    assert from_tensor.rms == pytest.approx(from_array.rms, abs=1e-6)


def test_short_clip_has_undefined_spectral_stats() -> None:
    report = analyze_quality(_tone(seconds=0.01), 16_000)  # 160 samples < n_fft // 2
    assert report.spectral_centroid_hz == 0.0
    assert report.spectral_rolloff_hz == 0.0
    assert np.isfinite(report.estimated_snr_db)


def test_to_dict_has_all_fields() -> None:
    report = analyze_quality(_tone(), 16_000)
    dumped = report.to_dict()
    assert set(dumped) == {
        "duration",
        "sample_rate",
        "num_samples",
        "peak",
        "rms",
        "peak_dbfs",
        "rms_dbfs",
        "crest_factor",
        "clipping_ratio",
        "dc_offset",
        "loudness",
        "estimated_snr_db",
        "silence_ratio",
        "zero_crossing_rate",
        "spectral_centroid_hz",
        "spectral_bandwidth_hz",
        "spectral_rolloff_hz",
        "spectral_flatness",
    }
    assert isinstance(dumped["loudness"], str)


def test_summary_mentions_loudness() -> None:
    assert LOUDNESS_NORMAL in analyze_quality(_tone(), 16_000).summary()


def test_custom_limits_are_honoured() -> None:
    report = analyze_quality(
        _tone(amp=1e-2), 16_000, limits=QualityLimits(silence_rms_dbfs=-1.0)
    )
    assert report.silence_ratio == pytest.approx(1.0)


def test_invalid_inputs_rejected() -> None:
    with pytest.raises(ValueError):
        analyze_quality(_tone(), 0)
    with pytest.raises(ValueError):
        analyze_quality(np.zeros((100, 2), dtype=np.float32), 16_000)
    with pytest.raises(ValueError):
        analyze_quality(np.array([0.1, np.nan, 0.2], dtype=np.float32), 16_000)

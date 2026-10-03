from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from tamil_voice.vad.detector import (
    VadConfig,
    _apply_hangover,
    _median_filter,
    _remove_short_runs,
    detect_speech,
)

FloatArray = npt.NDArray[np.float32]


def _tone(seconds: float = 1.0, freq: float = 300.0, amp: float = 0.4, sr: int = 16_000) -> FloatArray:
    t = np.arange(int(seconds * sr), dtype=np.float64) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _embedded(signal: FloatArray, lead: float = 0.5, trail: float = 0.5, sr: int = 16_000) -> FloatArray:
    silence = np.zeros(int(sr * lead), dtype=np.float32)
    tail = np.zeros(int(sr * trail), dtype=np.float32)
    return np.concatenate([silence, signal, tail])


# ------------------------------------------------------------------- helpers


def test_hangover_extends_to_the_right() -> None:
    mask = np.zeros(20, dtype=bool)
    mask[5] = True
    out = _apply_hangover(mask, 3)
    assert out[:5].sum() == 0
    assert out[5:9].all()
    assert not out[9:].any()


def test_hangover_zero_is_identity() -> None:
    mask = np.array([True, False, True])
    assert _apply_hangover(mask, 0) is mask


def test_median_filter_removes_single_frame_flicker() -> None:
    mask = np.array([False, False, True, False, False])
    assert not _median_filter(mask, 3).any()


def test_median_filter_keeps_sustained_run() -> None:
    mask = np.array([False, True, True, True, False])
    assert _median_filter(mask, 3).sum() == 3


def test_remove_short_runs() -> None:
    mask = np.array([True, True, False, True, False, True, True, True])
    out = _remove_short_runs(mask, 3)
    assert not out[:4].any()  # run of 2 and isolated frame are dropped
    assert out[5:8].all()  # run of 3 survives


# ----------------------------------------------------------------- detection


def test_silence_has_no_speech() -> None:
    result = detect_speech(np.zeros(16_000, dtype=np.float32))
    assert result.is_empty
    assert result.speech_ratio == 0.0


def test_embedded_tone_is_detected_in_place() -> None:
    result = detect_speech(_embedded(_tone(seconds=1.0)))
    assert not result.is_empty
    assert not result.mask[:20].any()  # first 0.2 s is silence
    assert not result.mask[-20:].any()  # last 0.2 s is silence
    speech_times = result.frame_times[result.mask]
    assert speech_times.min() < 0.6
    assert speech_times.max() > 0.9


def test_frame_times_step_by_hop() -> None:
    result = detect_speech(_embedded(_tone()))
    assert np.allclose(np.diff(result.frame_times), result.config.hop_seconds)


def test_spectral_gate_rejects_broadband_noise() -> None:
    rng = np.random.default_rng(0)
    noise = (0.3 * rng.standard_normal(16_000)).astype(np.float32)
    fixture = _embedded(noise)

    with_spectral = detect_speech(fixture, VadConfig(use_spectral=True))
    energy_only = detect_speech(fixture, VadConfig(use_spectral=False))

    assert energy_only.speech_ratio > 0.3  # loud noise passes the energy gate
    assert with_spectral.speech_ratio < energy_only.speech_ratio
    assert with_spectral.speech_ratio < 0.15  # but not the flatness gate


def test_flat_noise_flatness_is_high_tone_is_low() -> None:
    rng = np.random.default_rng(1)
    noise = _embedded((0.3 * rng.standard_normal(16_000)).astype(np.float32))
    tone = _embedded(_tone())

    def signal_flatness(waveform: FloatArray) -> float:
        result = detect_speech(waveform)
        assert result.flatness is not None
        inside = (result.frame_times >= 0.6) & (result.frame_times <= 1.4)
        return float(np.median(result.flatness[inside]))

    assert signal_flatness(noise) > 0.45
    assert signal_flatness(tone) < 0.1


def test_too_short_rejected() -> None:
    with pytest.raises(ValueError):
        detect_speech(np.zeros(100, dtype=np.float32))


def test_empty_and_2d_rejected() -> None:
    with pytest.raises(ValueError):
        detect_speech(np.zeros(0, dtype=np.float32))
    with pytest.raises(ValueError):
        detect_speech(np.zeros((16_000, 2), dtype=np.float32))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": 0},
        {"frame_seconds": 0.0},
        {"hop_seconds": -1.0},
        {"noise_percentile": 150.0},
        {"flatness_max": 0.0},
        {"smooth_frames": -1},
    ],
)
def test_config_validation(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        VadConfig(**kwargs)  # type: ignore[arg-type]


def test_speech_frames_are_indices() -> None:
    result = detect_speech(_embedded(_tone()))
    frames = result.speech_frames()
    assert frames
    assert all(result.mask[i] for i in frames)
    assert len(frames) == int(result.mask.sum())

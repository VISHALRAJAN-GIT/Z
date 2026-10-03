from __future__ import annotations

import math
from typing import Any

import librosa
import numpy as np
import numpy.typing as npt
import pytest
import torch

from tamil_voice.audio.features import (
    DEFAULT_N_FFT,
    DEFAULT_N_MELS,
    StftConfig,
    get_window,
    hz_to_mel,
    log_mel_spectrogram,
    magnitude_spectrogram,
    mel_filterbank,
    mel_spectrogram,
    mel_to_hz,
    phase_spectrogram,
    power_spectrogram,
    stft,
)

FloatArray = npt.NDArray[np.float32]


def _tone(freq: float = 440.0, seconds: float = 1.0, amp: float = 0.5, sr: int = 16_000) -> FloatArray:
    t = np.arange(int(seconds * sr), dtype=np.float64) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# --------------------------------------------------------------------- config


def test_default_config_matches_criterion_5() -> None:
    cfg = StftConfig()
    assert cfg.n_fft == 512
    assert cfg.win_length == 400  # 25 ms at 16 kHz
    assert cfg.hop_length == 160  # 10 ms at 16 kHz
    assert cfg.n_mels == 80
    assert cfg.sample_rate == 16_000
    assert cfg.n_freqs == 257


def test_config_resolves_nyquist_fmax() -> None:
    assert StftConfig().resolved_fmax == 8_000.0
    assert StftConfig(fmax=4_000.0).resolved_fmax == 4_000.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_fft": 0},
        {"win_length": 600, "n_fft": 512},
        {"win_length": 0},
        {"hop_length": 0},
        {"n_mels": 0},
        {"sample_rate": 0},
        {"fmin": -1.0},
        {"fmax": 9_000.0},
        {"fmin": 5_000.0, "fmax": 4_000.0},
    ],
)
def test_invalid_config_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        StftConfig(**kwargs)


# ----------------------------------------------------------------------- stft


def test_stft_shape_is_time_major() -> None:
    spectrum = stft(_tone(seconds=1.0))
    assert spectrum.shape == (101, 257)  # 1 + 16000 // 160 frames
    assert spectrum.is_complex()


def test_stft_accepts_tensor_and_array_equivalently() -> None:
    data = _tone(seconds=0.25)
    from_array = stft(data)
    from_tensor = stft(torch.from_numpy(data))
    assert torch.allclose(from_array, from_tensor)


def test_power_and_magnitude_agree() -> None:
    spectrum = stft(_tone(seconds=0.25))
    power = power_spectrogram(spectrum)
    magnitude = magnitude_spectrogram(spectrum)
    assert torch.allclose(power, magnitude.square(), atol=1e-6)
    assert bool((power >= 0).all())


def test_phase_within_pi() -> None:
    phase = phase_spectrogram(stft(_tone(seconds=0.25)))
    assert float(phase.abs().max()) <= math.pi + 1e-6


def test_tone_lands_in_expected_frequency_bin() -> None:
    spectrum = stft(_tone(freq=440.0, seconds=1.0))
    magnitude = magnitude_spectrogram(spectrum)
    freqs = torch.linspace(0.0, 8_000.0, 257)
    peak_bin = int(torch.argmax(magnitude, dim=1).median())
    assert float(freqs[peak_bin]) == pytest.approx(440.0, abs=32.0)


def test_istft_round_trip_recovers_signal() -> None:
    signal = _tone(freq=300.0, seconds=0.5, amp=0.4)
    tensor = torch.from_numpy(signal)
    window = get_window("hann", 400, device=tensor.device, dtype=tensor.dtype)
    spectrum = stft(tensor).transpose(-2, -1)
    reconstructed = torch.istft(
        spectrum,
        n_fft=512,
        hop_length=160,
        win_length=400,
        window=window,
        center=True,
        length=tensor.numel(),
    )
    assert torch.allclose(reconstructed, tensor, atol=1e-4)


def test_get_window_supported_and_rejected() -> None:
    hann = get_window("hann", 400, device=torch.device("cpu"), dtype=torch.float32)
    hamming = get_window("hamming", 400, device=torch.device("cpu"), dtype=torch.float32)
    assert hann.shape == (400,) and hamming.shape == (400,)
    assert float(hann.max()) == pytest.approx(1.0, abs=1e-5)
    with pytest.raises(ValueError):
        get_window("blackman", 400, device=torch.device("cpu"), dtype=torch.float32)


def test_empty_and_2d_waveforms_rejected() -> None:
    with pytest.raises(ValueError):
        stft(np.zeros(0, dtype=np.float32))
    with pytest.raises(ValueError):
        stft(np.zeros((100, 2), dtype=np.float32))


def test_too_short_for_centered_stft_rejected() -> None:
    with pytest.raises(ValueError):
        stft(np.zeros(100, dtype=np.float32))


# ------------------------------------------------------------------------ mel


def test_mel_scale_round_trip() -> None:
    hz = torch.tensor([0.0, 100.0, 1000.0, 4000.0, 8000.0], dtype=torch.float64)
    back = mel_to_hz(hz_to_mel(hz))
    assert torch.allclose(back, hz, atol=1e-6)


def test_mel_scale_is_monotonic() -> None:
    hz = torch.linspace(0.0, 8_000.0, 100, dtype=torch.float64)
    mels = hz_to_mel(hz)
    assert bool((mels[1:] > mels[:-1]).all())


def test_mel_scale_matches_librosa_htk() -> None:
    hz = torch.tensor([100.0, 700.0, 2000.0, 6000.0], dtype=torch.float64)
    expected = librosa.hz_to_mel(hz.numpy(), htk=True)
    assert np.allclose(hz_to_mel(hz).numpy(), expected, atol=1e-6)


def test_filterbank_shape_and_contract() -> None:
    bank = mel_filterbank()
    assert bank.shape == (257, 80)
    assert bank.dtype == torch.float32
    assert bool((bank >= 0).all())
    # Every filter is non-empty and no value exceeds a unit peak; individual
    # peaks may sit just below 1.0 when a band centre falls between FFT bins.
    column_max = bank.max(dim=0).values
    assert bool((column_max > 0.0).all())
    assert bool((column_max <= 1.0 + 1e-6).all())
    assert 0.99 < float(bank.max()) <= 1.0


def test_filterbank_matches_librosa_htk() -> None:
    bank = mel_filterbank().numpy()
    expected = librosa.filters.mel(sr=16_000, n_fft=512, n_mels=80, htk=True, norm=None)
    assert np.allclose(bank.T, expected, atol=1e-5)


def test_mel_and_log_mel_shapes() -> None:
    mel = mel_spectrogram(_tone(seconds=1.0))
    log_mel = log_mel_spectrogram(_tone(seconds=1.0))
    assert mel.shape == (101, 80)
    assert log_mel.shape == (101, 80)
    assert bool((mel >= 0).all())
    assert bool(torch.isfinite(log_mel).all())


def test_log_mel_of_silence_is_finite_floor() -> None:
    log_mel = log_mel_spectrogram(np.zeros(16_000, dtype=np.float32), epsilon=1e-6)
    assert bool(torch.isfinite(log_mel).all())
    assert float(log_mel.max()) == pytest.approx(math.log(1e-6), abs=1e-3)


def test_mel_filterbank_argument_is_checked() -> None:
    with pytest.raises(ValueError):
        mel_spectrogram(_tone(seconds=0.25), filterbank=torch.zeros(10, 10))


def test_log_mel_epsilon_must_be_positive() -> None:
    with pytest.raises(ValueError):
        log_mel_spectrogram(_tone(seconds=0.25), epsilon=0.0)


def test_constants_exposed() -> None:
    assert DEFAULT_N_FFT == 512
    assert DEFAULT_N_MELS == 80

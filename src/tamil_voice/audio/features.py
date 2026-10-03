"""STFT, mel filterbank and log-mel features.

GUIDE section 15 asks for an *own* wrapper around the underlying tensor
operations, not a call into a black box: waveform -> windowing -> FFT -> complex
spectrogram, exposing magnitude, phase and power. Section 16 then builds the
feature pipeline waveform -> STFT -> power -> mel -> log -> 80-dimensional.

That is why this module works in ``torch`` tensors rather than NumPy: the same
spectrogram must later feed the enhancement and ASR encoders, and converting
between frameworks at every stage is both wasteful and a source of silent rate
and dtype bugs.

Frame geometry is fixed by EXP-001 criterion 5: a 25 ms frame and a 10 ms hop at
16 kHz, i.e. 400- and 160-sample windows. The FFT size is 512, the next power of
two above the window, with the window zero-padded to it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from ..common.config import CANONICAL_AUDIO, CANONICAL_SAMPLE_RATE

_DEFAULT_WIN_LENGTH, _DEFAULT_HOP_LENGTH = CANONICAL_AUDIO.frame_length(0.025, 0.010)

#: FFT size: the smallest power of two >= the window length.
DEFAULT_N_FFT = 512
DEFAULT_N_MELS = 80
DEFAULT_FMIN = 0.0

_LOG_EPSILON = 1e-6


@dataclass(frozen=True)
class StftConfig:
    """Frame geometry and mel parameters.

    ``fmax`` defaults to the Nyquist frequency for ``sample_rate``. ``win_length``
    must not exceed ``n_fft``; the window is zero-padded to ``n_fft`` before the FFT.
    """

    n_fft: int = DEFAULT_N_FFT
    win_length: int = _DEFAULT_WIN_LENGTH
    hop_length: int = _DEFAULT_HOP_LENGTH
    n_mels: int = DEFAULT_N_MELS
    sample_rate: int = CANONICAL_SAMPLE_RATE
    fmin: float = DEFAULT_FMIN
    fmax: float | None = None
    center: bool = True
    window: str = "hann"

    def __post_init__(self) -> None:
        if self.n_fft <= 0:
            raise ValueError(f"n_fft must be positive, got {self.n_fft}")
        if not (0 < self.win_length <= self.n_fft):
            raise ValueError(f"win_length must be in (0, n_fft], got {self.win_length}")
        if self.hop_length <= 0:
            raise ValueError(f"hop_length must be positive, got {self.hop_length}")
        if self.n_mels <= 0:
            raise ValueError(f"n_mels must be positive, got {self.n_mels}")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.fmin < 0 or self.resolved_fmax > self.sample_rate / 2:
            raise ValueError(
                f"mel band [{self.fmin}, {self.resolved_fmax}] outside [0, {self.sample_rate / 2}]"
            )
        if self.fmin >= self.resolved_fmax:
            raise ValueError(f"fmin ({self.fmin}) must be below fmax ({self.resolved_fmax})")

    @property
    def resolved_fmax(self) -> float:
        return float(self.fmax) if self.fmax is not None else self.sample_rate / 2.0

    @property
    def n_freqs(self) -> int:
        """Number of one-sided frequency bins in the STFT."""
        return self.n_fft // 2 + 1


def _as_waveform_tensor(waveform: npt.ArrayLike | Tensor) -> Tensor:
    if isinstance(waveform, Tensor):
        tensor = waveform.detach().to(dtype=torch.float32)
    else:
        array = np.ascontiguousarray(waveform, dtype=np.float32)
        tensor = torch.from_numpy(array)
    if tensor.ndim != 1:
        raise ValueError(f"expected a 1-D waveform, got shape {tuple(tensor.shape)}")
    if tensor.numel() == 0:
        raise ValueError("waveform is empty")
    return tensor


def get_window(name: str, length: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Return a periodic analysis window of ``length`` samples."""
    if length <= 0:
        raise ValueError(f"window length must be positive, got {length}")
    if name == "hann":
        return torch.hann_window(length, periodic=True, device=device, dtype=dtype)
    if name == "hamming":
        return torch.hamming_window(length, periodic=True, device=device, dtype=dtype)
    raise ValueError(f"unsupported window: {name!r}")


def stft(waveform: npt.ArrayLike | Tensor, config: StftConfig | None = None) -> Tensor:
    """Complex STFT, returned time-major as ``(n_frames, n_freqs)``.

    Time-major ordering matches librosa's feature convention and the ``(n_frames,
    n_mels)`` shape required by EXP-001 criterion 5, so callers never have to
    remember which axis is which.
    """
    cfg = config or StftConfig()
    signal = _as_waveform_tensor(waveform)
    if cfg.center and signal.numel() < cfg.n_fft // 2:
        raise ValueError(
            f"waveform of {signal.numel()} samples is too short for centered STFT "
            f"with n_fft={cfg.n_fft}"
        )
    window = get_window(cfg.window, cfg.win_length, device=signal.device, dtype=signal.dtype)
    spectrum = torch.stft(
        signal,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=window,
        center=cfg.center,
        pad_mode="reflect",
        return_complex=True,
    )
    return spectrum.transpose(-2, -1)


def magnitude_spectrogram(spectrum: Tensor) -> Tensor:
    """Magnitude of a complex spectrogram."""
    return torch.abs(spectrum)


def power_spectrogram(spectrum: Tensor) -> Tensor:
    """Power (magnitude squared) of a complex spectrogram."""
    return spectrum.real.square() + spectrum.imag.square()


def phase_spectrogram(spectrum: Tensor) -> Tensor:
    """Phase angle of a complex spectrogram, in radians in ``[-pi, pi]``."""
    return torch.angle(spectrum)


def frame_rms(
    waveform: npt.ArrayLike | Tensor,
    frame_length: int = _DEFAULT_WIN_LENGTH,
    hop_length: int = _DEFAULT_HOP_LENGTH,
) -> npt.NDArray[np.float64]:
    """Per-frame RMS of a 1-D signal as float64.

    Frames step by ``hop_length``; trailing samples shorter than one frame are
    dropped. A signal shorter than one frame yields a single frame, so callers
    never get an empty result for non-empty audio.
    """
    signal = _as_waveform_tensor(waveform)
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length and hop_length must be positive")
    if signal.numel() < frame_length:
        return np.array([float(signal.square().mean().sqrt())], dtype=np.float64)
    windows = signal.unfold(0, frame_length, hop_length)
    return windows.square().mean(dim=1).sqrt().to(dtype=torch.float64).numpy()


def hz_to_mel(frequencies: Tensor, *, htk: bool = True) -> Tensor:
    """Convert Hz to the mel scale (HTK formula by default)."""
    if htk:
        return 2595.0 * torch.log10(1.0 + frequencies / 700.0)
    # Slaney's alternative formula, for comparison against librosa defaults.
    f_min, f_sp = 0.0, 200.0 / 3.0
    mels = (frequencies - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    log_step = math.log(6.4) / 27.0
    if bool(torch.any(frequencies >= min_log_hz)):
        mels = torch.where(
            frequencies >= min_log_hz,
            min_log_mel + torch.log(frequencies / min_log_hz) / log_step,
            mels,
        )
    return mels


def mel_to_hz(mels: Tensor, *, htk: bool = True) -> Tensor:
    """Inverse of :func:`hz_to_mel`."""
    if htk:
        return 700.0 * (torch.pow(10.0, mels / 2595.0) - 1.0)
    f_min, f_sp = 0.0, 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    log_step = math.log(6.4) / 27.0
    freqs = f_min + f_sp * mels
    return torch.where(
        mels >= min_log_mel,
        min_log_hz * torch.exp(log_step * (mels - min_log_mel)),
        freqs,
    )


def mel_filterbank(config: StftConfig | None = None) -> Tensor:
    """Triangular mel filterbank, shape ``(n_freqs, n_mels)``.

    Unit-peak triangles over the mel-spaced band edges, which is the unnormalised
    form (librosa's ``norm=None``). Values are float32.
    """
    cfg = config or StftConfig()
    fft_freqs = torch.linspace(0.0, cfg.sample_rate / 2.0, cfg.n_freqs, dtype=torch.float64)
    mel_points = torch.linspace(
        float(hz_to_mel(torch.tensor(cfg.fmin, dtype=torch.float64))),
        float(hz_to_mel(torch.tensor(cfg.resolved_fmax, dtype=torch.float64))),
        cfg.n_mels + 2,
        dtype=torch.float64,
    )
    hz_points = mel_to_hz(mel_points)

    left = hz_points[:-2].unsqueeze(1)
    center = hz_points[1:-1].unsqueeze(1)
    right = hz_points[2:].unsqueeze(1)
    freqs = fft_freqs.unsqueeze(0)

    rising = (freqs - left) / (center - left)
    falling = (right - freqs) / (right - center)
    triangles = torch.clamp(torch.minimum(rising, falling), min=0.0)
    return triangles.transpose(0, 1).to(dtype=torch.float32).contiguous()


def mel_spectrogram(
    waveform: npt.ArrayLike | Tensor,
    config: StftConfig | None = None,
    filterbank: Tensor | None = None,
) -> Tensor:
    """Power mel spectrogram, shape ``(n_frames, n_mels)``."""
    cfg = config or StftConfig()
    power = power_spectrogram(stft(waveform, cfg))
    bank = filterbank if filterbank is not None else mel_filterbank(cfg)
    if bank.shape != (cfg.n_freqs, cfg.n_mels):
        raise ValueError(
            f"filterbank shape {tuple(bank.shape)} does not match (n_freqs, n_mels) "
            f"= {(cfg.n_freqs, cfg.n_mels)}"
        )
    return power @ bank.to(dtype=power.dtype)


def log_mel_spectrogram(
    waveform: npt.ArrayLike | Tensor,
    config: StftConfig | None = None,
    filterbank: Tensor | None = None,
    *,
    epsilon: float = _LOG_EPSILON,
) -> Tensor:
    """Natural-log mel spectrogram, shape ``(n_frames, n_mels)``.

    ``epsilon`` floors the mel energies before the log so silence yields a large
    negative but finite value instead of ``-inf``.
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    mel = mel_spectrogram(waveform, config, filterbank)
    return torch.log(mel + epsilon)


__all__ = [
    "DEFAULT_FMIN",
    "DEFAULT_N_FFT",
    "DEFAULT_N_MELS",
    "StftConfig",
    "frame_rms",
    "get_window",
    "hz_to_mel",
    "log_mel_spectrogram",
    "magnitude_spectrogram",
    "mel_filterbank",
    "mel_spectrogram",
    "mel_to_hz",
    "phase_spectrogram",
    "power_spectrogram",
    "stft",
]

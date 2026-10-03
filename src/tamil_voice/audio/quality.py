"""Audio quality analysis — the diagnostic report GUIDE section 18 asks for.

When ASR fails, the useful question is rarely "what is the loss?" but "what was
wrong with the audio?". This module answers the second one. It aggregates the
level metrics from :mod:`normalization` with the spectral view from
:mod:`features` into a single :class:`QualityReport` covering duration, RMS, peak,
crest factor, clipping ratio, an *estimated* SNR, silence ratio, zero-crossing
rate and spectral statistics.

Two honesty notes are load-bearing:

* SNR here is **estimated**, not measured. There is no clean reference signal, so
  the noise floor is taken as a low percentile of frame energies and the signal as
  a high percentile. It is a diagnostic, not a calibration.
* Nothing is repaired. The report describes the audio that exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import torch

from .features import StftConfig, magnitude_spectrogram, stft
from .io import AudioIssue, FloatArray
from .normalization import LoudnessLimits, analyze_loudness

_SILENCE_FLOOR = 1e-12


@dataclass(frozen=True)
class QualityLimits:
    """Thresholds for the quality report. Conservative defaults, not tuned."""

    frame_seconds: float = 0.025
    hop_seconds: float = 0.010
    #: Per-frame RMS at or below this counts as silence for ``silence_ratio``.
    silence_rms_dbfs: float = -50.0
    #: Percentile of frame energies taken as the noise floor.
    noise_percentile: float = 10.0
    #: Percentile of frame energies taken as the signal level.
    signal_percentile: float = 90.0
    #: Cumulative-magnitude fraction used for the spectral roll-off.
    rolloff_fraction: float = 0.85


@dataclass(frozen=True)
class QualityReport:
    """A description of a recording's quality. Descriptive only."""

    duration: float
    sample_rate: int
    num_samples: int
    peak: float
    rms: float
    peak_dbfs: float
    rms_dbfs: float
    crest_factor: float
    clipping_ratio: float
    dc_offset: float
    loudness: str
    estimated_snr_db: float
    silence_ratio: float
    zero_crossing_rate: float
    spectral_centroid_hz: float
    spectral_bandwidth_hz: float
    spectral_rolloff_hz: float
    spectral_flatness: float
    issues: tuple[AudioIssue, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "num_samples": self.num_samples,
            "peak": self.peak,
            "rms": self.rms,
            "peak_dbfs": self.peak_dbfs,
            "rms_dbfs": self.rms_dbfs,
            "crest_factor": self.crest_factor,
            "clipping_ratio": self.clipping_ratio,
            "dc_offset": self.dc_offset,
            "loudness": self.loudness,
            "estimated_snr_db": self.estimated_snr_db,
            "silence_ratio": self.silence_ratio,
            "zero_crossing_rate": self.zero_crossing_rate,
            "spectral_centroid_hz": self.spectral_centroid_hz,
            "spectral_bandwidth_hz": self.spectral_bandwidth_hz,
            "spectral_rolloff_hz": self.spectral_rolloff_hz,
            "spectral_flatness": self.spectral_flatness,
        }

    def summary(self) -> str:
        base = (
            f"{self.duration:.2f}s @ {self.sample_rate} Hz | {self.loudness} "
            f"| rms {self.rms_dbfs:.1f} dBFS | est SNR {self.estimated_snr_db:.1f} dB "
            f"| silence {self.silence_ratio * 100:.1f}%"
        )
        if not self.issues:
            return base
        return base + " | " + "; ".join(f"{i.code}({i.severity})" for i in self.issues)


def _frame_rms(data: FloatArray, frame_length: int, hop_length: int) -> npt.NDArray[np.float64]:
    if data.size == 0:
        return np.zeros(0, dtype=np.float64)
    if data.size < frame_length:
        level = np.sqrt(np.mean(np.square(data, dtype=np.float64)))
        return np.array([level], dtype=np.float64)
    windows = np.lib.stride_tricks.sliding_window_view(data, frame_length)[::hop_length]
    return np.sqrt(np.mean(np.square(windows, dtype=np.float64), axis=1))


def _dbfs_array(amplitudes: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Element-wise dBFS, with ``-inf`` for zero amplitudes."""
    out = np.full(amplitudes.shape, -np.inf, dtype=np.float64)
    positive = amplitudes > 0.0
    out[positive] = 20.0 * np.log10(amplitudes[positive])
    return out


def _estimated_snr_db(frame_rms: npt.NDArray[np.float64], limits: QualityLimits) -> float:
    if frame_rms.size == 0:
        return 0.0
    frame_dbfs = _dbfs_array(frame_rms)
    finite = frame_dbfs[np.isfinite(frame_dbfs)]
    if finite.size == 0:
        return 0.0
    noise = float(np.percentile(finite, limits.noise_percentile))
    signal = float(np.percentile(finite, limits.signal_percentile))
    return max(signal - noise, 0.0)


def _spectral_statistics(
    data: FloatArray, sample_rate: int, config: StftConfig, limits: QualityLimits
) -> tuple[float, float, float, float]:
    """Return (centroid Hz, bandwidth Hz, rolloff Hz, flatness), averaged over frames."""
    spectrum = magnitude_spectrogram(stft(torch.from_numpy(data), config))
    freqs = torch.linspace(0.0, sample_rate / 2.0, config.n_freqs, dtype=spectrum.dtype)

    total = spectrum.sum(dim=1, keepdim=True)
    safe_total = total.clamp(min=_SILENCE_FLOOR)
    centroid = (spectrum * freqs).sum(dim=1) / safe_total.squeeze(1)

    deviations = (freqs.unsqueeze(0) - centroid.unsqueeze(1)).square()
    bandwidth = torch.sqrt((spectrum * deviations).sum(dim=1) / safe_total.squeeze(1))

    cumulative = torch.cumsum(spectrum, dim=1)
    threshold = limits.rolloff_fraction * cumulative[:, -1:]
    rolloff_index = torch.argmax((cumulative >= threshold).to(torch.int64), dim=1)
    rolloff = freqs[rolloff_index]

    geometric = torch.exp(torch.log(spectrum.clamp(min=_SILENCE_FLOOR)).mean(dim=1))
    arithmetic = spectrum.mean(dim=1).clamp(min=_SILENCE_FLOOR)
    flatness = geometric / arithmetic

    return (
        float(centroid.mean()),
        float(bandwidth.mean()),
        float(rolloff.mean()),
        float(flatness.mean()),
    )


def analyze_quality(
    waveform: npt.ArrayLike | torch.Tensor,
    sample_rate: int,
    *,
    config: StftConfig | None = None,
    limits: QualityLimits | None = None,
    loudness_limits: LoudnessLimits | None = None,
) -> QualityReport:
    """Build a :class:`QualityReport` for a waveform.

    Raises:
        ValueError: ``sample_rate`` is not positive, or the waveform contains
            non-finite samples (via :func:`normalization.analyze_loudness`).
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    limits = limits or QualityLimits()
    cfg = config or StftConfig(sample_rate=sample_rate)

    data: FloatArray
    if isinstance(waveform, torch.Tensor):
        data = waveform.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        data = np.ascontiguousarray(waveform, dtype=np.float32)
    if data.ndim != 1:
        raise ValueError(f"expected a 1-D waveform, got shape {data.shape}")

    loudness = analyze_loudness(data, loudness_limits)

    frame_length = int(round(limits.frame_seconds * sample_rate))
    hop_length = int(round(limits.hop_seconds * sample_rate))
    frame_rms = _frame_rms(data, frame_length, hop_length)

    if frame_rms.size:
        frame_dbfs = _dbfs_array(frame_rms)
        silence_ratio = float(np.mean(frame_dbfs <= limits.silence_rms_dbfs))
    else:
        silence_ratio = 1.0
    estimated_snr_db = _estimated_snr_db(frame_rms, limits)

    if data.size >= 2:
        crossings = int(np.count_nonzero(np.diff(np.signbit(data))))
        zero_crossing_rate = crossings / float(data.size)
    else:
        zero_crossing_rate = 0.0

    if data.size >= cfg.n_fft // 2 and data.size > 0:
        centroid, bandwidth, rolloff, flatness = _spectral_statistics(data, sample_rate, cfg, limits)
    else:
        # Too short for the configured STFT; spectral numbers are undefined.
        centroid = bandwidth = rolloff = flatness = 0.0

    return QualityReport(
        duration=data.size / float(sample_rate),
        sample_rate=sample_rate,
        num_samples=int(data.size),
        peak=loudness.peak,
        rms=loudness.rms,
        peak_dbfs=loudness.peak_dbfs,
        rms_dbfs=loudness.rms_dbfs,
        crest_factor=loudness.crest_factor,
        clipping_ratio=loudness.clipping_ratio,
        dc_offset=loudness.dc_offset,
        loudness=loudness.loudness,
        estimated_snr_db=estimated_snr_db,
        silence_ratio=silence_ratio,
        zero_crossing_rate=zero_crossing_rate,
        spectral_centroid_hz=centroid,
        spectral_bandwidth_hz=bandwidth,
        spectral_rolloff_hz=rolloff,
        spectral_flatness=flatness,
        issues=loudness.issues,
    )


__all__ = [
    "QualityLimits",
    "QualityReport",
    "analyze_quality",
]

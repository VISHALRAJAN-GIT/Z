"""Voice activity detection: energy + spectral flatness + smoothing + hangover.

GUIDE section 17 asks for a first VAD that is explicitly *not* neural, built from
four ingredients:

* **energy** — a frame is a candidate if it is clearly above the estimated noise
  floor, rather than above a fixed level, so quiet recordings still work;
* **spectral characteristics** — speech is spectrally structured (peaky, low
  flatness), whereas broadband noise is flat, so a loud hiss does not masquerade
  as speech;
* **smoothing** — a median filter over the frame decisions removes single-frame
  flicker;
* **hangover** — speech stays "on" for a short tail after the energy drops, so
  trailing consonants are not chopped off.

All frames come from one STFT grid, so the energy and spectral views share the
same frame count and stay aligned. The energy value is therefore a relative
windowed-power level, not a calibrated dBFS reading; it is used comparatively.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch

from ..audio.features import (
    StftConfig,
    power_spectrogram,
    stft,
)
from ..common.config import CANONICAL_SAMPLE_RATE

_EPSILON = 1e-12


@dataclass(frozen=True)
class VadConfig:
    """Parameters for :func:`detect_speech`. Defaults are reasoned, not tuned."""

    sample_rate: int = CANONICAL_SAMPLE_RATE
    frame_seconds: float = 0.025
    hop_seconds: float = 0.010
    #: A frame is a speech candidate if it exceeds the noise floor by this much.
    energy_margin_db: float = 12.0
    #: Percentile of frame energies taken as the noise floor.
    noise_percentile: float = 10.0
    #: Use the spectral flatness gate in addition to energy.
    use_spectral: bool = True
    #: Frames flatter than this are treated as noise, not speech.
    flatness_max: float = 0.45
    #: Median-filter window (frames) applied to the raw decisions. 1 disables it.
    smooth_frames: int = 5
    #: Keep speech active for this many frames after the decision drops.
    hangover_frames: int = 15
    #: Speech runs shorter than this (frames) are discarded.
    min_speech_frames: int = 3

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.frame_seconds <= 0 or self.hop_seconds <= 0:
            raise ValueError("frame_seconds and hop_seconds must be positive")
        if not 0.0 <= self.noise_percentile <= 100.0:
            raise ValueError(f"noise_percentile must be in [0, 100], got {self.noise_percentile}")
        if self.flatness_max <= 0.0:
            raise ValueError(f"flatness_max must be positive, got {self.flatness_max}")
        if self.smooth_frames < 0 or self.hangover_frames < 0 or self.min_speech_frames < 0:
            raise ValueError("smooth_frames, hangover_frames and min_speech_frames must be >= 0")


@dataclass(frozen=True)
class VadResult:
    """Per-frame decisions plus the evidence behind them."""

    mask: npt.NDArray[np.bool_]
    frame_times: npt.NDArray[np.float64]
    energy_db: npt.NDArray[np.float64]
    flatness: npt.NDArray[np.float64] | None
    threshold_db: float
    noise_floor_db: float
    config: VadConfig

    @property
    def num_frames(self) -> int:
        return int(self.mask.size)

    @property
    def speech_ratio(self) -> float:
        return float(self.mask.mean()) if self.mask.size else 0.0

    @property
    def is_empty(self) -> bool:
        return not bool(self.mask.any())

    def speech_frames(self) -> list[int]:
        """Indices of speech frames."""
        return [int(i) for i in np.flatnonzero(self.mask)]


def _as_waveform(data: npt.ArrayLike | torch.Tensor) -> npt.NDArray[np.float32]:
    if isinstance(data, torch.Tensor):
        array = data.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        array = np.ascontiguousarray(data, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"expected a 1-D waveform, got shape {array.shape}")
    if array.size == 0:
        raise ValueError("waveform is empty")
    return array


def _spectral_flatness(power: torch.Tensor) -> torch.Tensor:
    """Geometric-to-arithmetic mean ratio per frame. Near 1 for white noise."""
    floored = power.clamp(min=_EPSILON)
    geometric = torch.exp(torch.log(floored).mean(dim=1))
    arithmetic = floored.mean(dim=1)
    return geometric / arithmetic


def _median_filter(decisions: npt.NDArray[np.bool_], window: int) -> npt.NDArray[np.bool_]:
    """Majority-vote smoothing over ``window`` frames. Odd windows only."""
    if window <= 1 or decisions.size == 0:
        return decisions
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(decisions.astype(np.float64), pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.median(windows, axis=1) > 0.5


def _apply_hangover(mask: npt.NDArray[np.bool_], hangover: int) -> npt.NDArray[np.bool_]:
    """Extend every speech run ``hangover`` frames to the right."""
    if hangover <= 0 or mask.size == 0:
        return mask
    index = np.arange(mask.size)
    last_speech = np.maximum.accumulate(np.where(mask, index, -1))
    active = last_speech >= 0
    return mask | (active & ((index - last_speech) <= hangover))


def _remove_short_runs(mask: npt.NDArray[np.bool_], min_len: int) -> npt.NDArray[np.bool_]:
    """Drop speech runs shorter than ``min_len`` frames."""
    if min_len <= 1 or mask.size == 0:
        return mask
    out = mask.copy()
    changes = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    for start, end in zip(starts, ends, strict=True):
        if (end - start) < min_len:
            out[start:end] = False
    return out


def detect_speech(
    waveform: npt.ArrayLike | torch.Tensor,
    config: VadConfig | None = None,
) -> VadResult:
    """Detect speech frames in a 1-D waveform.

    Raises:
        ValueError: the waveform is empty, not 1-D, or too short for the configured
            STFT (fewer than ``n_fft // 2`` samples).
    """
    cfg = config or VadConfig()
    data = _as_waveform(waveform)

    stft_config = StftConfig(
        sample_rate=cfg.sample_rate,
        win_length=int(round(cfg.frame_seconds * cfg.sample_rate)),
        hop_length=int(round(cfg.hop_seconds * cfg.sample_rate)),
    )
    if data.size < stft_config.n_fft // 2:
        raise ValueError(
            f"waveform of {data.size} samples is too short for VAD "
            f"(needs >= {stft_config.n_fft // 2})"
        )

    spectrum = stft(torch.from_numpy(data), stft_config)
    power = power_spectrogram(spectrum)
    energy_db = (10.0 * torch.log10(power.sum(dim=1) + _EPSILON)).numpy()

    flatness: npt.NDArray[np.float64] | None = None
    if cfg.use_spectral:
        flatness = _spectral_flatness(power).numpy()

    finite = energy_db[np.isfinite(energy_db)]
    noise_floor = float(np.percentile(finite, cfg.noise_percentile)) if finite.size else float("-inf")
    threshold = noise_floor + cfg.energy_margin_db

    decision = energy_db > threshold
    if flatness is not None:
        decision = decision & (flatness < cfg.flatness_max)

    smoothed = _median_filter(decision, cfg.smooth_frames)
    with_hangover = _apply_hangover(smoothed, cfg.hangover_frames)
    final = _remove_short_runs(with_hangover, cfg.min_speech_frames)

    frame_times = np.arange(final.size, dtype=np.float64) * cfg.hop_seconds
    return VadResult(
        mask=final,
        frame_times=frame_times,
        energy_db=energy_db,
        flatness=flatness,
        threshold_db=threshold,
        noise_floor_db=noise_floor,
        config=cfg,
    )


__all__ = [
    "VadConfig",
    "VadResult",
    "detect_speech",
]

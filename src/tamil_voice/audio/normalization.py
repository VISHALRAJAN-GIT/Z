"""Waveform normalization *analysis*, and opt-in gain.

GUIDE section 14 is explicit: do **not** automatically normalize every recording
to maximum amplitude. Loudness is information. A quiet recording may be a quiet
speaker and a loud one may be a clipped phone capture; flattening both to the same
peak destroys exactly the distinction a downstream model might need.

So this module is, by default, read-only. :func:`analyze_loudness` measures and
classifies — too quiet, normal, too loud, clipped — and changes nothing. The gain
functions exist for callers who have *decided* to change the level, and they must
be called by name. Nothing here fires implicitly.

Every amplitude metric is also reported in dBFS, because that is the scale these
thresholds are actually reasoned about in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .io import (
    ERROR,
    ISSUE_CLIPPED,
    ISSUE_DC_OFFSET,
    ISSUE_LOW_AMPLITUDE,
    ISSUE_SILENT,
    WARNING,
    AudioData,
    AudioIssue,
    FloatArray,
)

#: The recording is all but empty.
LOUDNESS_SILENT = "silent"
#: Audible, but so far below nominal that it is likely to hurt a model.
LOUDNESS_TOO_QUIET = "too_quiet"
#: Within the useful range.
LOUDNESS_NORMAL = "normal"
#: Peaking close enough to full scale that distortion is likely.
LOUDNESS_TOO_LOUD = "too_loud"
#: Samples are actually touching the ceiling; damage already happened.
LOUDNESS_CLIPPED = "clipped"

#: Loudness classification returned when the recording is too loud but not clipped.
ISSUE_TOO_LOUD = "too_loud"


def amplitude_to_dbfs(amplitude: float) -> float:
    """Convert a linear amplitude to dBFS. Zero amplitude is ``-inf``."""
    if amplitude <= 0.0:
        return float("-inf")
    return 20.0 * math.log10(amplitude)


def dbfs_to_amplitude(dbfs: float) -> float:
    """Convert dBFS to a linear amplitude. ``-inf`` maps to 0."""
    return float(10.0 ** (dbfs / 20.0))


@dataclass(frozen=True)
class LoudnessLimits:
    """Thresholds for loudness classification. Conservative defaults, not tuned."""

    #: Sample values at or above this are counted as clipped.
    clipping_threshold: float = 0.999
    #: Fraction of clipped samples above which the recording is called clipped.
    clipping_ratio_warn: float = 0.001
    #: RMS at or below this is treated as silence.
    silence_rms_dbfs: float = -80.0
    #: RMS below this (but above silence) is "too quiet".
    too_quiet_rms_dbfs: float = -50.0
    #: Peak above this (but not clipped) is "too loud".
    too_loud_peak_dbfs: float = -1.0
    #: |mean| above this suggests a DC offset / bad ADC.
    dc_offset_warn: float = 0.05


@dataclass(frozen=True)
class LoudnessReport:
    """The result of :func:`analyze_loudness`. Descriptive only; nothing is altered."""

    peak: float
    rms: float
    dc_offset: float
    crest_factor: float
    peak_dbfs: float
    rms_dbfs: float
    clipping_ratio: float
    loudness: str
    issues: tuple[AudioIssue, ...] = field(default_factory=tuple)

    @property
    def is_normal(self) -> bool:
        return self.loudness == LOUDNESS_NORMAL

    def summary(self) -> str:
        if not self.issues:
            return self.loudness
        return f"{self.loudness}: " + "; ".join(f"{i.code}({i.severity})" for i in self.issues)


def analyze_loudness(
    waveform: FloatArray,
    limits: LoudnessLimits | None = None,
) -> LoudnessReport:
    """Measure peak, RMS, DC offset, crest factor and clipping, and classify.

    Read-only: the waveform is inspected, never rewritten. Non-finite samples raise
    ``ValueError`` rather than silently producing ``nan`` metrics — by this point in
    the pipeline ``io.validate_audio`` should already have rejected them.

    Raises:
        ValueError: the waveform contains NaN or infinite samples.
    """
    limits = limits or LoudnessLimits()
    data = np.asarray(waveform, dtype=np.float32)

    if data.size and not bool(np.all(np.isfinite(data))):
        raise ValueError("waveform contains non-finite values; validate it first")

    if data.size == 0:
        return LoudnessReport(
            peak=0.0,
            rms=0.0,
            dc_offset=0.0,
            crest_factor=0.0,
            peak_dbfs=float("-inf"),
            rms_dbfs=float("-inf"),
            clipping_ratio=0.0,
            loudness=LOUDNESS_SILENT,
        )

    absolute = np.abs(data)
    peak = float(absolute.max())
    rms = float(np.sqrt(np.mean(np.square(data, dtype=np.float64))))
    dc_offset = float(np.mean(data, dtype=np.float64))
    crest_factor = peak / rms if rms > 0.0 else 0.0
    peak_dbfs = amplitude_to_dbfs(peak)
    rms_dbfs = amplitude_to_dbfs(rms)

    clipped = int(np.count_nonzero(absolute >= limits.clipping_threshold))
    clipping_ratio = clipped / float(data.size)

    issues: list[AudioIssue] = []
    if clipping_ratio > limits.clipping_ratio_warn:
        issues.append(
            AudioIssue(
                ISSUE_CLIPPED,
                WARNING,
                f"{clipping_ratio * 100:.3f}% of samples at or above {limits.clipping_threshold}",
                clipping_ratio,
            )
        )

    if rms_dbfs <= limits.silence_rms_dbfs:
        loudness = LOUDNESS_SILENT
        issues.append(
            AudioIssue(ISSUE_SILENT, ERROR, f"effectively silent (rms {rms_dbfs:.1f} dBFS)", rms_dbfs)
        )
    elif rms_dbfs < limits.too_quiet_rms_dbfs:
        loudness = LOUDNESS_TOO_QUIET
        issues.append(
            AudioIssue(
                ISSUE_LOW_AMPLITUDE,
                WARNING,
                f"very low level (rms {rms_dbfs:.1f} dBFS)",
                rms_dbfs,
            )
        )
    elif peak_dbfs > limits.too_loud_peak_dbfs:
        loudness = LOUDNESS_TOO_LOUD
        issues.append(
            AudioIssue(ISSUE_TOO_LOUD, WARNING, f"peaking high (peak {peak_dbfs:.1f} dBFS)", peak_dbfs)
        )
    else:
        loudness = LOUDNESS_NORMAL

    # Clipping overrides the level-based verdict: if samples touched the ceiling,
    # that is the fact that matters, regardless of the average level.
    if clipping_ratio > limits.clipping_ratio_warn:
        loudness = LOUDNESS_CLIPPED

    if abs(dc_offset) > limits.dc_offset_warn:
        issues.append(
            AudioIssue(ISSUE_DC_OFFSET, WARNING, f"DC offset {dc_offset:+.4f}", dc_offset)
        )

    return LoudnessReport(
        peak=peak,
        rms=rms,
        dc_offset=dc_offset,
        crest_factor=crest_factor,
        peak_dbfs=peak_dbfs,
        rms_dbfs=rms_dbfs,
        clipping_ratio=clipping_ratio,
        loudness=loudness,
        issues=tuple(issues),
    )


def _require_finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")


def apply_gain(audio: AudioData, gain_db: float) -> AudioData:
    """Scale the waveform by ``gain_db`` decibels. Opt-in, never automatic.

    The returned audio carries ``report=None``: the original validation/loudness
    metrics described the old amplitude and no longer apply. A zero gain is a true
    no-op — the same object is returned.

    Raises:
        ValueError: ``gain_db`` is not finite.
    """
    _require_finite(gain_db, "gain_db")
    if gain_db == 0.0:
        return audio
    factor = dbfs_to_amplitude(gain_db)
    scaled = np.ascontiguousarray(audio.waveform * factor, dtype=np.float32)
    return AudioData(scaled, audio.sample_rate, audio.path, report=None)


def _peak_to_target_gain(waveform: FloatArray, target_dbfs: float) -> float:
    peak = float(np.abs(waveform).max()) if waveform.size else 0.0
    if peak <= 0.0:
        raise ValueError("cannot normalize silent audio: peak is zero")
    return target_dbfs - amplitude_to_dbfs(peak)


def normalize_peak(audio: AudioData, target_dbfs: float = -1.0) -> AudioData:
    """Bring the peak to ``target_dbfs``. Explicit, opt-in gain — not applied by default.

    Raises:
        ValueError: ``target_dbfs`` is not negative, or the audio is silent.
    """
    _require_finite(target_dbfs, "target_dbfs")
    if target_dbfs >= 0.0:
        raise ValueError(f"target peak must be below 0 dBFS, got {target_dbfs}")
    return apply_gain(audio, _peak_to_target_gain(audio.waveform, target_dbfs))


def normalize_rms(audio: AudioData, target_dbfs: float = -20.0) -> AudioData:
    """Bring the RMS to ``target_dbfs``. Explicit, opt-in gain — not applied by default.

    Raises:
        ValueError: ``target_dbfs`` is not negative, or the audio is silent.
    """
    _require_finite(target_dbfs, "target_dbfs")
    if target_dbfs >= 0.0:
        raise ValueError(f"target RMS must be below 0 dBFS, got {target_dbfs}")
    rms = float(np.sqrt(np.mean(np.square(audio.waveform, dtype=np.float64)))) if audio.waveform.size else 0.0
    if rms <= 0.0:
        raise ValueError("cannot normalize silent audio: rms is zero")
    return apply_gain(audio, target_dbfs - amplitude_to_dbfs(rms))


__all__ = [
    "LOUDNESS_CLIPPED",
    "LOUDNESS_NORMAL",
    "LOUDNESS_SILENT",
    "LOUDNESS_TOO_LOUD",
    "LOUDNESS_TOO_QUIET",
    "LoudnessLimits",
    "LoudnessReport",
    "amplitude_to_dbfs",
    "analyze_loudness",
    "apply_gain",
    "dbfs_to_amplitude",
    "normalize_peak",
    "normalize_rms",
]

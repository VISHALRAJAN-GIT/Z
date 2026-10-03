"""Audio loading and validation.

The canonical representation is a float32 waveform, mono by default, at the
recording's *native* sample rate. Conversion to the canonical 16 kHz belongs to
``resampling.py``; this module deliberately does not resample, so that loading and
rate conversion stay independently testable.

Validation never repairs audio. It reports what is wrong — clipping, silence,
NaN, absurd duration — and lets the caller decide. Silently "fixing" a recording
is how a broken dataset becomes a mysterious ASR failure three phases later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import soundfile as sf

from ..common.config import CANONICAL_SAMPLE_RATE

FloatArray = npt.NDArray[np.float32]

#: Lowest sample rate accepted. GUIDE section 13 requires 8 kHz input to work, so
#: this is the floor; anything lower cannot be resampled to 16 kHz sensibly.
MIN_SUPPORTED_SAMPLE_RATE = 8_000

#: Beyond this, the rate is almost certainly a corrupt header rather than audio.
MAX_SUPPORTED_SAMPLE_RATE = 384_000


class AudioError(Exception):
    """Base class for audio problems."""


class AudioLoadError(AudioError):
    """The file could not be read at all: missing, unreadable, or corrupt."""


class AudioValidationError(AudioError):
    """The file decoded, but is unusable for the reasons in ``report``."""

    def __init__(self, message: str, report: ValidationReport) -> None:
        super().__init__(message)
        self.report = report


ISSUE_EMPTY = "empty"
ISSUE_NAN = "nan"
ISSUE_INFINITE = "infinite"
ISSUE_UNSUPPORTED_SAMPLE_RATE = "unsupported_sample_rate"
ISSUE_TOO_LONG = "too_long"
ISSUE_CLIPPED = "clipped"
ISSUE_LOW_AMPLITUDE = "low_amplitude"
ISSUE_SILENT = "silent"
ISSUE_DC_OFFSET = "dc_offset"

ERROR = "error"
WARNING = "warning"


@dataclass(frozen=True)
class AudioIssue:
    """One thing wrong with a recording. Never auto-corrected, only reported."""

    code: str
    severity: str
    detail: str
    value: float | None = None

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return f"[{self.severity}] {self.code}: {self.detail}"


@dataclass(frozen=True)
class ValidationReport:
    """What validation found. ``ok`` means no errors; warnings may still be present."""

    issues: tuple[AudioIssue, ...] = ()
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def errors(self) -> tuple[AudioIssue, ...]:
        return tuple(i for i in self.issues if i.severity == ERROR)

    @property
    def warnings(self) -> tuple[AudioIssue, ...]:
        return tuple(i for i in self.issues if i.severity == WARNING)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        if not self.issues:
            return "ok"
        return "; ".join(f"{i.code}({i.severity})" for i in self.issues)


@dataclass(frozen=True)
class ValidationLimits:
    """Thresholds for validation. Defaults are conservative, not tuned."""

    max_duration_seconds: float = 3600.0
    min_duration_seconds: float = 0.0
    #: Sample values at or above this are considered clipped.
    clipping_threshold: float = 0.999
    #: Fraction of samples at/above the threshold that counts as clipped audio.
    clipping_ratio_warn: float = 0.001
    #: Below this RMS the recording is too quiet to be useful.
    min_rms: float = 1e-5
    #: At or below this RMS the recording is treated as effectively silent.
    silence_rms: float = 1e-7
    #: |mean| above this suggests a DC offset / bad ADC.
    dc_offset_warn: float = 0.05


@dataclass(frozen=True)
class AudioData:
    """A loaded waveform.

    ``waveform`` is float32. Mono audio is 1-D ``(frames,)``; multi-channel audio
    is 2-D ``(frames, channels)``. ``sample_rate`` is the file's own rate; it is
    not silently changed here.
    """

    waveform: FloatArray
    sample_rate: int
    path: Path | None = None
    report: ValidationReport | None = None

    @property
    def num_frames(self) -> int:
        return int(self.waveform.shape[0])

    @property
    def channels(self) -> int:
        return 1 if self.waveform.ndim == 1 else int(self.waveform.shape[1])

    @property
    def is_mono(self) -> bool:
        return self.waveform.ndim == 1

    @property
    def duration(self) -> float:
        return self.num_frames / float(self.sample_rate)

    def to_mono(self) -> AudioData:
        """Average channels down to a 1-D waveform. No-op if already mono."""
        if self.is_mono:
            return self
        mono = self.waveform.mean(axis=1, dtype=np.float32)
        return AudioData(mono, self.sample_rate, self.path, self.report)


def _issue(code: str, severity: str, detail: str, value: float | None = None) -> AudioIssue:
    return AudioIssue(code=code, severity=severity, detail=detail, value=value)


def validate_audio(
    waveform: FloatArray,
    sample_rate: int,
    limits: ValidationLimits | None = None,
) -> ValidationReport:
    """Inspect a waveform and report everything wrong with it.

    Never raises on audio quality. Missing files and decode failures are
    ``load_audio``'s problem, not this function's.
    """
    limits = limits or ValidationLimits()
    issues: list[AudioIssue] = []
    metrics: dict[str, float] = {}

    if sample_rate <= 0:
        issues.append(
            _issue(
                ISSUE_UNSUPPORTED_SAMPLE_RATE,
                ERROR,
                f"sample rate {sample_rate} is not positive",
                float(sample_rate),
            )
        )
    elif not (MIN_SUPPORTED_SAMPLE_RATE <= sample_rate <= MAX_SUPPORTED_SAMPLE_RATE):
        issues.append(
            _issue(
                ISSUE_UNSUPPORTED_SAMPLE_RATE,
                ERROR,
                f"sample rate {sample_rate} outside supported range "
                f"[{MIN_SUPPORTED_SAMPLE_RATE}, {MAX_SUPPORTED_SAMPLE_RATE}]",
                float(sample_rate),
            )
        )

    if waveform.size == 0:
        issues.append(_issue(ISSUE_EMPTY, ERROR, "waveform contains no samples", 0.0))
        return ValidationReport(tuple(issues), metrics)

    # Non-finite checks must come before any reduction, or the reductions are NaN.
    nan_count = int(np.isnan(waveform).sum())
    inf_count = int(np.isinf(waveform).sum())
    if nan_count:
        issues.append(
            _issue(ISSUE_NAN, ERROR, f"{nan_count} NaN sample(s)", float(nan_count))
        )
    if inf_count:
        issues.append(
            _issue(ISSUE_INFINITE, ERROR, f"{inf_count} infinite sample(s)", float(inf_count))
        )
    if nan_count or inf_count:
        return ValidationReport(tuple(issues), metrics)

    if sample_rate <= 0:
        # Duration and every rate-dependent metric are meaningless here, and
        # dividing by it would raise. The rate error is already recorded.
        return ValidationReport(tuple(issues), metrics)

    absolute = np.abs(waveform)
    peak = float(absolute.max())
    rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
    dc_offset = float(np.mean(waveform))

    metrics["peak"] = peak
    metrics["rms"] = rms
    metrics["dc_offset"] = dc_offset
    metrics["crest_factor"] = peak / rms if rms > 0 else 0.0

    duration = waveform.shape[0] / float(sample_rate)
    metrics["duration"] = duration

    if duration > limits.max_duration_seconds:
        issues.append(
            _issue(
                ISSUE_TOO_LONG,
                ERROR,
                f"duration {duration:.2f}s exceeds limit {limits.max_duration_seconds:.2f}s",
                duration,
            )
        )
    if duration < limits.min_duration_seconds:
        issues.append(
            _issue(
                ISSUE_EMPTY,
                ERROR,
                f"duration {duration:.4f}s below minimum {limits.min_duration_seconds:.4f}s",
                duration,
            )
        )

    clipped = int(np.count_nonzero(absolute >= limits.clipping_threshold))
    clipping_ratio = clipped / waveform.size
    metrics["clipping_ratio"] = clipping_ratio
    if clipping_ratio > limits.clipping_ratio_warn:
        issues.append(
            _issue(
                ISSUE_CLIPPED,
                WARNING,
                f"{clipping_ratio * 100:.3f}% of samples at or above "
                f"{limits.clipping_threshold}",
                clipping_ratio,
            )
        )

    if rms <= limits.silence_rms:
        issues.append(
            _issue(ISSUE_SILENT, ERROR, f"effectively silent (rms {rms:.3e})", rms)
        )
    elif rms < limits.min_rms:
        issues.append(
            _issue(ISSUE_LOW_AMPLITUDE, WARNING, f"very low amplitude (rms {rms:.3e})", rms)
        )

    if abs(dc_offset) > limits.dc_offset_warn:
        issues.append(
            _issue(ISSUE_DC_OFFSET, WARNING, f"DC offset {dc_offset:+.4f}", dc_offset)
        )

    return ValidationReport(tuple(issues), metrics)


def load_audio(
    path: str | Path,
    *,
    mono: bool = True,
    validate: bool = True,
    limits: ValidationLimits | None = None,
) -> AudioData:
    """Load an audio file as float32, converting to mono unless told otherwise.

    Returns the recording at its native sample rate. Resampling to 16 kHz is a
    separate step (`resampling.resample_audio`).

    Raises:
        AudioLoadError: the file is missing, unreadable, or corrupt.
        AudioValidationError: the audio decoded but failed validation. The
            attached ``report`` carries every issue found, not just the first.
    """
    target = Path(path)
    if not target.is_file():
        raise AudioLoadError(f"audio file not found: {target}")

    try:
        waveform, sample_rate = sf.read(str(target), dtype="float32", always_2d=False)
    except Exception as exc:  # soundfile raises a variety of low-level errors
        raise AudioLoadError(f"could not read {target}: {exc}") from exc

    data = np.ascontiguousarray(waveform, dtype=np.float32)

    if data.ndim > 2:
        raise AudioLoadError(f"{target}: unexpected waveform shape {data.shape}")

    if sample_rate <= 0:
        raise AudioLoadError(f"{target}: invalid sample rate {sample_rate}")

    if mono and data.ndim == 2:
        data = np.ascontiguousarray(data.mean(axis=1, dtype=np.float32), dtype=np.float32)

    report = validate_audio(data, sample_rate, limits) if validate else None

    if report is not None and not report.ok:
        raise AudioValidationError(
            f"{target}: {report.summary()}",
            report,
        )

    return AudioData(waveform=data, sample_rate=int(sample_rate), path=target, report=report)


def audio_info(path: str | Path) -> dict[str, Any]:
    """Read a file's header without decoding the audio. Cheap duration/frame probe.

    Uses the file's declared metadata, so treat the result as a hint. If the
    header lies, ``load_audio`` will disagree.
    """
    target = Path(path)
    if not target.is_file():
        raise AudioLoadError(f"audio file not found: {target}")
    try:
        info = sf.info(str(target))
    except Exception as exc:
        raise AudioLoadError(f"could not read header of {target}: {exc}") from exc
    return {
        "path": str(target),
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration": float(info.duration),
        "format": info.format,
        "subtype": info.subtype,
    }


__all__ = [
    "AudioData",
    "AudioError",
    "AudioIssue",
    "AudioLoadError",
    "AudioValidationError",
    "CANONICAL_SAMPLE_RATE",
    "FloatArray",
    "ValidationLimits",
    "ValidationReport",
    "audio_info",
    "load_audio",
    "validate_audio",
]

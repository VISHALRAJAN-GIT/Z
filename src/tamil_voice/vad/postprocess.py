"""Turn a VAD frame mask into clean speech segments.

GUIDE section 17's output is not a per-frame decision but speech *segments*. Raw
frame runs are fragmented: a brief drop in energy splits one utterance in two, and
a single noisy frame spawns a spurious 10 ms blob. This module cleans that up —
merge nearby runs, pad the edges so onsets and offsets survive cropping, then drop
anything still too short to be speech — in that documented order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .detector import VadResult


@dataclass(frozen=True)
class Segment:
    """A speech span in seconds, half-open as ``[start, end)``."""

    start: float
    end: float

    def __post_init__(self) -> None:
        if self.start < 0.0:
            raise ValueError(f"segment start must be >= 0, got {self.start}")
        if self.end < self.start:
            raise ValueError(f"segment end {self.end} precedes start {self.start}")

    @property
    def duration(self) -> float:
        return self.end - self.start

    def overlaps(self, other: Segment, tolerance: float = 0.0) -> bool:
        return self.start <= other.end + tolerance and other.start <= self.end + tolerance


@dataclass(frozen=True)
class SegmentConfig:
    """Post-processing parameters. Defaults are conservative, not tuned."""

    #: Segments closer than this are merged.
    merge_gap_seconds: float = 0.20
    #: Segments shorter than this are discarded (applied after padding).
    min_duration_seconds: float = 0.10
    #: Each segment edge is extended by this much.
    pad_seconds: float = 0.05

    def __post_init__(self) -> None:
        for name in ("merge_gap_seconds", "min_duration_seconds", "pad_seconds"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0")


def frames_to_segments(
    mask: npt.NDArray[np.bool_],
    hop_seconds: float,
    frame_seconds: float | None = None,
) -> list[Segment]:
    """Convert a boolean frame mask into raw segments.

    Frame ``i`` covers ``[i * hop, i * hop + frame)``; ``frame_seconds`` defaults
    to ``hop_seconds``.
    """
    if hop_seconds <= 0:
        raise ValueError(f"hop_seconds must be positive, got {hop_seconds}")
    if mask.size == 0:
        return []
    frame = hop_seconds if frame_seconds is None else frame_seconds
    changes = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [
        Segment(float(start) * hop_seconds, float(end - 1) * hop_seconds + frame)
        for start, end in zip(starts, ends, strict=True)
    ]


def merge_segments(segments: list[Segment], max_gap_seconds: float) -> list[Segment]:
    """Merge segments separated by at most ``max_gap_seconds``. Sorted by start."""
    if not segments:
        return []
    ordered = sorted(segments, key=lambda segment: segment.start)
    merged = [ordered[0]]
    for segment in ordered[1:]:
        last = merged[-1]
        if segment.start - last.end <= max_gap_seconds:
            merged[-1] = Segment(last.start, max(last.end, segment.end))
        else:
            merged.append(segment)
    return merged


def filter_short_segments(segments: list[Segment], min_duration_seconds: float) -> list[Segment]:
    """Keep only segments at least ``min_duration_seconds`` long."""
    return [segment for segment in segments if segment.duration >= min_duration_seconds]


def pad_segments(
    segments: list[Segment],
    pad_seconds: float,
    total_duration: float | None = None,
) -> list[Segment]:
    """Extend each segment by ``pad_seconds``, clamped to ``[0, total_duration]``."""
    padded: list[Segment] = []
    for segment in segments:
        start = max(0.0, segment.start - pad_seconds)
        end = segment.end + pad_seconds
        if total_duration is not None:
            end = min(end, total_duration)
        padded.append(Segment(start, end))
    return padded


def build_segments(
    result: VadResult,
    config: SegmentConfig | None = None,
    total_duration: float | None = None,
) -> list[Segment]:
    """Full cleanup pipeline: merge, pad, clamp, re-merge, drop short segments."""
    cfg = config or SegmentConfig()
    segments = frames_to_segments(
        result.mask,
        result.config.hop_seconds,
        result.config.frame_seconds,
    )
    segments = merge_segments(segments, cfg.merge_gap_seconds)
    segments = pad_segments(segments, cfg.pad_seconds, total_duration)
    segments = merge_segments(segments, 0.0)
    return filter_short_segments(segments, cfg.min_duration_seconds)


__all__ = [
    "Segment",
    "SegmentConfig",
    "build_segments",
    "filter_short_segments",
    "frames_to_segments",
    "merge_segments",
    "pad_segments",
]

from __future__ import annotations

import numpy as np
import pytest

from tamil_voice.vad.detector import VadConfig, VadResult
from tamil_voice.vad.postprocess import (
    Segment,
    SegmentConfig,
    build_segments,
    filter_short_segments,
    frames_to_segments,
    merge_segments,
    pad_segments,
)


def _result(mask: list[bool], hop: float = 0.01, frame: float = 0.025) -> VadResult:
    array = np.array(mask, dtype=bool)
    return VadResult(
        mask=array,
        frame_times=np.arange(array.size, dtype=np.float64) * hop,
        energy_db=np.zeros(array.size),
        flatness=None,
        threshold_db=0.0,
        noise_floor_db=0.0,
        config=VadConfig(hop_seconds=hop, frame_seconds=frame),
    )


# ------------------------------------------------------------------ framerate


def test_frames_to_segments_runs() -> None:
    mask = [False, False, True, True, True, False, False, True, False]
    segments = frames_to_segments(np.array(mask), hop_seconds=0.01, frame_seconds=0.025)
    assert len(segments) == 2
    assert segments[0].start == pytest.approx(0.02)
    assert segments[0].end == pytest.approx(0.065)  # frame 4 end: 0.04 + 0.025
    assert segments[1].start == pytest.approx(0.07)
    assert segments[1].end == pytest.approx(0.095)


def test_frames_to_segments_empty_and_validation() -> None:
    assert frames_to_segments(np.array([], dtype=bool), 0.01) == []
    with pytest.raises(ValueError):
        frames_to_segments(np.array([True]), hop_seconds=0.0)


# -------------------------------------------------------------------- merging


def test_merge_within_gap() -> None:
    merged = merge_segments([Segment(0.0, 0.5), Segment(0.6, 0.9)], max_gap_seconds=0.2)
    assert len(merged) == 1
    assert merged[0] == Segment(0.0, 0.9)


def test_no_merge_beyond_gap() -> None:
    merged = merge_segments([Segment(0.0, 0.5), Segment(1.0, 1.2)], max_gap_seconds=0.2)
    assert len(merged) == 2


def test_merge_sorts_and_handles_overlap() -> None:
    merged = merge_segments([Segment(1.0, 1.5), Segment(0.0, 0.3)], max_gap_seconds=0.0)
    assert merged == [Segment(0.0, 0.3), Segment(1.0, 1.5)]
    overlapped = merge_segments([Segment(0.0, 0.6), Segment(0.4, 0.9)], max_gap_seconds=0.0)
    assert overlapped == [Segment(0.0, 0.9)]


def test_merge_empty() -> None:
    assert merge_segments([], max_gap_seconds=0.5) == []


# ------------------------------------------------------------- filter and pad


def test_filter_short_segments() -> None:
    segments = [Segment(0.0, 0.05), Segment(0.2, 0.5)]
    kept = filter_short_segments(segments, min_duration_seconds=0.1)
    assert kept == [Segment(0.2, 0.5)]


def test_pad_clamps_at_zero_and_total() -> None:
    padded = pad_segments([Segment(0.01, 0.40)], pad_seconds=0.05, total_duration=0.42)
    assert padded[0].start == 0.0
    assert padded[0].end == pytest.approx(0.42)


# ------------------------------------------------------------------- pipeline


def test_build_segments_merges_pads_and_filters() -> None:
    mask = [False] * 50
    mask[10:20] = [True] * 10  # [0.100, 0.215]
    mask[30:32] = [True] * 2  # [0.300, 0.335]
    config = SegmentConfig(merge_gap_seconds=0.2, pad_seconds=0.05, min_duration_seconds=0.1)
    segments = build_segments(_result(mask), config, total_duration=0.5)
    assert len(segments) == 1
    assert segments[0].start == pytest.approx(0.05)
    assert segments[0].end == pytest.approx(0.385)


def test_build_segments_drops_isolated_short_run() -> None:
    mask = [False] * 50
    mask[10] = True  # single 25 ms frame, shorter than min_duration
    config = SegmentConfig(merge_gap_seconds=0.2, pad_seconds=0.0, min_duration_seconds=0.1)
    assert build_segments(_result(mask), config) == []


def test_build_segments_empty_mask() -> None:
    assert build_segments(_result([False] * 10)) == []


# -------------------------------------------------------------------- Segment


def test_segment_validation() -> None:
    with pytest.raises(ValueError):
        Segment(-0.1, 0.5)
    with pytest.raises(ValueError):
        Segment(0.5, 0.4)


def test_segment_overlaps() -> None:
    assert Segment(0.0, 0.5).overlaps(Segment(0.4, 0.9))
    assert not Segment(0.0, 0.5).overlaps(Segment(0.6, 0.9))
    assert Segment(0.0, 0.5).overlaps(Segment(0.6, 0.9), tolerance=0.2)


def test_segment_config_validation() -> None:
    with pytest.raises(ValueError):
        SegmentConfig(pad_seconds=-0.1)

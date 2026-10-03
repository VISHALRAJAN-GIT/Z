"""Sample-rate conversion to the canonical 16 kHz.

GUIDE section 13 is specific about two things:

* 8, 22.05, 44.1 and 48 kHz must all convert correctly.
* Audio that is *already* 16 kHz must not be resampled again.

The second rule matters. Resampling an already-correct signal is not a no-op that
"does no harm" — every pass through an interpolator is a filter, and repeated
filtering degrades the audio and makes results harder to attribute. So an
already-canonical input is returned untouched.
"""

from __future__ import annotations

import librosa
import numpy as np

from ..common.config import CANONICAL_SAMPLE_RATE
from .io import AudioData, FloatArray


def resample_waveform(
    waveform: FloatArray,
    orig_sample_rate: int,
    target_sample_rate: int = CANONICAL_SAMPLE_RATE,
) -> FloatArray:
    """Resample a waveform along its time axis.

    Mono ``(frames,)`` and multi-channel ``(frames, channels)`` are both
    supported; time is always axis 0. Returns float32.
    """
    if orig_sample_rate <= 0 or target_sample_rate <= 0:
        raise ValueError(
            f"sample rates must be positive, got {orig_sample_rate} -> {target_sample_rate}"
        )
    data = np.asarray(waveform, dtype=np.float32)
    if data.size == 0:
        return data
    if orig_sample_rate == target_sample_rate:
        return data

    axis = 0
    resampled = librosa.resample(
        data,
        orig_sr=orig_sample_rate,
        target_sr=target_sample_rate,
        axis=axis,
    )
    return np.ascontiguousarray(resampled, dtype=np.float32)


def resample_audio(
    audio: AudioData,
    target_sample_rate: int = CANONICAL_SAMPLE_RATE,
) -> AudioData:
    """Return ``audio`` at the target rate.

    When the audio is already at the target rate, the *same* object is returned
    rather than a copy — nothing is recomputed and nothing is re-filtered. A new
    validation report is not produced; resampling does not repair defects, so the
    original report still describes the audio.
    """
    if audio.sample_rate == target_sample_rate:
        return audio

    resampled = resample_waveform(audio.waveform, audio.sample_rate, target_sample_rate)
    return AudioData(
        waveform=resampled,
        sample_rate=target_sample_rate,
        path=audio.path,
        report=audio.report,
    )


def resample_to_canonical(audio: AudioData) -> AudioData:
    """Convert to the project's canonical 16 kHz."""
    return resample_audio(audio, CANONICAL_SAMPLE_RATE)


__all__ = [
    "CANONICAL_SAMPLE_RATE",
    "resample_audio",
    "resample_to_canonical",
    "resample_waveform",
]

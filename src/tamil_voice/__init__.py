"""Tamil Voice Foundation.

A compact, offline, low-latency Tamil voice intelligence foundation.

The package is organised as independent, individually testable modules:

    audio        loading, resampling, normalisation, features, quality analysis
    vad          voice activity detection
    enhancement  speech enhancement (classical first, neural later)
    asr          speech recognition (Conformer / CTC)
    text         Tamil text normalisation, Tanglish, code switching
    language     intent, entities, context, response generation
    tts          speech synthesis
    optimization distillation, quantization, pruning, export
    runtime      streaming pipeline assembly and session management
    common       logging, configuration, seeding, checkpoints, metrics

Modules that have not been implemented yet are intentionally empty rather than
stuffed with placeholders. See MEMORY.md for current phase.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]

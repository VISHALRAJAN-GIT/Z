"""Shared infrastructure: configuration, logging, seeding, checkpoints, metrics."""

from __future__ import annotations

from .config import (
    CANONICAL_CHANNELS,
    CANONICAL_DTYPE,
    CANONICAL_SAMPLE_RATE,
    AudioSpec,
    ProjectPaths,
    get_paths,
    load_yaml,
    project_root,
)

__all__ = [
    "CANONICAL_CHANNELS",
    "CANONICAL_DTYPE",
    "CANONICAL_SAMPLE_RATE",
    "AudioSpec",
    "ProjectPaths",
    "get_paths",
    "load_yaml",
    "project_root",
]

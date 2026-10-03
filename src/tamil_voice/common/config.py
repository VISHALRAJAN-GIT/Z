"""Project paths and configuration loading.

Two rules drive this module:

1. Experiment parameters live in YAML under `configs/`, never as literals in
   code (AGENTS.md principle 4).
2. The canonical audio representation is fixed and defined exactly once, here
   (GUIDE section 11), so no module can quietly invent its own.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

CANONICAL_SAMPLE_RATE: int = 16_000
CANONICAL_CHANNELS: int = 1
CANONICAL_DTYPE: str = "float32"

_MARKER_FILES = ("pyproject.toml", ".git")


@dataclass(frozen=True)
class AudioSpec:
    """The canonical internal audio representation."""

    sample_rate: int = CANONICAL_SAMPLE_RATE
    channels: int = CANONICAL_CHANNELS
    dtype: str = CANONICAL_DTYPE

    @property
    def ndim(self) -> int:
        return 1 if self.channels == 1 else 2

    def frame_length(self, window_seconds: float, hop_seconds: float) -> tuple[int, int]:
        """Return (window_samples, hop_samples) for a given window and hop."""
        if window_seconds <= 0 or hop_seconds <= 0:
            raise ValueError("window_seconds and hop_seconds must be positive")
        return int(round(window_seconds * self.sample_rate)), int(round(hop_seconds * self.sample_rate))


CANONICAL_AUDIO = AudioSpec()


@dataclass(frozen=True)
class ProjectPaths:
    """Every path the project needs, resolved once and validated."""

    root: Path
    data: Path
    configs: Path
    checkpoints: Path
    artifacts: Path
    experiments: Path
    docs: Path
    tests: Path

    @property
    def manifests(self) -> Path:
        return self.data / "manifests"

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def processed(self) -> Path:
        return self.data / "processed"

    @property
    def plots(self) -> Path:
        return self.artifacts / "plots"

    @property
    def benchmark_results(self) -> Path:
        return self.artifacts / "benchmark_results"

    def ensure(self) -> None:
        """Create the writable directories that git does not track."""
        for path in (
            self.data,
            self.manifests,
            self.checkpoints,
            self.artifacts,
            self.plots,
            self.benchmark_results,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def relative(self, path: Path) -> str:
        """Render a path relative to the project root, for logs and manifests."""
        try:
            return path.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return path.as_posix()


@lru_cache(maxsize=1)
def project_root() -> Path:
    """Locate the repository root by walking up until a marker file appears.

    Falls back to the current working directory so the function still behaves
    sensibly when the package is imported from an installed wheel.
    """
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if any((candidate / marker).exists() for marker in _MARKER_FILES):
            return candidate
    return Path.cwd().resolve()


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    if not raw:
        return default
    path = Path(raw)
    return path if path.is_absolute() else (project_root() / path)


@lru_cache(maxsize=1)
def get_paths() -> ProjectPaths:
    """Build the project's path map, honouring TVF_* environment overrides."""
    root = project_root()
    data = _env_path("TVF_DATA_ROOT", root / "data")
    return ProjectPaths(
        root=root,
        data=data,
        configs=_env_path("TVF_CONFIG_ROOT", root / "configs"),
        checkpoints=_env_path("TVF_CHECKPOINT_ROOT", root / "checkpoints"),
        artifacts=_env_path("TVF_ARTIFACT_ROOT", root / "artifacts"),
        experiments=root / "experiments",
        docs=root / "docs",
        tests=root / "tests",
    )


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return a mapping.

    Raises FileNotFoundError with the resolved absolute path so a missing config
    is never mistaken for an empty one.
    """
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = project_root() / resolved
    if not resolved.is_file():
        raise FileNotFoundError(f"config file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise TypeError(f"{resolved} must contain a mapping at the top level, got {type(loaded).__name__}")
    return loaded


def resolve_config_path(relative: str | Path) -> Path:
    """Resolve a config path against the configured `configs/` root.

    Tries the literal path first, then the same path with a `.yaml` suffix. When
    neither exists, the `.yaml` form is returned so that a caller reporting the
    failure names the path the file is expected to have.
    """
    candidate = Path(relative)
    if candidate.is_absolute():
        return candidate
    configs_root = get_paths().configs
    direct = configs_root / candidate
    if direct.is_file():
        return direct
    with_suffix = configs_root / candidate.with_suffix(".yaml")
    if with_suffix.is_file() or not direct.is_file():
        return with_suffix
    return direct

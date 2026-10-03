"""Determinism helpers.

Reproducibility is a hard requirement (AGENTS.md principle 11), so seeding lives
in one place and reports honestly what it could and could not seed.

torch is optional. If it is absent, `seed_everything` says so rather than
pretending the run is deterministic.
"""

from __future__ import annotations

import importlib.util
import os
import random
from dataclasses import dataclass
from typing import Any

DEFAULT_SEED = 1337


def torch_available() -> bool:
    """True when torch can be imported without importing it."""
    return importlib.util.find_spec("torch") is not None


@dataclass(frozen=True)
class SeedReport:
    """What was actually seeded. Absence of torch is recorded, not hidden."""

    seed: int
    python_hash_seed: bool
    numpy: bool
    torch: bool
    cuda_deterministic: bool

    @property
    def fully_deterministic(self) -> bool:
        return self.python_hash_seed and self.numpy and self.torch


def seed_everything(
    seed: int = DEFAULT_SEED,
    *,
    deterministic_torch: bool = True,
) -> SeedReport:
    """Seed Python, NumPy and (if present) torch.

    Sets PYTHONHASHSEED for child processes where the platform allows it, and
    requests deterministic cuDNN kernels when torch is available.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    numpy_seeded = False
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
        numpy_seeded = True
    except ImportError:
        pass

    torch_seeded = False
    cuda_deterministic = False
    if torch_available():
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch_seeded = True
        else:
            torch_seeded = True
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            cuda_deterministic = True

    return SeedReport(
        seed=seed,
        python_hash_seed=os.environ.get("PYTHONHASHSEED") == str(seed),
        numpy=numpy_seeded,
        torch=torch_seeded,
        cuda_deterministic=cuda_deterministic,
    )


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker initialiser that keeps augmentation reproducible."""
    base_seed: Any = torch_worker_seed() if torch_available() else None
    if base_seed is None:
        offset = int(os.getenv("TVF_SEED", DEFAULT_SEED))
        seed = offset + worker_id
    else:
        import torch

        seed = int(base_seed + worker_id)
        torch.manual_seed(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:
        pass


def torch_worker_seed() -> int | None:
    """The DataLoader's base seed, or None when torch is unavailable."""
    try:
        import torch

        return int(torch.initial_seed())
    except ImportError:
        return None

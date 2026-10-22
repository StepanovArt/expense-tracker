"""Global seeding so every experiment is reproducible from (config, seed)."""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np

DEFAULT_SEED = 1234


def seed_everything(seed: int = DEFAULT_SEED, deterministic_torch: bool = False) -> int:
    """Seed python, numpy and (if installed) torch. Returns the seed for logging."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    return seed


def derive_seed(base_seed: int, *names: str) -> int:
    """Stable sub-seed for a named component (e.g. one split group).

    Uses sha256 instead of ``hash()`` because python's string hash is salted per process.
    """
    h = hashlib.sha256(("/".join([str(base_seed), *names])).encode()).hexdigest()
    return int(h[:8], 16)


def rng(base_seed: int, *names: str) -> np.random.Generator:
    return np.random.default_rng(derive_seed(base_seed, *names))

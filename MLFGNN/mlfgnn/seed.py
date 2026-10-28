"""Seeding helpers.

Every number in the results table has to be reproducible from (config hash,
fold, seed). This module is the single place where global RNG state is touched.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy and (if installed) torch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is optional for stages 1-2
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def rng(seed: int) -> np.random.Generator:
    """A local generator -- preferred over the global numpy state."""
    return np.random.default_rng(seed)

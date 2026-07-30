"""RNG seeding reproducibility aid for experiments and ensemble members.

This module does not enable deterministic algorithms or promise bitwise
repeatability across devices, library versions, or execution environments.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed default ``random``, NumPy, and PyTorch CPU/CUDA RNG streams.

    This does not configure PyTorch deterministic algorithms or backend flags.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

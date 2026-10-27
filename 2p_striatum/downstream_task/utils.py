"""
decoders/utils.py
------------------

Shared utilities for the decoders package.
"""

import random

import numpy as np
import torch


def _seed_all(seed: int) -> None:
    """Set random seeds for reproducibility across all RNG backends."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

"""Configuration and reproducibility helpers."""
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path):
    def resolve(value):
        if isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v) for v in value]
        if isinstance(value, str):
            value = os.path.expandvars(value)
            try:
                return float(value) if any(c in value.lower() for c in ('.', 'e')) else int(value)
            except ValueError:
                return value
        return value
    with open(path) as handle:
        return resolve(yaml.safe_load(handle))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_config(config, path):
    Path(path).write_text(yaml.safe_dump(config, sort_keys=False))

"""Load fixed train/val split indices and apply to a dataset."""

import json
import os
from pathlib import Path

from torch.utils.data import Subset

SPLIT_FILE = Path(__file__).resolve().parent / "configs" / "train_val_split.json"


def load_split_indices(split_file: str | Path | None = None) -> dict:
    """Load the split JSON and return the parsed dict."""
    path = Path(split_file) if split_file is not None else SPLIT_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}. "
            "Run generate_train_val_split.py first."
        )
    with open(path) as f:
        return json.load(f)


def apply_train_val_split(dataset, split_file: str | Path | None = None):
    """
    Split *dataset* into train and val Subsets using pre-generated indices.

    Parameters
    ----------
    dataset : Dataset
        The full training dataset (before any train/val split).
    split_file : path, optional
        Path to the JSON file. Defaults to the checked-in
        ``reconstruction/config/train_val_split.json``.

    Returns
    -------
    train_subset, val_subset : Subset, Subset
    """
    split = load_split_indices(split_file)

    if len(dataset) != split["num_total"]:
        raise ValueError(
            f"Dataset has {len(dataset)} samples but split file expects "
            f"{split['num_total']}. Regenerate the split file if the dataset changed."
        )

    train_subset = Subset(dataset, split["train_indices"])
    val_subset = Subset(dataset, split["val_indices"])
    return train_subset, val_subset

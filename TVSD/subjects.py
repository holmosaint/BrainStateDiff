"""Neuron-to-subject mapping for the TVSD ventral stream dataset."""

import torch

SUBJECT_NEURON_MAP = {
    "monkeyN": {"start": 0, "end": 198, "count": 198, "subject_id": 0},
    "monkeyF": {"start": 198, "end": 315, "count": 117, "subject_id": 1},
}

NUM_SUBJECTS = len(SUBJECT_NEURON_MAP)
TOTAL_NEURONS = 315
MAX_NEURONS = max(info["count"] for info in SUBJECT_NEURON_MAP.values())  # 198


def get_subject_slices():
    """Return subject name -> slice mapping for indexing into (B, N) tensors."""
    return {
        name: slice(info["start"], info["end"])
        for name, info in SUBJECT_NEURON_MAP.items()
    }


def get_subject_neuron_counts():
    """Return ordered list of neuron counts per subject (ordered by subject_id)."""
    sorted_subjects = sorted(
        SUBJECT_NEURON_MAP.items(), key=lambda x: x[1]["subject_id"]
    )
    return [info["count"] for _, info in sorted_subjects]


def build_padding_mask(batch_size: int, device=None):
    """
    Build padding mask for the expanded 2*batch_size batch.
    Returns (2*batch_size, MAX_NEURONS) bool tensor where True = padded (ignore).
    """
    counts = get_subject_neuron_counts()
    mask_N = torch.zeros(batch_size, MAX_NEURONS, dtype=torch.bool, device=device)
    mask_F = torch.zeros(batch_size, MAX_NEURONS, dtype=torch.bool, device=device)
    mask_F[:, counts[1] :] = True
    return torch.cat([mask_N, mask_F], dim=0)

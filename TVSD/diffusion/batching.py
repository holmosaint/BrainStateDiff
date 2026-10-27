"""
Custom dataloader for NeuralDiT timeseries training.

Same independent per-subject sampling as dataloader_dit.py, but the base
dataset returns (image, timeseries) where timeseries has shape (T, N).

The collate function decomposes the timeseries into two temporal components:
  - baseline:  mean of bins 0..baseline_end   (early response / spontaneous)
  - target:    mean of bins target_start..target_end (evoked response)

Each component is (B, N) after averaging over the time axis.
"""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from TVSD.subjects import (
    MAX_NEURONS,
    SUBJECT_NEURON_MAP,
)


# ---------------------------------------------------------------------------
# Temporal decomposition config
# ---------------------------------------------------------------------------

DEFAULT_BASELINE_BINS = (0, 10)    # average of bins [0, 10)
DEFAULT_TARGET_BINS = (15, 25)     # average of bins [15, 25)


# ---------------------------------------------------------------------------
# Per-subject dataset wrapper for timeseries
# ---------------------------------------------------------------------------


class SubjectTimeseriesDataset(Dataset):
    """
    Wraps a base TimeseriesImageDataset and returns only the neurons
    belonging to one subject.

    Returns
    -------
    image : Tensor (3, H, W)
    timeseries : Tensor (T, n_neurons_for_this_subject)
    subject_id : int
    """

    def __init__(self, base_dataset: Dataset, neuron_slice: slice, subject_id: int):
        self.base_dataset = base_dataset
        self.neuron_slice = neuron_slice
        self.subject_id = subject_id

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        image, full_ts = self.base_dataset[idx]  # full_ts: (T, N_total)
        return image, full_ts[:, self.neuron_slice], self.subject_id


# ---------------------------------------------------------------------------
# Combined multi-subject dataset
# ---------------------------------------------------------------------------


class MultiSubjectTimeseriesDataset(Dataset):
    def __init__(self, subject_datasets: Sequence[SubjectTimeseriesDataset]):
        self.subject_datasets = list(subject_datasets)

    def __len__(self) -> int:
        return sum(len(ds) for ds in self.subject_datasets)

    def __getitem__(self, index: Tuple[int, int]):
        subject_id, sample_idx = index
        return self.subject_datasets[subject_id][sample_idx]


# ---------------------------------------------------------------------------
# Independent per-subject batch sampler (same as dataloader_dit.py)
# ---------------------------------------------------------------------------


class IndependentSubjectBatchSampler(Sampler):
    def __init__(
        self,
        num_samples: int,
        batch_size: int,
        num_subjects: int = 2,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = True,
    ):
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.num_subjects = num_subjects
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        pools: List[List[int]] = []
        for _ in range(self.num_subjects):
            idxs = list(range(self.num_samples))
            if self.shuffle:
                rng.shuffle(idxs)
            pools.append(idxs)

        ptrs = [0] * self.num_subjects
        remaining = self.num_samples

        while remaining > 0:
            actual_bs = min(self.batch_size, remaining)
            if actual_bs < self.batch_size and self.drop_last:
                break

            batch: List[Tuple[int, int]] = []
            for s in range(self.num_subjects):
                for _ in range(actual_bs):
                    if ptrs[s] >= len(pools[s]):
                        ptrs[s] = 0
                        if self.shuffle:
                            rng.shuffle(pools[s])
                    batch.append((s, pools[s][ptrs[s]]))
                    ptrs[s] += 1
            yield batch
            remaining -= actual_bs

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size
        return (self.num_samples + self.batch_size - 1) // self.batch_size


# ---------------------------------------------------------------------------
# Collate with temporal decomposition
# ---------------------------------------------------------------------------


def make_collate_fn(
    baseline_bins: Tuple[int, int] = DEFAULT_BASELINE_BINS,
    target_bins: Tuple[int, int] = DEFAULT_TARGET_BINS,
):
    """
    Returns a collate function that decomposes timeseries into baseline + target.

    Each item is (image, timeseries, subject_id) where timeseries: (T, n_neurons).

    Returns
    -------
    images : Tensor (2B, 3, H, W)
    baseline : Tensor (2B, MAX_NEURONS) -- zero-padded, mean of baseline bins
    target : Tensor (2B, MAX_NEURONS) -- zero-padded, mean of target bins
    subject_ids : LongTensor (2B,)
    pad_mask : BoolTensor (2B, MAX_NEURONS) -- True = padded position
    """

    def collate_fn(batch):
        images, timeseries_list, subject_ids = zip(*batch)
        images = torch.stack(images)
        subject_ids = torch.tensor(subject_ids, dtype=torch.long)

        baseline_padded = torch.zeros(len(batch), MAX_NEURONS)
        target_padded = torch.zeros(len(batch), MAX_NEURONS)
        mask = torch.ones(len(batch), MAX_NEURONS, dtype=torch.bool)

        for i, ts in enumerate(timeseries_list):
            # ts: (T, n_neurons)
            n = ts.shape[1]

            # Temporal decomposition
            bl = ts[baseline_bins[0]:baseline_bins[1]].mean(dim=0)  # (n_neurons,)
            tg = ts[target_bins[0]:target_bins[1]].mean(dim=0)      # (n_neurons,)

            baseline_padded[i, :n] = bl
            target_padded[i, :n] = tg
            mask[i, :n] = False

        return images, baseline_padded, target_padded, subject_ids, mask

    return collate_fn


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------


def build_subject_datasets(base_dataset: Dataset):
    datasets = []
    for name, info in sorted(
        SUBJECT_NEURON_MAP.items(), key=lambda x: x[1]["subject_id"]
    ):
        datasets.append(
            SubjectTimeseriesDataset(
                base_dataset,
                neuron_slice=slice(info["start"], info["end"]),
                subject_id=info["subject_id"],
            )
        )
    return datasets


def build_dataloader(
    base_dataset: Dataset,
    batch_size: int,
    baseline_bins: Tuple[int, int] = DEFAULT_BASELINE_BINS,
    target_bins: Tuple[int, int] = DEFAULT_TARGET_BINS,
    shuffle: bool = True,
    seed: int = 42,
    num_workers: int = 8,
    drop_last: bool = True,
):
    subject_datasets = build_subject_datasets(base_dataset)
    multi_dataset = MultiSubjectTimeseriesDataset(subject_datasets)
    sampler = IndependentSubjectBatchSampler(
        num_samples=len(base_dataset),
        batch_size=batch_size,
        num_subjects=len(subject_datasets),
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )
    return DataLoader(
        multi_dataset,
        batch_sampler=sampler,
        collate_fn=make_collate_fn(baseline_bins, target_bins),
        num_workers=num_workers,
        pin_memory=True,
    )

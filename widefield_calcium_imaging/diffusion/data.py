from __future__ import annotations
import argparse, copy, logging, math, os, random
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import h5py as h5
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from diffusers import DDIMScheduler
from tqdm import tqdm
class ImageEmbeddingStore:
    def __init__(self, path: Optional[str], emb_dim: int, save_dir: Optional[Path] = None, logger: Optional[logging.Logger] = None, force_flatten: bool = False) -> None:
        self.emb_dim = int(emb_dim)              # per-token embedding dim
        self.tokens_per_emb: Optional[int] = None
        self.token_dim: Optional[int] = None
        self.embeddings: Dict[str, np.ndarray] = {}
        self.keys: List[str] = []
        self.save_path: Optional[Path] = None
        self.dirty = False
        self._logger = logger
        self.force_flatten = force_flatten

        if logger is not None:
            logger.info(f"Initializing ImageEmbeddingStore with token_dim={self.emb_dim}...")
            logger.info(f"Loading existing embeddings from: {path}..." if path else "No existing embedding path provided.")

        if path is None:
            raise ValueError("Precomputed image embeddings are required; --image-embeddings must be set.")

        candidate = Path(path)
        if not candidate.is_file():
            raise FileNotFoundError(f"Image embedding file not found: {candidate}")

        self._load_embeddings(candidate, logger=logger)
        self.save_path = candidate

    def _normalize_and_validate(self, key: str, src: Any) -> np.ndarray:
        arr = np.asarray(src)
        assert arr.ndim == 2
        if self.force_flatten:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != self.emb_dim:
            raise ValueError(f"Embedding for key {key} has dim {arr.shape}, expected (*,{self.emb_dim})")
        tokens = arr.shape[0]

        if self.tokens_per_emb is None:
            self.tokens_per_emb = tokens
            self.token_dim = arr.shape[1]
        else:
            if tokens != self.tokens_per_emb or arr.shape[1] != self.token_dim:
                raise ValueError(
                    f"Inconsistent embedding shape for key {key}: got tokens={tokens}, dim={arr.shape[1]}, "
                    f"expected tokens={self.tokens_per_emb}, dim={self.token_dim}"
                )

        return np.asarray(arr, dtype=np.float32)

    def _load_embeddings(self, path: Path, logger: Optional[logging.Logger] = None) -> None:
        data = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(data, dict):
            raise ValueError("Image embedding file must be a dict mapping key -> embedding source")

        for key, src in data.items():
            k = str(key)
            arr = self._normalize_and_validate(k, src)
            self.embeddings[k] = arr
            self.keys.append(k)

        if logger is not None:
            logger.info(f"Loaded {len(self.embeddings)} image embeddings from {path} with tokens_per_emb={self.tokens_per_emb}, token_dim={self.token_dim}")

    def ensure(self, key: str, value: Optional[np.ndarray] = None) -> np.ndarray:
        if key not in self.embeddings:
            raise KeyError(f"Image embedding missing for key: {key}")
        return self.embeddings[key]

    def get(self, key: str) -> np.ndarray:
        return self.ensure(key)

@dataclass(frozen=True)
class SampleSlice:
    key: str
    trial_idx: int
    start_frame: int
    target_start: int
    history_len: int

class SlidingWindowLatentDataset(Dataset):
    """
    Returns:
        - history_clip: (H_hist, C, H, W) where H_hist==history_frames
        - target_clip:  (F_tgt, C, H, W)
        - onset_idx: target_start - stim_onset_frame
        - image_emb: (P, E) tokens_per_emb x token_dim (kept 2D)
        - cond_mask: (T_total,) float {0,1} (1 => apply image cond)
        - target_start: int
        - _key_str: str
    """
    def __init__(
        self,
        data_path: str,
        split: str,
        history_frames: int,
        target_frames: int,
        stim_onset_frame: int,
        image_store: ImageEmbeddingStore,
        cond_start_frame: Optional[int] = None,
        frame_st: int = 0,
        frame_end: Optional[int] = None,
    ) -> None:
        self.history_frames = int(history_frames)
        self.target_frames = int(target_frames)
        self.total_frames = self.history_frames + self.target_frames

        self.stim_onset_frame = int(stim_onset_frame)
        self.cond_start_frame = int(cond_start_frame) if cond_start_frame is not None else int(stim_onset_frame)

        self.frame_st = int(frame_st)
        self.frame_end = frame_end
        self.image_store = image_store

        split_path = str(Path(data_path) / split) if not Path(split).is_absolute() else split
        if not os.path.isfile(split_path):
            raise FileNotFoundError(f"Latent file not found: {split_path}")
        self.split_path = split_path
        self._h5: Optional[h5.File] = None

        self.frame_slices: List[SampleSlice] = []
        self.target_start_to_indices: Dict[int, List[int]] = {}
        onset_list: List[int] = []

        with h5.File(self.split_path, "r") as f:
            for key in f.keys():
                if key == "mask":
                    continue
                self.image_store.ensure(key)
                trials = f[key]
                n_trials, T = trials.shape[0], trials.shape[1]
                upper = T if frame_end is None else min(T, int(frame_end))
                start_min = max(0, self.frame_st)
                for trial_idx in range(n_trials):
                    start_max = upper - self.total_frames
                    if start_max < start_min:
                        continue
                    for start in range(start_min, start_max + 1):
                        # Allow variable history length by sliding target_start within the history span.
                        for hist_len in range(self.history_frames + 1):
                            target_start = start + hist_len
                            onset_idx = target_start - self.stim_onset_frame
                            self.frame_slices.append(
                                SampleSlice(
                                    key=key,
                                    trial_idx=trial_idx,
                                    start_frame=start,
                                    target_start=target_start,
                                    history_len=hist_len,
                                )
                            )
                            onset_list.append(onset_idx)
                            self.target_start_to_indices.setdefault(target_start, []).append(len(self.frame_slices) - 1)

        self.min_onset = int(min(onset_list)) if onset_list else 0
        self.max_onset = int(max(onset_list)) if onset_list else 0

    def _get_h5(self) -> h5.File:
        if self._h5 is None:
            self._h5 = h5.File(self.split_path, "r")
        return self._h5

    def __len__(self) -> int:
        return len(self.frame_slices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        h5f = self._get_h5()
        s = self.frame_slices[idx]
        group = h5f[s.key]
        trial = group[s.trial_idx]  # (T,H,W) or (T,C,H,W)
        window = trial[s.start_frame: s.start_frame + self.total_frames]

        if window.ndim == 3:
            T, H, W = window.shape
            C = 1
            window = window.reshape(T, C, H, W)
        elif window.ndim == 4:
            T, C, H, W = window.shape
        else:
            raise ValueError("Expected (T,H,W) or (T,C,H,W) arrays from H5")

        hist_len = int(s.history_len)

        # Extract history/target frames (history_len may be < history_frames)
        history_frames = window[:hist_len] if hist_len > 0 else np.empty((0, C, H, W), dtype=window.dtype)
        target_frames = window[hist_len: hist_len + self.target_frames]

        # Pack history into fixed (history_frames, C, H, W), right-aligned; pad missing with zeros
        history_clip = np.zeros((self.history_frames, C, H, W), dtype=np.float32)
        if hist_len > 0:
            history_clip[-hist_len:] = history_frames

        target_clip = target_frames.astype(np.float32)

        target_start = int(s.target_start)
        onset_idx = int(target_start - self.stim_onset_frame)

        abs_frames = s.start_frame + np.arange(self.total_frames, dtype=np.int64)
        cond_mask = (abs_frames >= self.cond_start_frame).astype(np.float32)  # (T_total,)

        image_emb = self.image_store.get(s.key).copy()

        return {
            "history_clip": torch.from_numpy(history_clip).float(),
            "target_clip": torch.from_numpy(target_clip).float(),
            "onset_idx": torch.tensor(onset_idx, dtype=torch.long),
            "image_emb": torch.from_numpy(image_emb).float(),
            "cond_mask": torch.from_numpy(cond_mask).float(),
            "target_start": torch.tensor(target_start, dtype=torch.long),
            "_key_str": s.key,
        }

    def close(self) -> None:
        if self._h5 is not None:
            try:
                self._h5.close()
            finally:
                self._h5 = None

    def __del__(self) -> None:
        self.close()

@dataclass
class TrainBatch:
    history_clip: torch.Tensor
    target_clip: torch.Tensor
    onset_idx: torch.Tensor
    image_emb: torch.Tensor
    cond_mask: torch.Tensor
    target_start: torch.Tensor
    subject_idx: torch.Tensor

class MultiSubjectSlidingWindowDataset(Dataset):
    def __init__(
        self,
        subject_datasets: Sequence[SlidingWindowLatentDataset],
        subject_names: Sequence[str],
    ) -> None:
        if len(subject_datasets) == 0:
            raise ValueError("subject_datasets cannot be empty")
        if len(subject_datasets) != len(subject_names):
            raise ValueError("subject_datasets and subject_names must have the same length")

        self.subject_datasets = list(subject_datasets)
        self.subject_names = list(subject_names)
        self.num_subjects = len(self.subject_datasets)

        self.subject_lengths = [len(ds) for ds in self.subject_datasets]
        self.offsets: List[int] = [0]
        for ln in self.subject_lengths:
            self.offsets.append(self.offsets[-1] + int(ln))

        self.target_start_to_indices: Dict[int, List[int]] = {}
        for si, ds in enumerate(self.subject_datasets):
            off = self.offsets[si]
            for target_start, idxs in ds.target_start_to_indices.items():
                self.target_start_to_indices.setdefault(int(target_start), []).extend(
                    [off + int(i) for i in idxs]
                )

        self.min_onset = min(ds.min_onset for ds in self.subject_datasets)
        self.max_onset = max(ds.max_onset for ds in self.subject_datasets)

    def __len__(self) -> int:
        return self.offsets[-1]

    def _locate(self, idx: int) -> Tuple[int, int]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        lo, hi = 0, self.num_subjects
        while lo < hi:
            mid = (lo + hi) // 2
            if idx < self.offsets[mid + 1]:
                hi = mid
            else:
                lo = mid + 1
        subject_idx = lo
        local_idx = idx - self.offsets[subject_idx]
        return subject_idx, local_idx

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        subject_idx, local_idx = self._locate(int(idx))
        sample = self.subject_datasets[subject_idx][local_idx]
        sample["subject_idx"] = torch.tensor(subject_idx, dtype=torch.long)
        sample["_subject_str"] = self.subject_names[subject_idx]
        return sample

    def close(self) -> None:
        for ds in self.subject_datasets:
            ds.close()

    def __del__(self) -> None:
        self.close()


class BalancedMultiSubjectBatchSampler(Sampler[List[int]]):
    """
    Yields balanced index batches where each subject contributes exactly
    `batch_size / num_subjects` samples in every mini-batch.
    """

    def __init__(
        self,
        subject_lengths: Sequence[int],
        subject_offsets: Sequence[int],
        batch_size: int,
        steps_per_epoch: int = -1,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.subject_lengths = [int(x) for x in subject_lengths]
        self.subject_offsets = [int(x) for x in subject_offsets]
        self.num_subjects = len(self.subject_lengths)

        if self.num_subjects <= 0:
            raise ValueError("At least one subject is required")
        if len(self.subject_offsets) != self.num_subjects + 1:
            raise ValueError("subject_offsets must have length num_subjects + 1")

        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.batch_size % self.num_subjects != 0:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be divisible by num_subjects ({self.num_subjects})"
            )

        self.per_subject_batch = self.batch_size // self.num_subjects
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

        max_balanced_steps = min(
            max(0, ln // self.per_subject_batch) for ln in self.subject_lengths
        )
        if max_balanced_steps <= 0:
            raise ValueError(
                "At least one balanced batch is impossible: increase dataset size or reduce batch_size/num_subjects"
            )

        if int(steps_per_epoch) <= 0:
            self.steps_per_epoch = max_balanced_steps
        else:
            self.steps_per_epoch = int(steps_per_epoch)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        pools: List[List[int]] = []
        ptrs: List[int] = []
        for si, ln in enumerate(self.subject_lengths):
            off = self.subject_offsets[si]
            idxs = list(range(off, off + ln))
            if self.shuffle:
                rng.shuffle(idxs)
            pools.append(idxs)
            ptrs.append(0)

        for _ in range(self.steps_per_epoch):
            batch: List[int] = []
            for si in range(self.num_subjects):
                pool = pools[si]
                for _k in range(self.per_subject_batch):
                    if ptrs[si] >= len(pool):
                        ptrs[si] = 0
                        if self.shuffle:
                            rng.shuffle(pool)
                    batch.append(pool[ptrs[si]])
                    ptrs[si] += 1
            if self.shuffle:
                rng.shuffle(batch)
            yield batch

def collate_fn(batch: Sequence[Dict[str, torch.Tensor]]) -> TrainBatch:
    return TrainBatch(
        history_clip=torch.stack([b["history_clip"] for b in batch], dim=0),
        target_clip=torch.stack([b["target_clip"] for b in batch], dim=0),
        onset_idx=torch.stack([b["onset_idx"] for b in batch], dim=0),
        image_emb=torch.stack([b["image_emb"] for b in batch], dim=0),
        cond_mask=torch.stack([b["cond_mask"] for b in batch], dim=0),
        target_start=torch.stack([b["target_start"] for b in batch], dim=0),
        subject_idx=torch.stack([b["subject_idx"] for b in batch], dim=0),
    )

"""
chunk_diffusion/chunk_dataset.py
---------------------------------

Chunk dataset for diffusion training and sampling.

Two modes:
  - "sliding": training loader. Anchor slides from min_anchor_frame onward,
    producing multiple samples per trial. History = [t-H : t], future = all
    non-overlapping K-frame chunks from t to end of trial.
  - "fixed": sampling loaders. Fixed anchor at min_anchor_frame. History is
    always frames [min_anchor_frame-H : min_anchor_frame], future from
    min_anchor_frame onward. One sample per trial.

All loaders discard frames before min_anchor_frame - H (= frame 30 by default).
Last chunk is zero-padded with a validity mask when partial.
"""

from __future__ import annotations

import csv as csv_mod
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Normalization stats
# ---------------------------------------------------------------------------

@dataclass
class ChunkNormStats:
    """Global normalization statistics for z, v, c channels."""
    z_mean: torch.Tensor   # (D,)
    z_std: torch.Tensor    # (D,)
    v_mean: torch.Tensor   # (1,)
    v_std: torch.Tensor    # (1,)
    c_mean: torch.Tensor   # (1,)
    c_std: torch.Tensor    # (1,)
    n_frames: int

    def save(self, path: Path) -> None:
        torch.save({
            "z_mean": self.z_mean, "z_std": self.z_std,
            "v_mean": self.v_mean, "v_std": self.v_std,
            "c_mean": self.c_mean, "c_std": self.c_std,
            "n_frames": self.n_frames,
        }, path)

    @staticmethod
    def load(path: Path) -> "ChunkNormStats":
        d = torch.load(path, map_location="cpu", weights_only=False)
        return ChunkNormStats(
            z_mean=d["z_mean"], z_std=d["z_std"],
            v_mean=d["v_mean"], v_std=d["v_std"],
            c_mean=d["c_mean"], c_std=d["c_std"],
            n_frames=d["n_frames"],
        )


def compute_global_norm_stats(
    latent_dir: Path,
    n_sessions: int,
    min_frame: int = 30,
) -> ChunkNormStats:
    """Compute global mean/std from training frames >= min_frame across all sessions."""
    z_sum = None
    z_sumsq = None
    v_sum = 0.0
    v_sumsq = 0.0
    c_sum = 0.0
    c_sumsq = 0.0
    count = 0

    for si in range(n_sessions):
        train_dir = latent_dir / f"session_{si}" / "train"
        if not train_dir.exists():
            continue

        for trial_path in sorted(train_dir.glob("trial_*.pt")):
            trial = torch.load(trial_path, map_location="cpu", weights_only=False)
            T = trial["length"]
            if T <= min_frame:
                continue

            z = trial["z_seq"][min_frame:T]   # (T', D)
            v = trial["v_seq"][min_frame:T]   # (T', 1)
            c = trial["c_seq"][min_frame:T]   # (T', 1)
            T_used = z.shape[0]

            D = z.shape[1]
            if z_sum is None:
                z_sum = torch.zeros(D, dtype=torch.float64)
                z_sumsq = torch.zeros(D, dtype=torch.float64)

            z_sum += z.double().sum(dim=0)
            z_sumsq += (z.double() ** 2).sum(dim=0)
            v_sum += v.double().sum().item()
            v_sumsq += (v.double() ** 2).sum().item()
            c_sum += c.double().sum().item()
            c_sumsq += (c.double() ** 2).sum().item()
            count += T_used

    if count == 0 or z_sum is None:
        raise RuntimeError("No training data found for normalization stats")

    z_mean = (z_sum / count).float()
    z_var = (z_sumsq / count).float() - z_mean ** 2
    z_std = torch.clamp(torch.sqrt(torch.clamp(z_var, min=0) + 1e-8), min=1e-6)

    v_mean_val = v_sum / count
    v_var = v_sumsq / count - v_mean_val ** 2
    v_std_val = max(np.sqrt(max(v_var, 0) + 1e-8), 1e-6)

    c_mean_val = c_sum / count
    c_var = c_sumsq / count - c_mean_val ** 2
    c_std_val = max(np.sqrt(max(c_var, 0) + 1e-8), 1e-6)

    stats = ChunkNormStats(
        z_mean=z_mean, z_std=z_std,
        v_mean=torch.tensor([v_mean_val], dtype=torch.float32),
        v_std=torch.tensor([v_std_val], dtype=torch.float32),
        c_mean=torch.tensor([c_mean_val], dtype=torch.float32),
        c_std=torch.tensor([c_std_val], dtype=torch.float32),
        n_frames=count,
    )
    logger.info(
        f"Global norm stats: {count} frames (from frame {min_frame}+), "
        f"z_mean range=[{z_mean.min():.4f}, {z_mean.max():.4f}], "
        f"z_std range=[{z_std.min():.4f}, {z_std.max():.4f}], "
        f"v_mean={v_mean_val:.4f}, v_std={v_std_val:.4f}, "
        f"c_mean={c_mean_val:.4f}, c_std={c_std_val:.4f}"
    )
    return stats


# ---------------------------------------------------------------------------
# Session-to-subject mapping
# ---------------------------------------------------------------------------

def build_session_to_subject_map(
    session_names: List[str],
) -> Tuple[Dict[int, int], List[str]]:
    """Derive session_idx -> subject_idx mapping from session names.

    Session names are expected to be ``<subject>_<date>`` (e.g. ``sw24_20231121``).

    Returns:
        session_to_subject: {session_idx: subject_idx}
        subject_names: ordered list of unique subject names
    """
    subject_names_ordered: List[str] = []
    subject_name_to_idx: Dict[str, int] = {}
    session_to_subject: Dict[int, int] = {}

    for si, name in enumerate(session_names):
        subj = name.split("_")[0]
        if subj not in subject_name_to_idx:
            subject_name_to_idx[subj] = len(subject_names_ordered)
            subject_names_ordered.append(subj)
        session_to_subject[si] = subject_name_to_idx[subj]

    return session_to_subject, subject_names_ordered


# ---------------------------------------------------------------------------
# Cursor model
# ---------------------------------------------------------------------------

class ChunkDataset(Dataset):
    """Chunk dataset with sliding or fixed anchor.

    Args:
        latent_dir: Root directory of extracted latents.
        split: "train", "valid", or "test".
        session_indices: List of session indices to include.
        session_to_subject: {session_idx: subject_idx} mapping.
        mode: "sliding" (training) or "fixed" (sampling).
        history_len: H, number of history frames. Default 16.
        chunk_len: K, future chunk size. Default 8.
        stride: Sliding window stride (only for mode="sliding"). Default 1.
        min_anchor_frame: Earliest anchor position. Default 46.
        norm_stats: Optional global normalization stats.
    """

    def __init__(
        self,
        latent_dir: Path,
        split: str,
        session_indices: List[int],
        session_to_subject: Dict[int, int],
        mode: str = "fixed",
        history_len: int = 16,
        chunk_len: int = 8,
        stride: int = 1,
        min_anchor_frame: int = 46,
        norm_stats: Optional[ChunkNormStats] = None,
    ):
        super().__init__()
        assert mode in ("sliding", "fixed"), f"Unknown mode: {mode}"
        self.mode = mode
        self.history_len = history_len
        self.chunk_len = chunk_len
        self.min_anchor_frame = min_anchor_frame
        self.norm_stats = norm_stats
        self.session_to_subject = session_to_subject

        H, K = history_len, chunk_len

        # Load all trials
        self._trials: List[Dict[str, Any]] = []
        # Index: list of (trial_idx, anchor_frame)
        self._samples: List[Tuple[int, int]] = []
        # Track max chunks for padding
        self._max_chunks = 0

        for si in session_indices:
            split_dir = Path(latent_dir) / f"session_{si}" / split
            if not split_dir.exists():
                continue

            for trial_path in sorted(split_dir.glob("trial_*.pt")):
                trial = torch.load(trial_path, map_location="cpu", weights_only=False)
                T = trial["length"]

                if mode == "fixed":
                    anchor = min_anchor_frame
                    if T <= anchor:
                        continue
                    trial_idx = len(self._trials)
                    self._trials.append(trial)
                    n_chunks = math.ceil((T - anchor) / K)
                    self._max_chunks = max(self._max_chunks, n_chunks)
                    self._samples.append((trial_idx, anchor))

                else:  # sliding
                    # Anchors: t from min_anchor_frame to T-K (inclusive)
                    # so at least one full chunk fits
                    if T < min_anchor_frame + K:
                        continue
                    trial_idx = len(self._trials)
                    self._trials.append(trial)
                    for t in range(min_anchor_frame, T - K + 1, stride):
                        n_chunks = math.ceil((T - t) / K)
                        self._max_chunks = max(self._max_chunks, n_chunks)
                        self._samples.append((trial_idx, t))

        logger.info(
            f"ChunkDataset({split}, {mode}): {len(self._trials)} trials, "
            f"{len(self._samples)} samples, max_chunks={self._max_chunks} "
            f"(H={H}, K={K}, anchor>={min_anchor_frame})"
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        trial_idx, anchor = self._samples[idx]
        trial = self._trials[trial_idx]

        H = self.history_len
        K = self.chunk_len
        T = trial["length"]
        M = self._max_chunks

        # --- History ---
        z_hist = trial["z_seq"][anchor - H: anchor].clone()   # (H, D)
        v_hist = trial["v_seq"][anchor - H: anchor].clone()   # (H, 1)
        c_hist = trial["c_seq"][anchor - H: anchor].clone()   # (H, 1)

        # --- Future chunks ---
        D = z_hist.shape[1]
        z_chunks = torch.zeros(M, K, D)
        v_chunks = torch.zeros(M, K, 1)
        c_chunks = torch.zeros(M, K, 1)
        chunk_masks = torch.zeros(M, K, dtype=torch.bool)

        future_frames = T - anchor
        n_chunks = math.ceil(future_frames / K)

        for i in range(n_chunks):
            start = anchor + i * K
            end = min(anchor + (i + 1) * K, T)
            length = end - start

            z_chunks[i, :length] = trial["z_seq"][start:end]
            v_chunks[i, :length] = trial["v_seq"][start:end]
            c_chunks[i, :length] = trial["c_seq"][start:end]
            chunk_masks[i, :length] = True

        session_id = trial["session_id"]
        subject_id = self.session_to_subject[session_id]

        # --- Normalize ---
        if self.norm_stats is not None:
            s = self.norm_stats
            z_hist = (z_hist - s.z_mean) / s.z_std
            v_hist = (v_hist - s.v_mean) / s.v_std
            c_hist = (c_hist - s.c_mean) / s.c_std
            z_chunks = (z_chunks - s.z_mean) / s.z_std
            v_chunks = (v_chunks - s.v_mean) / s.v_std
            c_chunks = (c_chunks - s.c_mean) / s.c_std
            # Zero out padded positions after normalization
            z_chunks[~chunk_masks] = 0.0
            v_chunks[~chunk_masks.unsqueeze(-1).expand_as(v_chunks)] = 0.0
            c_chunks[~chunk_masks.unsqueeze(-1).expand_as(c_chunks)] = 0.0

        chunk_offset = anchor - self.min_anchor_frame  # 0 for fixed mode

        return {
            "z_hist": z_hist,              # (H, D)
            "v_hist": v_hist,              # (H, 1)
            "c_hist": c_hist,              # (H, 1)
            "trial_type": trial["trial_type"],
            "session_id": session_id,
            "subject_id": subject_id,
            "chunk_offset": chunk_offset,  # int, anchor position relative to frame 46
            "z_chunks": z_chunks,          # (M, K, D)
            "v_chunks": v_chunks,          # (M, K, 1)
            "c_chunks": c_chunks,          # (M, K, 1)
            "chunk_masks": chunk_masks,    # (M, K)
            "n_chunks": n_chunks,
        }


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------

def build_chunk_dataloaders(
    latent_dir: Path,
    session_names: List[str],
    history_len: int = 16,
    chunk_len: int = 8,
    stride: int = 1,
    min_anchor_frame: int = 46,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
    norm_stats: Optional[ChunkNormStats] = None,
    compute_norm: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader,
           ChunkNormStats, Dict[int, int], List[str]]:
    """Build four DataLoaders for chunk diffusion.

    Returns:
        (train_loader,          # sliding mode, shuffled
         sample_train_loader,   # fixed mode, unshuffled
         sample_valid_loader,   # fixed mode, unshuffled
         sample_test_loader,    # fixed mode, unshuffled
         norm_stats, session_to_subject, subject_names)
    """
    latent_dir = Path(latent_dir)
    session_indices = list(range(len(session_names)))
    session_to_subject, subject_names = build_session_to_subject_map(session_names)

    logger.info(
        f"Session-to-subject mapping: {len(session_names)} sessions, "
        f"{len(subject_names)} subjects {subject_names}"
    )

    # Compute or load normalization stats
    min_frame = min_anchor_frame - history_len  # = 30
    if norm_stats is None and compute_norm:
        norm_stats = compute_global_norm_stats(latent_dir, len(session_indices), min_frame)

    ds_kwargs = dict(
        session_indices=session_indices,
        session_to_subject=session_to_subject,
        history_len=history_len,
        chunk_len=chunk_len,
        min_anchor_frame=min_anchor_frame,
        norm_stats=norm_stats,
    )

    train_ds = ChunkDataset(latent_dir, "train", mode="sliding", stride=stride, **ds_kwargs)
    sample_train_ds = ChunkDataset(latent_dir, "train", mode="fixed", **ds_kwargs)
    sample_valid_ds = ChunkDataset(latent_dir, "valid", mode="fixed", **ds_kwargs)
    sample_test_ds = ChunkDataset(latent_dir, "test", mode="fixed", **ds_kwargs)

    def _make_loader(ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
        )

    return (
        _make_loader(train_ds, shuffle=True),
        _make_loader(sample_train_ds, shuffle=False),
        _make_loader(sample_valid_ds, shuffle=False),
        _make_loader(sample_test_ds, shuffle=False),
        norm_stats,
        session_to_subject,
        subject_names,
    )

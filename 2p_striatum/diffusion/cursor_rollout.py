"""
chunk_diffusion/cursor_rollout.py
----------------------------------

Session-specific cursor update rule for closed-loop rollout.

The cursor position is updated from wheel velocity using a per-session
linear model: delta_pos[t] = coef[session] * velocity[t] + intercept[session].

Coefficients are loaded from vel_to_dpos_linear_model.csv.

This module is used by:
  - training.py: cursor recomputation during rollout training
  - sampling.py: closed-loop autoregressive generation
"""

from __future__ import annotations

import csv as csv_mod
from pathlib import Path
from typing import Dict, List

import torch


def load_cursor_model(csv_path: Path) -> Dict[str, Dict[str, float]]:
    """Load per-session velocity-to-delta-position linear model.

    CSV format: session, r2, coef, intercept

    Returns: {session_name: {"coef": float, "intercept": float, "r2": float}}
    """
    models: Dict[str, Dict[str, float]] = {}
    with csv_path.open("r") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            name = row["session"].strip()
            if name == "GLOBAL":
                continue
            models[name] = {
                "coef": float(row["coef"]),
                "intercept": float(row["intercept"]),
                "r2": float(row["r2"]),
            }
    return models


def build_cursor_coef_tensors(
    session_names: List[str],
    cursor_models: Dict[str, Dict[str, float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (cursor_coefs, cursor_intercepts) tensors indexed by session_id.

    Returns:
        cursor_coefs: (num_sessions,) float32
        cursor_intercepts: (num_sessions,) float32
    """
    n = len(session_names)
    coefs = torch.zeros(n)
    intercepts = torch.zeros(n)
    for si, name in enumerate(session_names):
        coefs[si] = cursor_models[name]["coef"]
        intercepts[si] = cursor_models[name]["intercept"]
    return coefs, intercepts


def rollout_cursor_chunk(
    cursor_start: torch.Tensor,      # (B, 1)
    velocity_chunk: torch.Tensor,    # (B, K, 1)
    session_ids: torch.Tensor,       # (B,)
    cursor_coefs: torch.Tensor,      # (num_sessions,) — per-session coef
    cursor_intercepts: torch.Tensor, # (num_sessions,) — per-session intercept
) -> torch.Tensor:                   # (B, K, 1)
    """Compute cursor positions for a chunk from velocity using session-specific linear model.

    For each frame t in [0, K):
        c[t] = c[t-1] + coef[session] * v[t] + intercept[session]

    where c[-1] = cursor_start.

    All inputs should be in the same space (raw or normalized — caller's responsibility).
    Typically called in raw (unnormalized) space during rollout.

    Args:
        cursor_start: Last cursor position before this chunk. Shape (B, 1).
        velocity_chunk: Velocity values for K frames. Shape (B, K, 1).
        session_ids: Session index per sample. Shape (B,).
        cursor_coefs: Per-session linear coefficient. Shape (num_sessions,).
        cursor_intercepts: Per-session intercept. Shape (num_sessions,).

    Returns:
        cursor_chunk: Cursor positions for K frames. Shape (B, K, 1).
    """
    B, K, _ = velocity_chunk.shape
    coef = cursor_coefs[session_ids].unsqueeze(-1).unsqueeze(-1)            # (B, 1, 1)
    intercept = cursor_intercepts[session_ids].unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)

    delta = coef * velocity_chunk + intercept  # (B, K, 1)
    cumulative = delta.cumsum(dim=1)           # (B, K, 1)
    cursor_chunk = cursor_start.unsqueeze(1) + cumulative  # (B, K, 1)
    return cursor_chunk


def rollout_cursor_multichunk(
    cursor_start: torch.Tensor,      # (B, 1)
    velocity_chunks: torch.Tensor,   # (B, M, K, 1)
    session_ids: torch.Tensor,       # (B,)
    cursor_coefs: torch.Tensor,      # (num_sessions,)
    cursor_intercepts: torch.Tensor, # (num_sessions,)
    n_chunks: torch.Tensor,          # (B,) number of valid chunks per sample
) -> torch.Tensor:                   # (B, M, K, 1)
    """Compute cursor positions for multiple consecutive chunks.

    Chains rollout_cursor_chunk across chunks, using the last cursor of
    each chunk as the starting cursor for the next.

    Returns zero for invalid chunks (where chunk index >= n_chunks[b]).
    """
    B, M, K, _ = velocity_chunks.shape
    cursor_all = torch.zeros(B, M, K, 1, device=velocity_chunks.device)
    c_start = cursor_start

    for i in range(M):
        valid = n_chunks > i  # (B,)
        if not valid.any():
            break

        c_chunk = rollout_cursor_chunk(
            c_start, velocity_chunks[:, i], session_ids,
            cursor_coefs, cursor_intercepts,
        )  # (B, K, 1)

        # Only write valid samples
        cursor_all[:, i] = c_chunk * valid.view(B, 1, 1).float()

        # Update start for next chunk (use last frame of current chunk)
        c_start = c_chunk[:, -1, :]  # (B, 1)

    return cursor_all

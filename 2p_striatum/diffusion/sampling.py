"""
chunk_diffusion/sampling.py
-----------------------------

Closed-loop autoregressive sampling pipeline.

Given a real history prefix, generates future chunks autoregressively:
  1. Sample next chunk from diffusion model (DDIM)
  2. Denormalize velocity, compute cursor via session-specific rule
  3. Renormalize cursor, update history
  4. Check stopping conditions (success / wrong direction / max length)
  5. Repeat

All model I/O is in normalized space. Cursor rollout happens in raw space.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from diffusers import DDIMScheduler

from .chunk_denoiser import ChunkDenoiser
from .chunk_dataset import ChunkNormStats
from .cursor_rollout import rollout_cursor_chunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DDIM chunk sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def ddim_sample_chunk(
    model: ChunkDenoiser,
    scheduler: DDIMScheduler,
    z_hist: torch.Tensor,        # (B, H, D) normalized
    v_hist: torch.Tensor,        # (B, H, 1) normalized
    c_hist: torch.Tensor,        # (B, H, 1) normalized
    trial_type_ids: torch.Tensor,
    session_ids: torch.Tensor,
    subject_ids: torch.Tensor,
    chunk_offset: torch.Tensor,
    num_steps: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample one future chunk using DDIM.

    Returns:
        z_chunk: (B, K, D) normalized neural latents
        v_chunk: (B, K, 1) normalized velocity
    """
    B = z_hist.shape[0]
    K = model.chunk_len
    D = model.latent_dim
    D1 = model.token_dim  # D + 1
    device = z_hist.device

    scheduler.set_timesteps(num_steps, device=device)
    x = torch.randn(B, K, D1, device=device)

    for t in scheduler.timesteps:
        ts = torch.full((B,), int(t), device=device, dtype=torch.long)
        pred = model(x, ts, z_hist, v_hist, c_hist,
                     trial_type_ids, session_ids, subject_ids, chunk_offset)
        x = scheduler.step(pred, t, x).prev_sample

    z_chunk = x[:, :, :D]   # (B, K, D)
    v_chunk = x[:, :, D:]   # (B, K, 1)
    return z_chunk, v_chunk


# ---------------------------------------------------------------------------
# Autoregressive closed-loop rollout
# ---------------------------------------------------------------------------

@dataclass
class RolloutResult:
    """Result of one autoregressive rollout."""
    z_generated: torch.Tensor       # (T_syn, D) normalized
    v_generated: torch.Tensor       # (T_syn, 1) normalized
    c_generated: torch.Tensor       # (T_syn, 1) normalized
    z_generated_raw: torch.Tensor   # (T_syn, D) raw (denormalized)
    v_generated_raw: torch.Tensor   # (T_syn, 1) raw
    c_generated_raw: torch.Tensor   # (T_syn, 1) raw
    trial_type: int
    session_id: int
    subject_id: int
    n_chunks_generated: int
    stop_reason: str                # "success" | "wrong_direction" | "max_len"


@torch.no_grad()
def autoregressive_rollout(
    model: ChunkDenoiser,
    scheduler: DDIMScheduler,
    z_hist_init: torch.Tensor,      # (1, H, D) normalized
    v_hist_init: torch.Tensor,      # (1, H, 1) normalized
    c_hist_init: torch.Tensor,      # (1, H, 1) normalized
    trial_type_id: int,
    session_id: int,
    subject_id: int,
    chunk_offset: int,
    norm_stats: ChunkNormStats,
    cursor_coefs: torch.Tensor,     # (num_sessions,)
    cursor_intercepts: torch.Tensor,
    max_rollout_chunks: int = 18,
    num_ddim_steps: int = 50,
    target_center: float = 0.0,
    target_threshold: float = 0.2,
    wrong_dir_limit: int = 2,
) -> RolloutResult:
    """Run closed-loop autoregressive rollout for a single trial.

    Stopping conditions (checked in order):
      1. Success: any cursor position within target_threshold of target_center
      2. Wrong direction: cursor moves away from target for wrong_dir_limit consecutive chunks
      3. Max length: generated max_rollout_chunks chunks

    Args:
        z/v/c_hist_init: History prefix, shape (1, H, *). Must be normalized.
        trial_type_id, session_id, subject_id: Scalars for this trial.
        chunk_offset: Anchor offset relative to frame 46 (typically 0 for fixed mode).
        target_center: Target cursor position in RAW space.
        target_threshold: Threshold for success in RAW space.
    """
    device = z_hist_init.device
    K = model.chunk_len
    D = model.latent_dim

    v_mean = norm_stats.v_mean.to(device)
    v_std = norm_stats.v_std.to(device)
    c_mean = norm_stats.c_mean.to(device)
    c_std = norm_stats.c_std.to(device)
    z_mean = norm_stats.z_mean.to(device)
    z_std = norm_stats.z_std.to(device)
    cursor_coefs = cursor_coefs.to(device)
    cursor_intercepts = cursor_intercepts.to(device)

    # Batch dimension = 1
    trial_type_ids = torch.tensor([trial_type_id], device=device)
    session_ids = torch.tensor([session_id], device=device)
    subject_ids = torch.tensor([subject_id], device=device)
    chunk_offset_t = torch.tensor([chunk_offset], device=device)

    # Working history
    z_hist = z_hist_init.clone()
    v_hist = v_hist_init.clone()
    c_hist = c_hist_init.clone()

    # Collected generated chunks (normalized)
    z_gen_list: List[torch.Tensor] = []
    v_gen_list: List[torch.Tensor] = []
    c_gen_list: List[torch.Tensor] = []

    # Stopping state
    consecutive_wrong_dir = 0
    stop_reason = "max_len"

    # Determine expected direction from trial type
    # trial_type 0 = Cue-Left: cursor starts negative, must move positive toward 0
    # trial_type 1 = Cue-Right: cursor starts positive, must move negative toward 0
    expected_sign = -1.0 if trial_type_id == 1 else 1.0

    model.eval()

    for chunk_idx in range(max_rollout_chunks):
        # Sample next chunk (normalized)
        z_chunk, v_chunk = ddim_sample_chunk(
            model, scheduler, z_hist, v_hist, c_hist,
            trial_type_ids, session_ids, subject_ids, chunk_offset_t,
            num_steps=num_ddim_steps,
        )  # z_chunk: (1, K, D), v_chunk: (1, K, 1) normalized

        # --- Cursor rollout in raw space ---
        v_chunk_raw = v_chunk * v_std + v_mean          # (1, K, 1)
        c_last_raw = (c_hist[:, -1:, :] * c_std + c_mean).squeeze(1)  # (1, 1)

        c_chunk_raw = rollout_cursor_chunk(
            c_last_raw, v_chunk_raw, session_ids, cursor_coefs, cursor_intercepts,
        )  # (1, K, 1) raw

        # Renormalize cursor
        c_chunk_norm = (c_chunk_raw - c_mean) / c_std   # (1, K, 1) normalized

        # Denormalize z for storage
        z_chunk_raw = z_chunk * z_std + z_mean

        # Collect
        z_gen_list.append(z_chunk[0])        # (K, D) normalized
        v_gen_list.append(v_chunk[0])        # (K, 1) normalized
        c_gen_list.append(c_chunk_norm[0])   # (K, 1) normalized

        # --- Stopping checks ---

        # 1. Success check (in raw space)
        cursor_raw_flat = c_chunk_raw[0, :, 0]  # (K,)
        if (cursor_raw_flat - target_center).abs().min().item() <= target_threshold:
            stop_reason = "success"
            break

        # 2. Wrong direction check (in raw space)
        c_first_raw = c_chunk_raw[0, 0, 0].item()
        c_last_chunk_raw = c_chunk_raw[0, -1, 0].item()
        delta_c = c_last_chunk_raw - c_first_raw
        if delta_c * expected_sign < 0:
            consecutive_wrong_dir += 1
        else:
            consecutive_wrong_dir = 0

        if consecutive_wrong_dir >= wrong_dir_limit:
            stop_reason = "wrong_direction"
            break

        # --- Update history ---
        z_hist = torch.cat([z_hist[:, K:], z_chunk], dim=1)
        v_hist = torch.cat([v_hist[:, K:], v_chunk], dim=1)
        c_hist = torch.cat([c_hist[:, K:], c_chunk_norm], dim=1)

    # Concatenate all generated chunks and move to CPU
    z_gen = torch.cat(z_gen_list, dim=0).cpu()     # (T_syn, D) normalized
    v_gen = torch.cat(v_gen_list, dim=0).cpu()     # (T_syn, 1) normalized
    c_gen = torch.cat(c_gen_list, dim=0).cpu()     # (T_syn, 1) normalized

    # Raw (denormalized) versions
    z_gen_raw = z_gen * z_std.cpu() + z_mean.cpu()
    v_gen_raw = v_gen * v_std.cpu() + v_mean.cpu()
    c_gen_raw = c_gen * c_std.cpu() + c_mean.cpu()

    return RolloutResult(
        z_generated=z_gen,
        v_generated=v_gen,
        c_generated=c_gen,
        z_generated_raw=z_gen_raw,
        v_generated_raw=v_gen_raw,
        c_generated_raw=c_gen_raw,
        trial_type=trial_type_id,
        session_id=session_id,
        subject_id=subject_id,
        n_chunks_generated=len(z_gen_list),
        stop_reason=stop_reason,
    )

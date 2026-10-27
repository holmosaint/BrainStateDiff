"""
chunk_diffusion/training.py
-----------------------------

Unified training loop with mixed teacher-forcing and autoregressive rollout.

Key design:
  - Per-sample coin flip: teacher-forcing or rollout
  - TF probability decays over epochs (tf_start * tf_decay^epoch, floored at tf_min)
  - Rollout: iterate through all chunks, sample from model, recompute cursor
    from generated velocity using session-specific linear model
  - Loss computed in normalized space on [z, v] jointly
  - Last chunk masked for partial frames
  - Cursor denormalized for rollout computation, renormalized for model input
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler

from .chunk_denoiser import ChunkDenoiser
from .chunk_dataset import ChunkNormStats
from .cursor_rollout import rollout_cursor_chunk


def compute_training_target(
    scheduler: DDIMScheduler,
    x0: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """Compute training target based on scheduler prediction_type.

    Args:
        x0: clean sample (B, K, D+1)
        noise: sampled noise (B, K, D+1)
        timesteps: (B,)

    Returns:
        target: (B, K, D+1) — what the model should predict
    """
    pred_type = scheduler.config.prediction_type
    if pred_type == "epsilon":
        return noise
    elif pred_type == "sample":
        return x0
    elif pred_type == "v_prediction":
        alpha_prod = scheduler.alphas_cumprod.to(device=timesteps.device)[timesteps]
        alpha = alpha_prod.sqrt().view(-1, 1, 1)
        sigma = (1.0 - alpha_prod).sqrt().view(-1, 1, 1)
        return alpha * noise - sigma * x0
    else:
        raise ValueError(f"Unknown prediction_type: {pred_type}")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {name: p.data.clone() for name, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name])

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return dict(self.shadow)

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.clone() for k, v in state.items()}


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def masked_mse(
    pred: torch.Tensor,    # (B, K, C)
    target: torch.Tensor,  # (B, K, C)
    mask: torch.Tensor,    # (B, K) bool
) -> torch.Tensor:
    """MSE loss with per-frame masking."""
    m = mask.unsqueeze(-1).float()              # (B, K, 1)
    sq = (pred - target) ** 2 * m               # (B, K, C)
    return sq.sum() / m.sum().clamp(min=1.0) / pred.shape[-1]


def split_masked_mse(
    pred: torch.Tensor,      # (B, K, D+1)
    target: torch.Tensor,    # (B, K, D+1)
    mask: torch.Tensor,      # (B, K) bool
    latent_dim: int,
    lambda_z: float = 1.0,
    lambda_v: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split MSE loss for z (latent) and v (velocity) channels.

    Returns: (total_loss, loss_z, loss_v)
    """
    loss_z = masked_mse(pred[:, :, :latent_dim], target[:, :, :latent_dim], mask)
    loss_v = masked_mse(pred[:, :, latent_dim:], target[:, :, latent_dim:], mask)
    total = lambda_z * loss_z + lambda_v * loss_v
    return total, loss_z, loss_v


# ---------------------------------------------------------------------------
# DDIM sampling for rollout during training
# ---------------------------------------------------------------------------

@torch.no_grad()
def ddim_sample_chunk(
    model: ChunkDenoiser,
    scheduler: DDIMScheduler,
    z_hist: torch.Tensor,        # (B, H, D)
    v_hist: torch.Tensor,        # (B, H, 1)
    c_hist: torch.Tensor,        # (B, H, 1)
    trial_type_ids: torch.Tensor,
    session_ids: torch.Tensor,
    subject_ids: torch.Tensor,
    chunk_offset: torch.Tensor,
    num_steps: int = 10,
) -> torch.Tensor:               # (B, K, D+1)
    """Sample a future chunk using DDIM. Returns normalized [z, v].

    Uses fewer steps than inference for training efficiency.
    """
    B = z_hist.shape[0]
    K = model.chunk_len
    D1 = model.token_dim  # D + 1
    device = z_hist.device

    scheduler.set_timesteps(num_steps, device=device)
    z = torch.randn(B, K, D1, device=device)

    for t in scheduler.timesteps:
        ts = torch.full((B,), int(t), device=device, dtype=torch.long)
        pred = model(z, ts, z_hist, v_hist, c_hist,
                     trial_type_ids, session_ids, subject_ids, chunk_offset)
        z = scheduler.step(pred, t, z).prev_sample

    return z  # (B, K, D+1) in normalized space


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def compute_teacher_force_prob(epoch: int, tf_start: float, tf_decay: float, tf_min: float) -> float:
    return max(tf_min, tf_start * (tf_decay ** epoch))


def train_step(
    model: ChunkDenoiser,
    scheduler: DDIMScheduler,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    norm_stats: ChunkNormStats,
    cursor_coefs: torch.Tensor,      # (num_sessions,) raw-space
    cursor_intercepts: torch.Tensor, # (num_sessions,) raw-space
    teacher_force_prob: float = 1.0,
    lambda_z: float = 1.0,
    lambda_v: float = 1.0,
    rollout_ddim_steps: int = 10,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """One training step with mixed teacher-forcing and rollout.

    All model inputs/outputs are in normalized space.
    Cursor rollout happens in raw space (denormalize v -> compute cursor -> renormalize c).
    """
    z_chunks = batch["z_chunks"].to(device)       # (B, M, K, D)
    v_chunks = batch["v_chunks"].to(device)       # (B, M, K, 1)
    c_chunks = batch["c_chunks"].to(device)       # (B, M, K, 1) — normalized real cursor
    chunk_masks = batch["chunk_masks"].to(device)  # (B, M, K)
    n_chunks = batch["n_chunks"].to(device)        # (B,)
    z_hist = batch["z_hist"].to(device)            # (B, H, D)
    v_hist = batch["v_hist"].to(device)            # (B, H, 1)
    c_hist = batch["c_hist"].to(device)            # (B, H, 1)
    trial_type_ids = batch["trial_type"].to(device)
    session_ids = batch["session_id"].to(device)
    subject_ids = batch["subject_id"].to(device)
    chunk_offset = batch["chunk_offset"].to(device)

    B, M, K, D = z_chunks.shape
    latent_dim = D

    # Move norm stats to device
    v_mean = norm_stats.v_mean.to(device)
    v_std = norm_stats.v_std.to(device)
    c_mean = norm_stats.c_mean.to(device)
    c_std = norm_stats.c_std.to(device)
    cursor_coefs = cursor_coefs.to(device)
    cursor_intercepts = cursor_intercepts.to(device)

    # Per-sample: decide teacher-force or rollout
    is_tf = (torch.rand(B, device=device) < teacher_force_prob)  # (B,) bool

    # Working copies of history (updated for rollout samples)
    z_hist_cur = z_hist.clone()
    v_hist_cur = v_hist.clone()
    c_hist_cur = c_hist.clone()

    total_loss = torch.tensor(0.0, device=device)
    # Track TF and rollout losses separately
    tf_loss_sum = 0.0
    tf_loss_z_sum = 0.0
    tf_loss_v_sum = 0.0
    tf_count = 0
    ro_loss_sum = 0.0
    ro_loss_z_sum = 0.0
    ro_loss_v_sum = 0.0
    ro_count = 0
    n_loss_terms = 0

    max_chunks_in_batch = int(n_chunks.max().item())

    for i in range(max_chunks_in_batch):
        valid = n_chunks > i  # (B,) bool — which samples have chunk i
        if not valid.any():
            break

        # --- Build target from real data (always real, even for rollout) ---
        y0 = torch.cat([z_chunks[:, i], v_chunks[:, i]], dim=-1)  # (B, K, D+1) normalized
        mask = chunk_masks[:, i] & valid.unsqueeze(-1)             # (B, K) bool

        # --- Noise and denoise ---
        t_diff = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device)
        noise = torch.randn_like(y0)
        yt = scheduler.add_noise(y0, noise, t_diff)

        pred = model(
            yt, t_diff, z_hist_cur, v_hist_cur, c_hist_cur,
            trial_type_ids, session_ids, subject_ids, chunk_offset,
        )

        # Training target depends on prediction type (epsilon, sample, v_prediction)
        target = compute_training_target(scheduler, y0, noise, t_diff)

        # --- Masked loss (in normalized space) ---
        loss_i, lz_i, lv_i = split_masked_mse(
            pred, target, mask, latent_dim, lambda_z, lambda_v,
        )
        total_loss = total_loss + loss_i
        n_loss_terms += 1

        # --- Track TF vs rollout losses separately ---
        with torch.no_grad():
            sq_err = (pred - target) ** 2  # (B, K, D+1)
            m_float = mask.unsqueeze(-1).float()  # (B, K, 1)
            per_sample_loss = (sq_err * m_float).sum(dim=(1, 2)) / m_float.sum(dim=(1, 2)).clamp(min=1)

            is_tf_valid = is_tf & valid
            is_ro_valid = (~is_tf) & valid

            if is_tf_valid.any():
                tf_loss_sum += per_sample_loss[is_tf_valid].mean().item()
                sq_z = (pred[:, :, :latent_dim] - target[:, :, :latent_dim]) ** 2
                tf_loss_z_sum += (sq_z * m_float).sum(dim=(1, 2))[is_tf_valid].mean().item() / (m_float.sum(dim=(1, 2))[is_tf_valid].mean().item() * latent_dim + 1e-8)
                sq_v = (pred[:, :, latent_dim:] - target[:, :, latent_dim:]) ** 2
                tf_loss_v_sum += (sq_v * m_float).sum(dim=(1, 2))[is_tf_valid].mean().item() / (m_float.sum(dim=(1, 2))[is_tf_valid].mean().item() + 1e-8)
                tf_count += 1

            if is_ro_valid.any():
                ro_loss_sum += per_sample_loss[is_ro_valid].mean().item()
                sq_z = (pred[:, :, :latent_dim] - target[:, :, :latent_dim]) ** 2
                ro_loss_z_sum += (sq_z * m_float).sum(dim=(1, 2))[is_ro_valid].mean().item() / (m_float.sum(dim=(1, 2))[is_ro_valid].mean().item() * latent_dim + 1e-8)
                sq_v = (pred[:, :, latent_dim:] - target[:, :, latent_dim:]) ** 2
                ro_loss_v_sum += (sq_v * m_float).sum(dim=(1, 2))[is_ro_valid].mean().item() / (m_float.sum(dim=(1, 2))[is_ro_valid].mean().item() + 1e-8)
                ro_count += 1

        # --- Update history for next chunk ---
        if i < max_chunks_in_batch - 1:
            # For rollout samples: sample chunk, compute cursor from generated velocity
            # For TF samples: use real data
            with torch.no_grad():
                y_gen = ddim_sample_chunk(
                    model, scheduler, z_hist_cur, v_hist_cur, c_hist_cur,
                    trial_type_ids, session_ids, subject_ids, chunk_offset,
                    num_steps=rollout_ddim_steps,
                )  # (B, K, D+1) normalized

            z_gen = y_gen[:, :, :latent_dim]  # (B, K, D) normalized
            v_gen = y_gen[:, :, latent_dim:]  # (B, K, 1) normalized

            # Denormalize v for cursor rollout
            v_gen_raw = v_gen * v_std + v_mean  # (B, K, 1) raw
            # Denormalize last cursor from history
            c_last_raw = c_hist_cur[:, -1:, :] * c_std + c_mean  # (B, 1, 1) -> (B, 1) raw
            c_last_raw = c_last_raw.squeeze(1)  # (B, 1)

            # Compute cursor in raw space
            c_gen_raw = rollout_cursor_chunk(
                c_last_raw, v_gen_raw, session_ids, cursor_coefs, cursor_intercepts,
            )  # (B, K, 1) raw

            # Renormalize cursor
            c_gen = (c_gen_raw - c_mean) / c_std  # (B, K, 1) normalized

            # For TF samples: use real chunks
            z_real = z_chunks[:, i]   # (B, K, D) normalized
            v_real = v_chunks[:, i]   # (B, K, 1) normalized
            c_real = c_chunks[:, i]   # (B, K, 1) normalized

            # Merge: TF uses real, rollout uses generated
            is_tf_exp = is_tf.view(B, 1, 1)  # (B, 1, 1)
            z_next = torch.where(is_tf_exp, z_real, z_gen)
            v_next = torch.where(is_tf_exp, v_real, v_gen)
            c_next = torch.where(is_tf_exp, c_real, c_gen)

            # Shift history: drop oldest K, append new chunk
            z_hist_cur = torch.cat([z_hist_cur[:, K:], z_next], dim=1)
            v_hist_cur = torch.cat([v_hist_cur[:, K:], v_next], dim=1)
            c_hist_cur = torch.cat([c_hist_cur[:, K:], c_next], dim=1)

    loss = total_loss / max(n_loss_terms, 1)
    metrics = {
        "loss": loss.item(),
        "tf_loss": tf_loss_sum / max(tf_count, 1),
        "tf_loss_z": tf_loss_z_sum / max(tf_count, 1),
        "tf_loss_v": tf_loss_v_sum / max(tf_count, 1),
        "ro_loss": ro_loss_sum / max(ro_count, 1),
        "ro_loss_z": ro_loss_z_sum / max(ro_count, 1),
        "ro_loss_v": ro_loss_v_sum / max(ro_count, 1),
        "n_chunks_avg": n_chunks.float().mean().item(),
        "tf_prob": teacher_force_prob,
        "n_tf_samples": tf_count,
        "n_ro_samples": ro_count,
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# Epoch loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: ChunkDenoiser,
    scheduler: DDIMScheduler,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    norm_stats: ChunkNormStats,
    cursor_coefs: torch.Tensor,
    cursor_intercepts: torch.Tensor,
    epoch: int,
    tf_start: float = 1.0,
    tf_decay: float = 0.95,
    tf_min: float = 0.2,
    lambda_z: float = 1.0,
    lambda_v: float = 1.0,
    rollout_ddim_steps: int = 10,
    grad_clip: float = 1.0,
    ema: Optional[EMA] = None,
) -> Dict[str, float]:
    """Train one epoch. Returns averaged metrics."""
    model.train()
    tf_prob = compute_teacher_force_prob(epoch, tf_start, tf_decay, tf_min)

    running_keys = [
        "loss", "tf_loss", "tf_loss_z", "tf_loss_v",
        "ro_loss", "ro_loss_z", "ro_loss_v", "n_chunks_avg",
    ]
    running = {k: 0.0 for k in running_keys}
    n_batches = 0

    for batch in train_loader:
        optimizer.zero_grad()

        loss, metrics = train_step(
            model, scheduler, batch, device, norm_stats,
            cursor_coefs, cursor_intercepts,
            teacher_force_prob=tf_prob,
            lambda_z=lambda_z, lambda_v=lambda_v,
            rollout_ddim_steps=rollout_ddim_steps,
        )

        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if ema is not None:
            ema.update(model)

        for k in running:
            running[k] += metrics.get(k, 0.0)
        n_batches += 1

    avg = {k: v / max(n_batches, 1) for k, v in running.items()}
    avg["tf_prob"] = tf_prob
    avg["epoch"] = epoch
    return avg


@torch.no_grad()
def evaluate_loss(
    model: ChunkDenoiser,
    scheduler: DDIMScheduler,
    loader,
    device: torch.device,
    norm_stats: ChunkNormStats,
    cursor_coefs: torch.Tensor,
    cursor_intercepts: torch.Tensor,
    lambda_z: float = 1.0,
    lambda_v: float = 1.0,
    rollout_ddim_steps: int = 10,
) -> Dict[str, float]:
    """Evaluate denoising loss with pure rollout (no teacher forcing)."""
    model.eval()

    running_keys = ["loss", "ro_loss", "ro_loss_z", "ro_loss_v"]
    running = {k: 0.0 for k in running_keys}
    n_batches = 0

    for batch in loader:
        _, metrics = train_step(
            model, scheduler, batch, device, norm_stats,
            cursor_coefs, cursor_intercepts,
            teacher_force_prob=0.0,  # pure rollout
            lambda_z=lambda_z, lambda_v=lambda_v,
            rollout_ddim_steps=rollout_ddim_steps,
        )
        for k in running:
            running[k] += metrics.get(k, 0.0)
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in running.items()}

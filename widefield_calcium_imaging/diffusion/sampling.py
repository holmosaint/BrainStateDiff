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
from .model import MultiSubjectSlidingWindowModel

@torch.no_grad()
def ddim_sample_target(
    model: MultiSubjectSlidingWindowModel,
    scheduler: DDIMScheduler,
    history_vol: torch.Tensor,
    image_emb: torch.Tensor,
    cond_mask: torch.Tensor,
    subject_idx: torch.Tensor,
    target_frames: int,
    num_steps: int,
    device: torch.device,
    null_cond_prob: float = 0.0,
    prob_focus_present: float = 0.0,
) -> torch.Tensor:
    model.eval()
    scheduler.set_timesteps(int(num_steps), device=device)

    B, C, _T_hist, H_lat, W_lat = history_vol.shape
    sample = torch.randn(B, C, target_frames, H_lat, W_lat, device=device)

    for t in scheduler.timesteps:
        ts = torch.full((B,), int(t), device=device, dtype=torch.long)
        volume_in = torch.cat([history_vol, sample], dim=2)
        pred_all = model(
            volume_in,
            ts,
            image_emb=image_emb,
            cond_mask=cond_mask,
            subject_idx=subject_idx,
            null_cond_prob=null_cond_prob,
            prob_focus_present=prob_focus_present,
        )
        pred_tgt = pred_all[:, :, -target_frames:]
        step_out = scheduler.step(pred_tgt, t, sample)
        sample = step_out.prev_sample
    return sample
@torch.no_grad()
def guided_ddim_sample_target(
    model: MultiSubjectSlidingWindowModel,
    scheduler: DDIMScheduler,
    history_vol: torch.Tensor,      # (B,C,T_hist,H,W) normalized
    image_emb: torch.Tensor,        # (B,P,E)
    cond_mask: torch.Tensor,        # (B,T_total)
    subject_idx: torch.Tensor,      # (B,)
    target_frames: int,
    num_steps: int,
    device: torch.device,
    cfg_scale: float,
    null_cond_prob: float,
    prob_focus_present: float,
) -> torch.Tensor:
    """
    Frame-specific CFG with subject conditioning.

    cfg_scale semantics:
      - 1.0 => conditional only
      - 0.0 => unconditional only (cond_mask=0)
    """
    model.eval()
    scheduler.set_timesteps(int(num_steps), device=device)

    B, C, _T_hist, H_lat, W_lat = history_vol.shape
    sample = torch.randn(B, C, target_frames, H_lat, W_lat, device=device)

    cond_mask = cond_mask.to(device)
    uncond_mask = torch.zeros_like(cond_mask)

    for t in scheduler.timesteps:
        ts = torch.full((B,), int(t), device=device, dtype=torch.long)
        volume_in = torch.cat([history_vol, sample], dim=2)

        if cfg_scale == 1.0:
            pred_all = model(
                volume_in,
                ts,
                image_emb=image_emb,
                cond_mask=cond_mask,
                subject_idx=subject_idx,
                null_cond_prob=null_cond_prob,
                prob_focus_present=prob_focus_present,
            )
        elif cfg_scale == 0.0:
            pred_all = model(
                volume_in,
                ts,
                image_emb=image_emb,
                cond_mask=uncond_mask,
                subject_idx=subject_idx,
                null_cond_prob=0.0,
                prob_focus_present=prob_focus_present,
            )
        else:
            pred_uncond = model(
                volume_in,
                ts,
                image_emb=image_emb,
                cond_mask=uncond_mask,
                subject_idx=subject_idx,
                null_cond_prob=0.0,
                prob_focus_present=prob_focus_present,
            )
            pred_cond = model(
                volume_in,
                ts,
                image_emb=image_emb,
                cond_mask=cond_mask,
                subject_idx=subject_idx,
                null_cond_prob=null_cond_prob,
                prob_focus_present=prob_focus_present,
            )
            pred_all = pred_uncond + cfg_scale * (pred_cond - pred_uncond)

        pred_tgt = pred_all[:, :, -target_frames:]
        step_out = scheduler.step(pred_tgt, t, sample)
        sample = step_out.prev_sample
    return sample

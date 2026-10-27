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
from .vae import UNet2D

class EMA:
    def __init__(self, beta: float):
        self.beta = float(beta)

    @torch.no_grad()
    def update_model_average(self, ma_model: nn.Module, current_model: nn.Module) -> None:
        ma_params = list(ma_model.parameters())
        cur_params = list(current_model.parameters())
        if len(ma_params) != len(cur_params):
            raise RuntimeError("EMA models do not have the same number of parameters.")
        for ma_p, cur_p in zip(ma_params, cur_params):
            ma_p.data.mul_(self.beta).add_(cur_p.data, alpha=(1.0 - self.beta))

        # Copy buffers for completeness (e.g., norms)
        ma_bufs = list(ma_model.buffers())
        cur_bufs = list(current_model.buffers())
        if len(ma_bufs) == len(cur_bufs):
            for ma_b, cur_b in zip(ma_bufs, cur_bufs):
                ma_b.data.copy_(cur_b.data)

@dataclass
class LatentStats:
    mean: torch.Tensor  # (C,)
    std: torch.Tensor   # (C,)
    count: int

    def to(self, device: torch.device, dtype: torch.dtype = torch.float32) -> "LatentStats":
        return LatentStats(
            mean=self.mean.to(device=device, dtype=dtype),
            std=self.std.to(device=device, dtype=dtype),
            count=int(self.count),
        )

    def _view(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        C = x.shape[1] if x.dim() >= 2 else x.shape[0]
        if self.mean.numel() != C or self.std.numel() != C:
            raise ValueError(f"LatentStats channel mismatch: stats C={self.mean.numel()} vs tensor C={C}")
        mean = self.mean.view(1, C, 1, 1, 1)
        std = self.std.view(1, C, 1, 1, 1)
        return mean, std

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean, std = self._view(x)
        return (x - mean) / (std + 1e-8)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        mean, std = self._view(x)
        return x * (std + 1e-8) + mean

@dataclass
class TrainState:
    global_step: int = 0

@torch.no_grad()
def encode_frames_with_vae(
    vae: UNet2D,
    history_clip: torch.Tensor,  # (B, H_hist, C, H, W)
    target_clip: torch.Tensor,   # (B, F, C, H, W)
    latent_stats: Optional[LatentStats] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, H_hist, C, H, W = history_clip.shape
    F_t = target_clip.shape[1]
    hist_frames = history_clip.reshape(B * H_hist, C, H, W)
    targ_frames = target_clip.reshape(B * F_t, C, H, W)

    hist_mean, _ = vae.encode(hist_frames)
    targ_mean, _ = vae.encode(targ_frames)

    _, C_lat, H_lat, W_lat = hist_mean.shape
    history_vol = hist_mean.view(B, H_hist, C_lat, H_lat, W_lat).permute(0, 2, 1, 3, 4)
    target_vol = targ_mean.view(B, F_t, C_lat, H_lat, W_lat).permute(0, 2, 1, 3, 4)

    if latent_stats is not None:
        history_vol = latent_stats.normalize(history_vol)
        target_vol = latent_stats.normalize(target_vol)

    return history_vol, target_vol

@torch.no_grad()
def decode_latent_video_tensor(
    vae: UNet2D,
    latent_video: torch.Tensor,   # (B,C,F,H_lat,W_lat) normalized if stats provided
    latent_stats: Optional[LatentStats] = None,
) -> torch.Tensor:
    if latent_stats is not None:
        latent_video = latent_stats.denormalize(latent_video)

    B, C, F_t, H_lat, W_lat = latent_video.shape
    lat = latent_video.permute(0, 2, 1, 3, 4).reshape(B * F_t, C, H_lat, W_lat)
    recon = vae.decode(lat)  # (B*F, C_out, H, W)
    if recon.ndim != 4:
        raise ValueError(f"Unexpected VAE decode output shape: {tuple(recon.shape)}")
    _, C_out, H, W = recon.shape
    recon = recon.view(B, F_t, C_out, H, W)
    return recon

def add_log_normal_history_noise(
    history_vol: torch.Tensor,    # (B,C,T_hist,H,W) normalized
    p_mean: float = -3.0,
    p_std: float = 2.0,
    prob: float = 1.0,
) -> torch.Tensor:
    """
    Add independent Gaussian noise to history frames:
        sigma = exp(x), x ~ N(p_mean, p_std^2)
        history <- history + sigma * N(0,1)

    Noise is applied per-sample (one sigma per batch element), independently across all pixels/frames/channels.
    """
    if prob <= 0.0:
        return history_vol

    B = history_vol.shape[0]
    device = history_vol.device

    if prob >= 1.0:
        apply = torch.ones(B, device=device, dtype=torch.bool)
    else:
        apply = (torch.rand(B, device=device) < prob)

    if not apply.any():
        return history_vol

    x = torch.randn(B, device=device) * float(p_std) + float(p_mean)  # (B,)
    sigma = torch.exp(x).to(dtype=history_vol.dtype)                  # (B,)
    sigma = sigma.view(B, 1, 1, 1, 1)
    sigma = sigma * apply.to(dtype=history_vol.dtype).view(B, 1, 1, 1, 1)

    noise = torch.randn_like(history_vol) * sigma
    return history_vol + noise

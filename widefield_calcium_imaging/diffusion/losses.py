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
def _gaussian_window(window_size: int, sigma: float, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / g.sum()
    w2d = torch.outer(g, g)
    return w2d

def _ssim_per_channel(x: torch.Tensor, y: torch.Tensor, window: torch.Tensor,
                      C1: float, C2: float, pad: int) -> torch.Tensor:
    """
    x, y: (B,1,H,W)
    window: (1,1,ks,ks)
    returns: (B,) SSIM
    """
    mu_x = F.conv2d(x, window, padding=pad, groups=1)
    mu_y = F.conv2d(y, window, padding=pad, groups=1)

    mu_x2  = mu_x.pow(2)
    mu_y2  = mu_y.pow(2)
    mu_xy  = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=pad, groups=1) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=pad, groups=1) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=pad, groups=1) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + 1e-12)
    return ssim_map.mean(dim=(1, 2, 3))

def ms_ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 2.0,    # if inputs are tanh outputs in [-1,1], range=2
    levels: int = 5,
    weight: Tuple[float,...] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
) -> torch.Tensor:
    """
    Multi-Scale SSIM between x and y (both in [B, 1, H, W]).

    Returns mean MS-SSIM over the batch (higher is better).
    """
    assert x.shape == y.shape, "x and y must have same shape"
    B, C, H, W = x.shape
    assert C == 1, "ms_ssim expects single-channel input per call"

    device, dtype = x.device, x.dtype
    window = _gaussian_window(window_size, sigma, device, dtype).unsqueeze(0).unsqueeze(0)
    pad = window_size // 2

    # constants from original SSIM
    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    # weights per scale
    assert levels == len(weight), "levels and weight length must match"

    mcs = []
    ssim_vals = []
    x_s, y_s = x, y

    for i in range(levels):
        ssim_val = _ssim_per_channel(x_s, y_s, window, C1, C2, pad)  # (B,)
        ssim_vals.append(ssim_val)

        # contrast-structure component (mcs)
        mu_x = F.conv2d(x_s, window, padding=pad)
        mu_y = F.conv2d(y_s, window, padding=pad)
        sigma_x2 = F.conv2d(x_s * x_s, window, padding=pad) - mu_x.pow(2)
        sigma_y2 = F.conv2d(y_s * y_s, window, padding=pad) - mu_y.pow(2)
        sigma_xy = F.conv2d(x_s * y_s, window, padding=pad) - (mu_x * mu_y)

        cs_map = (2.0 * sigma_xy + C2) / (sigma_x2 + sigma_y2 + C2 + 1e-12)
        mcs.append(cs_map.mean(dim=(1, 2, 3)))  # (B,)

        # downsample for next scale (except last)
        if i < levels - 1:
            x_s = F.avg_pool2d(x_s, kernel_size=2, stride=2, ceil_mode=True)
            y_s = F.avg_pool2d(y_s, kernel_size=2, stride=2, ceil_mode=True)

    mcs_stack = torch.stack(mcs[:-1], dim=0)           # (levels-1, B)
    ssim_last = ssim_vals[-1]                          # (B,)

    # Aggregate across scales
    pow1 = torch.prod(mcs_stack ** torch.tensor(weight[:-1], device=device).view(-1, 1), dim=0)
    ms_ssim_val = pow1 * (ssim_last ** weight[-1])
    return ms_ssim_val.mean()  # scalar

def ms_ssim_loss(
    x: torch.Tensor, y: torch.Tensor,
    channels_as_time: bool = True,
    data_range: float = 2.0,
    **kwargs
) -> torch.Tensor:
    """
    If channels_as_time=True, treat C as T and average MS-SSIM over frames.
    Otherwise, compute per-channel MS-SSIM and average.
    Returns: loss = 1 - MS-SSIM (so lower is better).
    """
    B, C, H, W = x.shape
    vals = []
    for c in range(C):
        xs = x[:, c:c+1]
        ys = y[:, c:c+1]
        vals.append(ms_ssim(xs, ys, data_range=data_range, **kwargs))
    ms = torch.stack(vals).mean()
    return 1.0 - ms

def sobel_edges(img: torch.Tensor) -> torch.Tensor:
    """
    img: (B,1,H,W) -> gradient magnitude (B,1,H,W)
    """
    dtype, device = img.dtype, img.device
    kx = torch.tensor([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]], dtype=dtype, device=device).view(1, 1, 3, 3) / 8.0
    ky = torch.tensor([[ 1,  2,  1],
                       [ 0,  0,  0],
                       [-1, -2, -1]], dtype=dtype, device=device).view(1, 1, 3, 3) / 8.0
    gx = F.conv2d(img, kx, padding=1)
    gy = F.conv2d(img, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)

def sobel_loss(x: torch.Tensor, y: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    L1 between gradient magnitudes, averaged over frames/channels.
    x,y: (B,C,H,W)
    """
    B, C, H, W = x.shape
    losses = []
    for c in range(C):
        gx = sobel_edges(x[:, c:c+1])
        gy = sobel_edges(y[:, c:c+1])
        losses.append(F.l1_loss(gx, gy, reduction="mean"))
    loss = torch.stack(losses).mean()
    return loss

def temporal_consistency_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    mode: str = "diff_l1"
) -> torch.Tensor:
    """
    Match temporal derivatives along the 'channel-as-time' axis.

    modes:
      - "diff_l1" : L1 between (y[:,t+1]-y[:,t]) and (x[:,t+1]-x[:,t]) averaged over t
      - "smooth_l1": same with smooth_l1
    x: target  (B,C,H,W)
    y: recon   (B,C,H,W)
    """
    assert x.shape == y.shape and x.ndim == 4
    if x.size(1) < 2:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    x_diff = x[:, 1:] - x[:, :-1]
    y_diff = y[:, 1:] - y[:, :-1]

    if mode == "diff_l1":
        return F.l1_loss(y_diff, x_diff, reduction="mean")
    elif mode == "smooth_l1":
        return F.smooth_l1_loss(y_diff, x_diff, reduction="mean", beta=0.02)
    else:
        raise ValueError(f"Unknown temporal mode: {mode}")

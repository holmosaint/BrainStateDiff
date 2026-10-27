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
from .losses import ms_ssim_loss, sobel_loss, temporal_consistency_loss

class SpatialAdaGN(nn.Module):
    def __init__(self, num_channels: int, num_groups: int, cond_channels: int):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.to_gamma = nn.Conv2d(cond_channels, num_channels, kernel_size=1)
        self.to_beta  = nn.Conv2d(cond_channels, num_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, cond_feat: torch.Tensor) -> torch.Tensor:
        x = self.gn(x)
        if x.shape[-2:] != cond_feat.shape[-2:]:
            cond_feat = F.interpolate(cond_feat, size=x.shape[-2:], mode='bilinear', align_corners=False)
        gamma = self.to_gamma(cond_feat)
        beta  = self.to_beta(cond_feat)
        return x * (1 + gamma) + beta

class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int = 32,
        activation: str = 'relu',
        downsample: bool = False,
        upsample: bool = False,
        use_spatial_adagn: bool = False,
        cond_channels: int = 0,
        dilation: int = 1,
    ):
        super().__init__()

        ng = min(num_groups, out_channels)
        while out_channels % ng != 0 and ng > 1:
            ng -= 1

        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'silu':
            self.act = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unknown activation: {activation}")

        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        elif upsample:
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                                   padding=dilation, dilation=dilation)

        self.use_spatial_adagn = use_spatial_adagn
        if use_spatial_adagn:
            assert cond_channels > 0
            self.norm1 = SpatialAdaGN(out_channels, ng, cond_channels)
        else:
            self.norm1 = nn.GroupNorm(ng, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=dilation, dilation=dilation)
        self.norm2 = SpatialAdaGN(out_channels, ng, cond_channels) if use_spatial_adagn else nn.GroupNorm(ng, out_channels)

        if in_channels != out_channels or downsample or upsample:
            if downsample:
                self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
            elif upsample:
                self.residual = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                              nn.Conv2d(in_channels, out_channels, kernel_size=1))
            else:
                self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor, cond_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h, cond_feat) if self.use_spatial_adagn else self.norm1(h)
        h = self.act(h)
        h = self.conv2(h)
        h = self.norm2(h, cond_feat) if self.use_spatial_adagn else self.norm2(h)
        return self.act(h + self.residual(x))

class UNet2D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        down_block_dims: List[int] = [64, 128, 256, 512],
        up_block_dims: List[int] = [512, 256, 128, 64],
        latent_dim: int = 512,
        num_groups_per_norm: int = 32,
        activation: str = 'relu',
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.down_block_dims = down_block_dims
        self.up_block_dims = up_block_dims
        self.latent_dim = latent_dim

        self.input_conv = nn.Conv2d(in_channels, down_block_dims[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        prev = down_block_dims[0]
        for dim in down_block_dims:
            self.down_blocks.append(
                ResidualConvBlock(prev, dim, num_groups_per_norm, activation,
                                  downsample=(prev != dim), use_spatial_adagn=False)
            )
            prev = dim

        self.bottleneck_dim = down_block_dims[-1]
        self.to_mean   = nn.Conv2d(self.bottleneck_dim, latent_dim, kernel_size=1)
        self.to_logvar = nn.Conv2d(self.bottleneck_dim, latent_dim, kernel_size=1)

        self.from_latent = nn.Conv2d(latent_dim, up_block_dims[0], kernel_size=1)

        self.up_blocks = nn.ModuleList()
        prev = up_block_dims[0]
        for i, dim in enumerate(up_block_dims):
            self.up_blocks.append(
                ResidualConvBlock(prev, dim, num_groups_per_norm, activation,
                                  upsample=(i < len(up_block_dims) - 1),
                                  use_spatial_adagn=True, cond_channels=dim,
                                  dilation=1 if i < len(up_block_dims) - 1 else 2)
            )
            prev = dim

        self.cond_proj = nn.ModuleList([nn.Conv2d(latent_dim, ch, kernel_size=1) for ch in up_block_dims])

        self.output_conv  = nn.Conv2d(up_block_dims[-1], out_channels, kernel_size=3, padding=1)
        self.output_scale = nn.Parameter(torch.tensor(1.0))

    # --- VAE ---
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_conv(x)
        for blk in self.down_blocks:
            h = blk(h)
        mean   = self.to_mean(h)
        logvar = self.to_logvar(h)
        return mean, logvar

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mean + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z)
        cond_z = z
        cond_feats = []
        for i, proj in enumerate(self.cond_proj):
            if i == 0:
                cond_z = F.interpolate(cond_z, scale_factor=2, mode='bilinear', align_corners=False)
            else:
                if i < len(self.up_blocks) - 1:
                    cond_z = F.interpolate(cond_z, scale_factor=2, mode='bilinear', align_corners=False)
            cond_feats.append(proj(cond_z))
        for i, blk in enumerate(self.up_blocks):
            h = blk(h, cond_feat=cond_feats[i])
        out = torch.tanh(self.output_conv(h)) * self.output_scale
        return out

    def forward(self, x: torch.Tensor, return_latent: bool = False) -> dict:
        mean, logvar = self.encode(x)
        z     = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        out = {'recon': recon, 'mean': mean, 'logvar': logvar}
        if return_latent:
            out['z'] = z
        return out

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = 0.0; self.count = 0
    def update(self, v, n=1): self.val = v; self.sum += v*n; self.count += n; self.avg = self.sum / self.count

class KLDivergenceLossFreeBits(nn.Module):
    def __init__(self, tau: float = 0.05):
        super().__init__()
        self.tau = tau
    def forward(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kl = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())    # (B,C,H,W)
        kl = kl.mean(dim=[2,3])                                   # (B,C)
        kl = torch.clamp(kl, min=self.tau)
        return kl.mean()

class BetaScheduler:
    def __init__(self, warmup_steps=1000, start_beta=0.0, end_beta=1.0, schedule_type='linear'):
        self.warmup_steps = warmup_steps
        self.start_beta   = start_beta
        self.end_beta     = end_beta
        self.schedule_type = schedule_type
        self.current_step  = 0
    def get_beta(self) -> float:
        if self.current_step >= self.warmup_steps: return self.end_beta
        p = self.current_step / self.warmup_steps
        if self.schedule_type == 'linear':
            return self.start_beta + (self.end_beta - self.start_beta) * p
        elif self.schedule_type == 'cosine':
            return self.start_beta + (self.end_beta - self.start_beta) * (1 - np.cos(p * np.pi)) / 2
        else:
            return self.end_beta
    def step(self): self.current_step += 1

def pyramid_recon_loss(recon: torch.Tensor, target: torch.Tensor, loss_type='mse', num_scales=3) -> torch.Tensor:
    def _loss(a,b): return F.mse_loss(a,b) if loss_type=='mse' else F.l1_loss(a,b)
    H, W = target.shape[-2], target.shape[-1]
    total = 0.0; denom = 0.0
    for s in range(num_scales):
        scale = 2 ** s
        if scale > 1:
            t = F.interpolate(target, size=(H//scale, W//scale), mode='bilinear', align_corners=False)
            r = F.interpolate(recon,  size=(H//scale, W//scale), mode='bilinear', align_corners=False)
        else:
            t, r = target, recon
        w = 1.0 / scale
        total += w * _loss(r, t); denom += w
    return total / denom

def compute_losses(recon, data, args):
    """
    Mix-and-match losses. Inputs are (B,C,H,W) with C possibly = #frames.
    """
    losses = {}

    # Base pyramid recon loss (MSE/MAE)
    if args.w_pyr > 0.0:
        losses['pyr'] = pyramid_recon_loss(recon, data, loss_type=args.recon_loss_type, num_scales=args.pyr_scales) * args.w_pyr
    else:
        losses['pyr'] = torch.tensor(0.0, device=recon.device)

    # MS-SSIM (1 - MS-SSIM)
    if args.w_msssim > 0.0:
        losses['msssim'] = ms_ssim_loss(recon, data, channels_as_time=args.channels_as_time) * args.w_msssim
    else:
        losses['msssim'] = torch.tensor(0.0, device=recon.device)

    # Sobel edge loss
    if args.w_sobel > 0.0:
        losses['sobel'] = sobel_loss(recon, data) * args.w_sobel
    else:
        losses['sobel'] = torch.tensor(0.0, device=recon.device)

    # Temporal consistency loss
    if args.w_temp > 0.0 and data.size(1) > 1:
        losses['temp'] = temporal_consistency_loss(data, recon, mode=args.temp_mode) * args.w_temp
    else:
        losses['temp'] = torch.tensor(0.0, device=recon.device)

    # Sum
    total_px = losses['pyr'] + losses['msssim'] + losses['sobel'] + losses['temp']
    return losses, total_px

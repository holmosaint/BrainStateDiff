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
from . import latent as base
from .vae import UNet2D
from .model import MultiSubjectSlidingWindowModel
from .data import MultiSubjectSlidingWindowDataset

def train_one_epoch(model: MultiSubjectSlidingWindowModel, ema: Optional[base.EMA], ema_model: Optional[MultiSubjectSlidingWindowModel], scheduler: DDIMScheduler, dataloader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, args: argparse.Namespace, logger, state: base.TrainState, vae: UNet2D, latent_stats: base.LatentStats, train_dataset: MultiSubjectSlidingWindowDataset, viz_dir: Path, target_frames: int) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    pbar = tqdm(dataloader, desc=f'Train epoch {epoch}')
    for batch in pbar:
        history_clip = batch.history_clip.to(device, non_blocking=True)
        target_clip = batch.target_clip.to(device, non_blocking=True)
        (history_vol, target_vol) = base.encode_frames_with_vae(vae, history_clip, target_clip, latent_stats=latent_stats)
        history_vol = base.add_log_normal_history_noise(history_vol=history_vol, p_mean=args.history_noise_p_mean, p_std=args.history_noise_p_std, prob=args.history_noise_prob)
        image_emb = batch.image_emb.to(device, non_blocking=True)
        cond_mask = batch.cond_mask.to(device, non_blocking=True)
        subject_idx = batch.subject_idx.to(device, non_blocking=True)
        B = target_vol.shape[0]
        if args.history_drop_prob > 0.0:
            drop_hist = torch.rand(B, device=device) < float(args.history_drop_prob)
            if drop_hist.any():
                keep = (~drop_hist).to(dtype=history_vol.dtype).view(B, 1, 1, 1, 1)
                history_vol = history_vol * keep
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(target_vol)
        noisy_target = scheduler.add_noise(target_vol, noise, timesteps)
        volume_in = torch.cat([history_vol, noisy_target], dim=2)
        pred_all = model(volume_in, timesteps, image_emb=image_emb, cond_mask=cond_mask, subject_idx=subject_idx, null_cond_prob=args.cfg_drop_prob, prob_focus_present=args.prob_focus_present)
        pred_tgt = pred_all[:, :, -target_frames:]
        if args.prediction_type == 'epsilon':
            loss = F.mse_loss(pred_tgt, noise)
        else:
            loss = F.mse_loss(pred_tgt, target_vol)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        cur_loss = float(loss.item())
        total_loss += cur_loss * int(B)
        total_count += int(B)
        state.global_step += 1
        pbar.set_postfix({'loss': f'{cur_loss:.4f}'})
        if ema is not None and ema_model is not None and args.ema:
            if state.global_step % args.ema_update_every == 0:
                raw = model.module if hasattr(model, 'module') else model
                if state.global_step < args.ema_start_step:
                    ema_model.load_state_dict(raw.state_dict())
                else:
                    ema.update_model_average(ema_model, raw)
    return total_loss / max(total_count, 1)

@torch.no_grad()
def evaluate(model: MultiSubjectSlidingWindowModel, scheduler: DDIMScheduler, dataloader: DataLoader, device: torch.device, args: argparse.Namespace, vae: UNet2D, latent_stats: base.LatentStats, target_frames: int) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    for batch in tqdm(dataloader, desc='Eval', leave=False):
        history_clip = batch.history_clip.to(device, non_blocking=True)
        target_clip = batch.target_clip.to(device, non_blocking=True)
        (history_vol, target_vol) = base.encode_frames_with_vae(vae, history_clip, target_clip, latent_stats=latent_stats)
        image_emb = batch.image_emb.to(device, non_blocking=True)
        cond_mask = batch.cond_mask.to(device, non_blocking=True)
        subject_idx = batch.subject_idx.to(device, non_blocking=True)
        B = target_vol.shape[0]
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(target_vol)
        noisy_target = scheduler.add_noise(target_vol, noise, timesteps)
        volume_in = torch.cat([history_vol, noisy_target], dim=2)
        pred_all = model(volume_in, timesteps, image_emb=image_emb, cond_mask=cond_mask, subject_idx=subject_idx, null_cond_prob=0.0, prob_focus_present=args.prob_focus_present)
        pred_tgt = pred_all[:, :, -target_frames:]
        if args.prediction_type == 'epsilon':
            loss = F.mse_loss(pred_tgt, noise)
        else:
            loss = F.mse_loss(pred_tgt, target_vol)
        total_loss += float(loss.item()) * int(B)
        total_count += int(B)
    return total_loss / max(total_count, 1)

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
from .vae import AverageMeter, KLDivergenceLossFreeBits, compute_losses

def train_one_epoch(model, dataloader, optimizer, device, epoch, args, beta_scheduler, logger, global_step, is_main=True):
    model.train()
    pyr_m = AverageMeter()
    msssim_m = AverageMeter()
    sobel_m = AverageMeter()
    temp_m = AverageMeter()
    px_m = AverageMeter()
    kl_m = AverageMeter()
    total_m = AverageMeter()
    kl_criterion = KLDivergenceLossFreeBits(tau=0.05)
    for (batch_idx, batch) in enumerate(dataloader):
        data = batch[0].to(device)
        if data.ndim == 3:
            data = data.unsqueeze(1)
        out = model(data, return_latent=True)
        (recon, mean, logvar) = (out['recon'], out['mean'], out['logvar'])
        (loss_dict, px_loss) = compute_losses(recon, data, args)
        pyr_loss = loss_dict['pyr']
        msssim_loss = loss_dict['msssim']
        sobel_l = loss_dict['sobel']
        temp_l = loss_dict['temp']
        kl_loss = kl_criterion(mean, logvar)
        beta = beta_scheduler.get_beta()
        loss = px_loss + beta * kl_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        beta_scheduler.step()
        bs = data.size(0)
        pyr_m.update(pyr_loss.item(), bs)
        msssim_m.update(msssim_loss.item(), bs)
        sobel_m.update(sobel_l.item(), bs)
        temp_m.update(temp_l.item(), bs)
        px_m.update(px_loss.item(), bs)
        kl_m.update(kl_loss.item(), bs)
        total_m.update(loss.item(), bs)
        global_step += 1
        if is_main and (batch_idx + 1) % args.log_interval == 0:
            logger.info(f'Epoch [{epoch}/{args.epochs}] Step [{batch_idx + 1}/{len(dataloader)}] | CURR -> Pyr:{pyr_loss.item():.6f} MS-SSIM:{msssim_loss.item():.6f} Sobel:{sobel_l.item():.6f} Temp:{temp_l.item():.6f} KL:{kl_loss.item():.6f} PX:{px_loss.item():.6f} Total:{loss.item():.6f} (β={beta:.4f}) | AVG -> Pyr:{pyr_m.avg:.6f} MS-SSIM:{msssim_m.avg:.6f} Sobel:{sobel_m.avg:.6f} Temp:{temp_m.avg:.6f} KL:{kl_m.avg:.6f} PX:{px_m.avg:.6f} Total:{total_m.avg:.6f}')
    return ({'px_loss': px_m.avg, 'kl_loss': kl_m.avg, 'total_loss': total_m.avg, 'beta': beta, 'pyr': pyr_m.avg, 'msssim': msssim_m.avg, 'sobel': sobel_m.avg, 'temp': temp_m.avg}, global_step)

@torch.no_grad()
def evaluate(model, dataloader, device, args, beta, phase='val', epoch=0, global_step=0, is_main=True):
    model.eval()
    pyr_m = AverageMeter()
    msssim_m = AverageMeter()
    sobel_m = AverageMeter()
    temp_m = AverageMeter()
    px_m = AverageMeter()
    kl_m = AverageMeter()
    total_m = AverageMeter()
    kl_criterion = KLDivergenceLossFreeBits(tau=0.05)
    first_data = None
    first_recon = None
    for (batch_idx, batch) in enumerate(dataloader):
        data = batch[0].to(device)
        if data.ndim == 3:
            data = data.unsqueeze(1)
        out = model(data, return_latent=True)
        (recon, mean, logvar) = (out['recon'], out['mean'], out['logvar'])
        (loss_dict, px_loss) = compute_losses(recon, data, args)
        kl_loss = kl_criterion(mean, logvar)
        loss = px_loss + beta * kl_loss
        bs = data.size(0)
        pyr_m.update(loss_dict['pyr'].item(), bs)
        msssim_m.update(loss_dict['msssim'].item(), bs)
        sobel_m.update(loss_dict['sobel'].item(), bs)
        temp_m.update(loss_dict['temp'].item(), bs)
        px_m.update(px_loss.item(), bs)
        kl_m.update(kl_loss.item(), bs)
        total_m.update(loss.item(), bs)
        if batch_idx == 0:
            first_data = data
            first_recon = recon
    return {'px_loss': px_m.avg, 'kl_loss': kl_m.avg, 'total_loss': total_m.avg, 'pyr': pyr_m.avg, 'msssim': msssim_m.avg, 'sobel': sobel_m.avg, 'temp': temp_m.avg}

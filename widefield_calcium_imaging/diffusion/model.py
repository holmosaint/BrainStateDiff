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
from . import backbone as base

class MultiSubjectSlidingWindowModel(base.SlidingWindowModel):
    """
    Extends SlidingWindowModel with a learnable subject embedding injected as
    an extra conditioning token appended to the image embedding sequence.
    """

    def __init__(
        self,
        latent_channels: int,
        unet_dim: int,
        cond_dim: int,
        image_emb_dim: int,
        tokens_per_emb: int,
        dim_mults: Sequence[int],
        attn_heads: int,
        attn_dim_head: int,
        init_dim: Optional[int],
        init_kernel_size: int,
        use_sparse_linear_attn: bool,
        num_subjects: int,
    ):
        super().__init__(
            latent_channels=latent_channels,
            unet_dim=unet_dim,
            cond_dim=cond_dim,
            image_emb_dim=image_emb_dim,
            tokens_per_emb=int(tokens_per_emb) + 1,  # +1 for subject token
            dim_mults=dim_mults,
            attn_heads=attn_heads,
            attn_dim_head=attn_dim_head,
            init_dim=init_dim,
            init_kernel_size=init_kernel_size,
            use_sparse_linear_attn=use_sparse_linear_attn,
        )
        self.num_subjects = int(num_subjects)
        self.subject_embedding = nn.Embedding(self.num_subjects, int(image_emb_dim))

    def forward(
        self,
        volume: torch.Tensor,
        timesteps: torch.Tensor,
        image_emb: torch.Tensor,
        cond_mask: torch.Tensor,
        subject_idx: torch.Tensor,
        null_cond_prob: float = 0.0,
        prob_focus_present: float = 0.0,
    ) -> torch.Tensor:
        if subject_idx.dim() != 1 or subject_idx.shape[0] != image_emb.shape[0]:
            raise ValueError(
                f"Expected subject_idx shape (B,), got {tuple(subject_idx.shape)} for B={image_emb.shape[0]}"
            )
        if subject_idx.dtype != torch.long:
            subject_idx = subject_idx.long()

        # Append subject token to image embedding sequence
        subj_token = self.subject_embedding(subject_idx).unsqueeze(1)  # (B,1,E)
        image_plus_subject = torch.cat([image_emb, subj_token], dim=1)  # (B,T+1,E)

        return super().forward(
            volume=volume,
            timesteps=timesteps,
            image_emb=image_plus_subject,
            cond_mask=cond_mask,
            null_cond_prob=null_cond_prob,
            prob_focus_present=prob_focus_present,
        )

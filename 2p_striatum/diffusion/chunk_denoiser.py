"""
chunk_diffusion/chunk_denoiser.py
----------------------------------

Cross-attention Transformer denoiser for chunk-level conditional diffusion.

Architecture:
  1. Condition Encoder — self-attention Transformer over history tokens
     [z_hist, v_hist, c_hist] + trial_type token → condition memory
  2. Denoising Decoder — cross-attention Transformer over noisy future chunk
     tokens, with AdaLayerNorm conditioning from diffusion timestep +
     subject/session identity

The model predicts noise (epsilon) or clean target (x0) for the future chunk
y = [z_future, v_future] of shape (K, D+1).
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class AdaLayerNorm(nn.Module):
    """Adaptive layer norm: LN(x) * (1 + scale) + shift, conditioned on a global vector."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.to_scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim * 2),
        )
        nn.init.zeros_(self.to_scale_shift[-1].weight)
        nn.init.zeros_(self.to_scale_shift[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x: (B, L, dim), cond: (B, cond_dim) or (B, L, cond_dim)."""
        if cond.dim() == 2:
            cond = cond.unsqueeze(1).expand(-1, x.shape[1], -1)
        ss = self.to_scale_shift(cond)
        scale, shift = ss.chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale) + shift


# ---------------------------------------------------------------------------
# Denoising Decoder blocks
# ---------------------------------------------------------------------------

class DenoiserBlock(nn.Module):
    """Cross-attention decoder block with AdaLayerNorm conditioning.

    Each block:
      1. AdaLN-conditioned self-attention over future chunk tokens
      2. Cross-attention: future tokens attend to condition memory
      3. AdaLN-conditioned MLP
    """

    def __init__(self, model_dim: int, cond_dim: int, n_heads: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        # Self-attention with AdaLN
        self.norm_self = AdaLayerNorm(model_dim, cond_dim)
        self.self_attn = nn.MultiheadAttention(model_dim, n_heads, dropout=dropout, batch_first=True)

        # Cross-attention
        self.norm_cross_q = nn.LayerNorm(model_dim)
        self.norm_cross_kv = nn.LayerNorm(model_dim)
        self.cross_attn = nn.MultiheadAttention(model_dim, n_heads, dropout=dropout, batch_first=True)

        # MLP with AdaLN
        self.norm_mlp = AdaLayerNorm(model_dim, cond_dim)
        mlp_dim = int(model_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, mlp_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, model_dim),
            nn.Dropout(dropout),
        )
        # Zero-init output projections for stable training
        nn.init.zeros_(self.mlp[-2].weight)
        nn.init.zeros_(self.mlp[-2].bias)

    def forward(self, x: torch.Tensor, cond_memory: torch.Tensor,
                global_cond: torch.Tensor) -> torch.Tensor:
        """
        x: (B, K, model_dim) — future chunk tokens
        cond_memory: (B, H+1, model_dim) — condition encoder output
        global_cond: (B, cond_dim) — timestep + identity conditioning
        """
        # Self-attention
        h = self.norm_self(x, global_cond)
        h, _ = self.self_attn(h, h, h)
        x = x + h

        # Cross-attention
        q = self.norm_cross_q(x)
        kv = self.norm_cross_kv(cond_memory)
        h, _ = self.cross_attn(q, kv, kv)
        x = x + h

        # MLP
        x = x + self.mlp(self.norm_mlp(x, global_cond))
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class ChunkDenoiser(nn.Module):
    """Conditional chunk-level diffusion denoiser.

    Predicts noise (or clean target) for a future chunk of shape (K, D+1)
    where D = latent_dim and the +1 is velocity.

    Conditioning:
      - History: z_hist (H, D), v_hist (H, 1), c_hist (H, 1) — projected per-frame
      - Trial type embedding — appended as an extra token
      - Subject + session identity (factored) — added to global conditioning
      - Diffusion timestep — added to global conditioning
      - Chunk offset (relative timing in trial) — added to global conditioning

    Architecture:
      - Condition Encoder: per-frame projection + positional embedding (no self-attention)
      - Denoising Decoder: cross-attention Transformer with AdaLN conditioning
    """

    def __init__(
        self,
        latent_dim: int,
        chunk_len: int = 8,
        history_len: int = 16,
        model_dim: int = 256,
        cond_dim: int = 256,
        denoise_depth: int = 6,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_sessions: int = 13,
        num_subjects: int = 3,
        session_to_subject: Optional[Dict[int, int]] = None,
        num_trial_types: int = 2,
        max_chunk_offset: int = 256,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.chunk_len = chunk_len
        self.history_len = history_len
        self.model_dim = model_dim
        self.cond_dim = cond_dim
        self.token_dim = latent_dim + 1  # z + v

        # --- Session/subject identity ---
        self.subject_emb = nn.Embedding(num_subjects, cond_dim)
        nn.init.normal_(self.subject_emb.weight, std=0.02)

        self.session_emb = nn.Embedding(num_sessions, cond_dim)
        nn.init.zeros_(self.session_emb.weight)  # residual, zero-init

        if session_to_subject is not None:
            s2s = torch.zeros(num_sessions, dtype=torch.long)
            for sess_idx, subj_idx in session_to_subject.items():
                s2s[int(sess_idx)] = int(subj_idx)
            self.register_buffer("session_to_subject_map", s2s)
        else:
            self.register_buffer("session_to_subject_map",
                                 torch.zeros(num_sessions, dtype=torch.long))

        self.identity_proj = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        nn.init.zeros_(self.identity_proj[-1].weight)
        nn.init.zeros_(self.identity_proj[-1].bias)

        # --- Trial type ---
        self.trial_type_emb = nn.Embedding(num_trial_types, model_dim)
        nn.init.normal_(self.trial_type_emb.weight, std=0.02)

        # --- Diffusion timestep ---
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(cond_dim),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # --- Chunk offset (relative trial timing) ---
        self.offset_emb = nn.Sequential(
            SinusoidalPosEmb(cond_dim),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        nn.init.zeros_(self.offset_emb[-1].weight)
        nn.init.zeros_(self.offset_emb[-1].bias)

        # --- Condition Encoder (no self-attention) ---
        # History token projection: [z, v, c] -> model_dim
        self.hist_proj = nn.Linear(latent_dim + 2, model_dim)  # D + v(1) + c(1)
        # Positional embeddings for history + trial_type token
        self.cond_pos_emb = nn.Parameter(torch.randn(history_len + 1, model_dim) * 0.02)
        self.cond_out_norm = nn.LayerNorm(model_dim)

        # --- Denoising Decoder ---
        # Noisy future projection: [z, v] -> model_dim
        self.future_proj = nn.Linear(self.token_dim, model_dim)
        # Positional embeddings for chunk frame positions (0..K-1)
        self.future_pos_emb = nn.Parameter(torch.randn(chunk_len, model_dim) * 0.02)
        self.denoise_blocks = nn.ModuleList([
            DenoiserBlock(model_dim, cond_dim, n_heads, mlp_ratio, dropout)
            for _ in range(denoise_depth)
        ])
        self.denoise_out_norm = AdaLayerNorm(model_dim, cond_dim)

        # --- Output head ---
        self.out_proj = nn.Linear(model_dim, self.token_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        y_noisy: torch.Tensor,          # (B, K, D+1) noised future chunk
        timesteps: torch.Tensor,         # (B,) diffusion timestep indices
        z_hist: torch.Tensor,            # (B, H, D) history latents
        v_hist: torch.Tensor,            # (B, H, 1) history velocity
        c_hist: torch.Tensor,            # (B, H, 1) history cursor
        trial_type_ids: torch.Tensor,    # (B,) trial type
        session_ids: torch.Tensor,       # (B,) session index
        subject_ids: torch.Tensor,       # (B,) subject index
        chunk_offset: torch.Tensor,      # (B,) anchor offset relative to frame 46
    ) -> torch.Tensor:                   # (B, K, D+1) predicted noise/x0
        B = y_noisy.shape[0]

        # --- Global conditioning vector ---
        t_emb = self.time_emb(timesteps)                           # (B, cond_dim)
        id_emb = self.subject_emb(subject_ids) + self.session_emb(session_ids)
        id_emb = self.identity_proj(id_emb)                        # (B, cond_dim)
        off_emb = self.offset_emb(chunk_offset.float())            # (B, cond_dim)

        global_cond = t_emb + id_emb + off_emb                    # (B, cond_dim)

        # --- Condition Encoder (projection + positional embedding, no self-attention) ---
        hist_tokens = torch.cat([z_hist, v_hist, c_hist], dim=-1)  # (B, H, D+2)
        hist_tokens = self.hist_proj(hist_tokens)                   # (B, H, model_dim)

        tt_token = self.trial_type_emb(trial_type_ids).unsqueeze(1) # (B, 1, model_dim)

        cond_seq = torch.cat([hist_tokens, tt_token], dim=1)       # (B, H+1, model_dim)
        cond_seq = cond_seq + self.cond_pos_emb                    # (B, H+1, model_dim)
        cond_memory = self.cond_out_norm(cond_seq)                 # (B, H+1, model_dim)

        # --- Denoising Decoder ---
        x = self.future_proj(y_noisy)                              # (B, K, model_dim)
        x = x + self.future_pos_emb                                # (B, K, model_dim)

        for block in self.denoise_blocks:
            x = block(x, cond_memory, global_cond)

        x = self.denoise_out_norm(x, global_cond)                 # (B, K, model_dim)
        out = self.out_proj(x)                                     # (B, K, D+1)
        return out

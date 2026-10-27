"""
NeuralDiT-TS -- Diffusion Transformer for neural activity with temporal decomposition.

Same architecture as NeuralDiT, but with an additional baseline neural state
condition injected into the global AdaLN-Zero vector via subject-specific
"stitcher" projections.

Conditioning (use_baseline=True, default):
  global_cond = [time_emb ; mean_pool(image) ; subject_emb ; baseline_emb]

  - baseline_emb: early-response neural activity (avg of first ~10 time bins),
    projected through a subject-specific stitcher MLP to a fixed dim.
  - Cross-attention to image tokens is unchanged from NeuralDiT.

Conditioning (use_baseline=False):
  global_cond = [time_emb ; mean_pool(image) ; subject_emb]

  No neural history is used as a condition. The forward signature is unchanged
  for compatibility -- the `baseline` argument is accepted but ignored.

Diffusion target:
  Evoked neural activity (avg of bins ~15-25), same as NeuralDiT.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from TVSD.subjects import (
    MAX_NEURONS,
    NUM_SUBJECTS,
    get_subject_neuron_counts,
)
from TVSD.diffusion.blocks import (
    TimestepEmbedder,
    FrozenViTMAEEncoder,
    DiTCrossAttnBlock,
)


class NeuralDiTTS(nn.Module):
    """
    Diffusion Transformer for neural activity with baseline state conditioning.

    Compared to NeuralDiT:
      - Accepts an additional `baseline` input (2B, MAX_NEURONS), the early
        neural response averaged over the first ~10 time bins.
      - Each subject has a "stitcher" MLP that projects the baseline from
        (n_neurons,) -> (baseline_dim,).
      - The stitched baseline is concatenated into the global AdaLN-Zero vector:
        global_cond = [time_emb ; mean_pool(image) ; subject_emb ; baseline_emb]
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        cond_dim: int = 384,
        time_dim: int = 384,
        subject_emb_dim: int = 128,
        baseline_dim: int = 256,
        depth: int = 8,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        cond_drop_prob: float = 0.1,
        vit_model_id: str = "facebook/vit-mae-base",
        use_baseline: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.baseline_dim = baseline_dim
        self.cond_drop_prob = cond_drop_prob
        self.use_baseline = use_baseline
        self.num_subjects = NUM_SUBJECTS
        self.neuron_counts = get_subject_neuron_counts()

        global_cond_dim = time_dim + cond_dim + subject_emb_dim
        if use_baseline:
            global_cond_dim += baseline_dim

        # ---- Image encoder (frozen) ----
        self.image_encoder = FrozenViTMAEEncoder(model_id=vit_model_id)
        vit_dim = self.image_encoder.embed_dim  # 768
        self.image_proj = nn.Linear(vit_dim, cond_dim)

        # Null image tokens for CFG
        self.null_image_token = nn.Parameter(torch.zeros(1, 1, cond_dim))

        # ---- Timestep embedding ----
        self.time_embed = TimestepEmbedder(time_dim)

        # ---- Subject embedding ----
        self.subject_embed = nn.Embedding(NUM_SUBJECTS, subject_emb_dim)

        # ---- Subject-specific baseline stitchers ----
        # Project per-subject baseline (n_neurons,) -> (baseline_dim,)
        if use_baseline:
            self.baseline_stitchers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.neuron_counts[s], baseline_dim),
                    nn.SiLU(),
                    nn.Linear(baseline_dim, baseline_dim),
                )
                for s in range(NUM_SUBJECTS)
            ])

            # Null baseline for CFG (learned)
            self.null_baseline = nn.Parameter(torch.zeros(1, baseline_dim))

        # ---- Subject-specific input projections (for diffusion target) ----
        self.input_projs = nn.ModuleList(
            [nn.Linear(1, hidden_dim) for _ in range(NUM_SUBJECTS)]
        )

        # ---- Subject-specific positional embeddings ----
        self.pos_embeds = nn.ParameterList([
            nn.Parameter(torch.zeros(MAX_NEURONS, hidden_dim))
            for _ in range(NUM_SUBJECTS)
        ])
        for pe in self.pos_embeds:
            nn.init.trunc_normal_(pe, std=0.02)

        # ---- Transformer blocks ----
        self.blocks = nn.ModuleList([
            DiTCrossAttnBlock(
                hidden_dim=hidden_dim,
                global_cond_dim=global_cond_dim,
                cond_dim=cond_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        # ---- Output ----
        self.norm_out = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.output_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(NUM_SUBJECTS)]
        )
        for head in self.output_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _stitch_baseline(
        self,
        baseline: torch.Tensor,
        subject_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Subject-specific baseline stitching.

        Args:
            baseline:    (2B, MAX_NEURONS) zero-padded baseline activity
            subject_ids: (2B,)
        Returns:
            baseline_emb: (2B, baseline_dim)
        """
        B_all = baseline.shape[0]
        out = torch.zeros(B_all, self.baseline_dim, device=baseline.device)
        for s in range(self.num_subjects):
            mask = subject_ids == s
            if mask.any():
                n = self.neuron_counts[s]
                bl_s = baseline[mask, :n]  # (B_s, n_neurons_s)
                out[mask] = self.baseline_stitchers[s](bl_s)
        return out

    def _tokenize(
        self,
        neural_activity: torch.Tensor,
        subject_ids: torch.Tensor,
    ) -> torch.Tensor:
        B_all = neural_activity.shape[0]
        tokens = torch.zeros(
            B_all, MAX_NEURONS, self.hidden_dim,
            device=neural_activity.device, dtype=neural_activity.dtype,
        )
        x = neural_activity.unsqueeze(-1)  # (2B, MAX_NEURONS, 1)
        for s in range(self.num_subjects):
            mask = subject_ids == s
            if mask.any():
                tokens[mask] = self.input_projs[s](x[mask]) + self.pos_embeds[s]
        return tokens

    def _detokenize(
        self,
        output: torch.Tensor,
        subject_ids: torch.Tensor,
    ):
        results = {}
        for s in range(self.num_subjects):
            mask = subject_ids == s
            if mask.any():
                n = self.neuron_counts[s]
                out_s = output[mask][:, :n, :]
                pred_s = self.output_heads[s](out_s).squeeze(-1)
                results[s] = pred_s
        return results

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            raw_tokens = self.image_encoder(images)
        return self.image_proj(raw_tokens)

    def project_image_tokens(self, raw_tokens: torch.Tensor) -> torch.Tensor:
        """Project cached frozen-ViT tokens (B, S, 768) -> (B, S, cond_dim).

        The frozen ViT-MAE output is checkpoint-independent, so a caller can
        precompute it once for the whole stimulus set and hoist image encoding out
        of the denoising loop; only this projection is per-checkpoint. Feed the
        result to ``forward``/``forward_cfg`` as ``image_tokens=``.
        """
        return self.image_proj(raw_tokens)

    def forward(
        self,
        neural_activity: torch.Tensor,
        timesteps: torch.Tensor,
        images: torch.Tensor,
        baseline: torch.Tensor,
        subject_ids: torch.Tensor,
        pad_mask: torch.Tensor,
        image_tokens: torch.Tensor | None = None,
    ):
        """
        Args:
            neural_activity: (2B, MAX_NEURONS) noisy target, zero-padded
            timesteps:       (2B,) or (1,)
            images:          (2B, 3, H, W); may be None if image_tokens is given
            baseline:        (2B, MAX_NEURONS) baseline activity, zero-padded
            subject_ids:     (2B,)
            pad_mask:        (2B, MAX_NEURONS) bool, True = padded
            image_tokens:    optional precomputed (2B, S, cond_dim) projected tokens.
                             When given, ``images`` is ignored and no ViT runs.

        Returns:
            dict mapping subject_id -> predicted noise/x0 (B_s, n_neurons_s)
        """
        B_all = neural_activity.shape[0]

        # 1. Image encoding (skipped when the caller supplies projected tokens)
        if image_tokens is None:
            image_tokens = self.encode_image(images)  # (2B, S, cond_dim)
        elif self.training and self.cond_drop_prob > 0:
            # The CFG drop below assigns in place; never mutate a caller-owned cache.
            image_tokens = image_tokens.clone()

        # CFG: randomly drop image condition
        if self.training and self.cond_drop_prob > 0:
            drop_mask = torch.rand(B_all, device=image_tokens.device) < self.cond_drop_prob
            if drop_mask.any():
                null = self.null_image_token.expand(-1, image_tokens.shape[1], -1)
                image_tokens[drop_mask] = null

        global_image = image_tokens.mean(dim=1)  # (2B, cond_dim)

        # 2. Timestep
        if timesteps.ndim == 0 or timesteps.shape[0] == 1:
            timesteps = timesteps.expand(B_all)
        time_emb = self.time_embed(timesteps)  # (2B, time_dim)

        # 3. Subject embedding
        subject_emb = self.subject_embed(subject_ids)  # (2B, subject_emb_dim)

        # 4. Baseline stitching (skipped when use_baseline=False; `baseline` arg ignored)
        if self.use_baseline:
            baseline_emb = self._stitch_baseline(baseline, subject_ids)  # (2B, baseline_dim)

            # CFG: drop baseline together with image
            if self.training and self.cond_drop_prob > 0:
                if drop_mask.any():
                    baseline_emb[drop_mask] = self.null_baseline

        # 5. Global condition
        if self.use_baseline:
            global_cond = torch.cat(
                [time_emb, global_image, subject_emb, baseline_emb], dim=-1
            )
        else:
            global_cond = torch.cat([time_emb, global_image, subject_emb], dim=-1)

        # 6. Tokenize target
        tokens = self._tokenize(neural_activity, subject_ids)

        # 7. Transformer
        for block in self.blocks:
            tokens = block(tokens, global_cond, image_tokens, pad_mask)

        # 8. Output
        out = self.norm_out(tokens)
        return self._detokenize(out, subject_ids)

    def forward_cfg(
        self,
        neural_activity: torch.Tensor,
        timesteps: torch.Tensor,
        images: torch.Tensor,
        baseline: torch.Tensor,
        subject_ids: torch.Tensor,
        pad_mask: torch.Tensor,
        guidance_scale: float = 1.0,
        image_tokens: torch.Tensor | None = None,
    ):
        B_all = neural_activity.shape[0]

        # Precomputed projected tokens skip the ViT entirely (see project_image_tokens).
        if image_tokens is None:
            image_tokens = self.encode_image(images)
        global_image_cond = image_tokens.mean(dim=1)

        null_tokens = self.null_image_token.expand(B_all, image_tokens.shape[1], -1)
        global_image_uncond = null_tokens.mean(dim=1)

        if timesteps.ndim == 0 or timesteps.shape[0] == 1:
            timesteps = timesteps.expand(B_all)
        time_emb = self.time_embed(timesteps)
        subject_emb = self.subject_embed(subject_ids)

        if self.use_baseline:
            baseline_emb = self._stitch_baseline(baseline, subject_ids)
            null_baseline_emb = self.null_baseline.expand(B_all, -1)

            global_cond = torch.cat([time_emb, global_image_cond, subject_emb, baseline_emb], dim=-1)
            global_uncond = torch.cat([time_emb, global_image_uncond, subject_emb, null_baseline_emb], dim=-1)
        else:
            global_cond = torch.cat([time_emb, global_image_cond, subject_emb], dim=-1)
            global_uncond = torch.cat([time_emb, global_image_uncond, subject_emb], dim=-1)

        tokens_input = self._tokenize(neural_activity, subject_ids)

        # Conditioned
        tokens_c = tokens_input.clone()
        for block in self.blocks:
            tokens_c = block(tokens_c, global_cond, image_tokens, pad_mask)
        out_c = self.norm_out(tokens_c)
        pred_cond = self._detokenize(out_c, subject_ids)

        # Unconditioned
        tokens_u = tokens_input.clone()
        for block in self.blocks:
            tokens_u = block(tokens_u, global_uncond, null_tokens, pad_mask)
        out_u = self.norm_out(tokens_u)
        pred_uncond = self._detokenize(out_u, subject_ids)

        pred_cfg = {}
        for s in pred_cond:
            pred_cfg[s] = pred_uncond[s] + guidance_scale * (pred_cond[s] - pred_uncond[s])
        return pred_cfg

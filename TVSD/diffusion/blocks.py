from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTMAEModel

class TimestepEmbedder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    @staticmethod
    def sinusoidal_embedding(
        timesteps: torch.Tensor, dim: int, max_period: int = 10000
    ) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(0, half, device=timesteps.device).float()
            / half
        )
        args = timesteps.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        emb = self.sinusoidal_embedding(timesteps, self.dim)
        return self.mlp(emb)


class FrozenViTMAEEncoder(nn.Module):
    """
    Wraps HuggingFace ViTMAEModel to extract patch tokens.

    - Masking is disabled (mask_ratio=0) so all patches are returned.
    - Positional encoding interpolation is enabled so any input resolution works
      (e.g. 256x256 -> 256 patches on a 16x16 grid).

    Output: (B, S, embed_dim) where S = (H/patch_size) * (W/patch_size).
    """

    def __init__(self, model_id: str = "facebook/vit-mae-base"):
        super().__init__()
        self.vit = ViTMAEModel.from_pretrained(model_id)
        self.vit.config.mask_ratio = 0.0  # disable masking -> keep all patches
        self.vit.eval()
        for p in self.vit.parameters():
            p.requires_grad = False
        self.embed_dim = self.vit.config.hidden_size  # 768

        # ImageNet normalization expected by ViT-MAE.
        # The dataset produces images in [-1, 1] (Normalize(0.5, 0.5)).
        # We convert [-1,1] -> [0,1] then apply ImageNet mean/std.
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, H, W) in [-1, 1] (dataset convention)
        Returns:
            patch_tokens: (B, S, 768) -- CLS token excluded
        """
        # [-1, 1] -> [0, 1] -> ImageNet normalized
        x = pixel_values * 0.5 + 0.5  # to [0, 1]
        x = (x - self.imagenet_mean) / self.imagenet_std

        outputs = self.vit(
            pixel_values=x,
            interpolate_pos_encoding=True,
        )
        # last_hidden_state: (B, 1+S, embed_dim) where first token is CLS
        tokens = outputs.last_hidden_state[:, 1:, :]  # drop CLS -> (B, S, 768)
        return tokens


class DiTCrossAttnBlock(nn.Module):
    """
    Sub-layer 1: LN -> AdaLN-Zero(scale,shift) -> Self-Attn -> gate -> residual
    Sub-layer 2: LN -> Cross-Attn(Q=neural, KV=image) -> residual
    Sub-layer 3: LN -> AdaLN-Zero(scale,shift) -> MLP -> gate -> residual
    """

    def __init__(
        self,
        hidden_dim: int,
        global_cond_dim: int,
        cond_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # AdaLN-Zero: shift1, scale1, gate1, shift2, scale2, gate2
        self.ada = nn.Sequential(
            nn.SiLU(), nn.Linear(global_cond_dim, hidden_dim * 6)
        )
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

        # Self-attention
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Cross-attention
        self.norm_cross = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.kv_proj = nn.Linear(cond_dim, hidden_dim)

        # FFN
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        ffn_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, hidden_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        global_cond: torch.Tensor,
        image_tokens: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:            (B, N, D)  neural tokens
            global_cond:  (B, global_cond_dim)
            image_tokens: (B, S, cond_dim)  projected image patch tokens
            pad_mask:     (B, N) bool, True = padded (ignore)
        """
        # AdaLN-Zero params
        ada_out = self.ada(global_cond)  # (B, 6*D)
        shift1, scale1, gate1, shift2, scale2, gate2 = ada_out.chunk(6, dim=-1)
        # Each: (B, D) -> unsqueeze to (B, 1, D)

        # Sub-layer 1: Self-Attention
        h = self.norm1(x)
        h = h * (1.0 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        h, _ = self.self_attn(
            h, h, h, key_padding_mask=pad_mask, need_weights=False
        )
        x = x + gate1.unsqueeze(1) * h

        # Sub-layer 2: Cross-Attention to image tokens
        h = self.norm_cross(x)
        kv = self.kv_proj(image_tokens)  # (B, S, D)
        h, _ = self.cross_attn(h, kv, kv, need_weights=False)
        x = x + h

        # Sub-layer 3: FFN
        h = self.norm2(x)
        h = h * (1.0 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        h = self.mlp(h)
        x = x + gate2.unsqueeze(1) * h

        return x


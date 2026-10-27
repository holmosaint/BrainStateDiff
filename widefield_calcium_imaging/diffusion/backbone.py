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
def exists(x) -> bool:
    return x is not None

def default(val, d):
    return val if exists(val) else (d() if callable(d) else d)

def prob_mask_like(shape, prob: float, device):
    if prob <= 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    if prob >= 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    return (torch.rand(shape, device=device) < prob)

def is_odd(n: int) -> bool:
    return (n % 2) == 1

class RelativePositionBias(nn.Module):
    def __init__(self, heads: int = 8, num_buckets: int = 32, max_distance: int = 128):
        super().__init__()
        self.num_buckets = int(num_buckets)
        self.max_distance = int(max_distance)
        self.relative_attention_bias = nn.Embedding(self.num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets: int = 32, max_distance: int = 128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n: int, device: torch.device):
        q_pos = torch.arange(n, dtype=torch.long, device=device)
        k_pos = torch.arange(n, dtype=torch.long, device=device)
        rel_pos = k_pos[None, :] - q_pos[:, None]  # (n,n)
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)  # (n,n,heads)
        return values.permute(2, 0, 1).contiguous()       # (heads,n,n)

class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn
    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None].float() * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

def Upsample(dim: int) -> nn.Module:
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

def Downsample(dim: int) -> nn.Module:
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma

class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1, 1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1) * self.scale * self.gamma

class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.norm(x)
        return self.fn(x, **kwargs)

class Block(nn.Module):
    def __init__(self, dim: int, dim_out: int):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding=(0, 1, 1))
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
    def forward(self, x: torch.Tensor, scale_shift: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)

class ResnetBlock(nn.Module):
    """
    Supports time_emb:
      - (B, D)      : global
      - (B, F, D)   : per-frame
    """
    def __init__(self, dim: int, dim_out: int, *, time_emb_dim: Optional[int] = None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), "time_emb must be passed"
            te = self.mlp(time_emb)  # (B,2*dim_out) or (B,F,2*dim_out)

            if te.dim() == 2:
                te = te.view(te.shape[0], te.shape[1], 1, 1, 1)
                scale_shift = te.chunk(2, dim=1)  # each (B,dim_out,1,1,1)
            elif te.dim() == 3:
                scale, shift = te.chunk(2, dim=-1)          # (B,F,dim_out)
                scale = scale.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # (B,dim_out,F,1,1)
                shift = shift.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
                scale_shift = (scale, shift)
            else:
                raise ValueError(f"Unexpected time embedding shape: {tuple(te.shape)}")

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

class SpatialLinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = int(heads)
        hidden_dim = dim_head * self.heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, f, h, w = x.shape
        x2 = x.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)  # (B*F,C,H,W)

        q, k, v = self.to_qkv(x2).chunk(3, dim=1)
        hidden = q.shape[1]
        dim_head = hidden // self.heads

        q = q.view(b * f, self.heads, dim_head, h * w)
        k = k.view(b * f, self.heads, dim_head, h * w)
        v = v.view(b * f, self.heads, dim_head, h * w)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = out.reshape(b * f, self.heads * dim_head, h, w)
        out = self.to_out(out)

        out = out.view(b, f, c, h, w).permute(0, 2, 1, 3, 4)
        return out

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def get_sin_cos(self, n: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(n, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("n,d->nd", t, self.inv_freq)  # (n, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)            # (n, dim)
        sin = emb.sin().to(dtype=dtype)
        cos = emb.cos().to(dtype=dtype)
        return sin, cos

    def rotate_queries_or_keys(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[-2]
        sin, cos = self.get_sin_cos(n, x.device, x.dtype)
        while sin.dim() < x.dim():
            sin = sin.unsqueeze(0)
            cos = cos.unsqueeze(0)
        return (x * cos) + (rotate_half(x) * sin)

class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32, rotary_emb: Optional[RotaryEmbedding] = None):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        hidden_dim = self.heads * self.dim_head
        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,                          # (..., n, dim)
        pos_bias: Optional[torch.Tensor] = None,  # (heads,n,n)
        focus_present_mask: Optional[torch.Tensor] = None,  # (B,)
    ) -> torch.Tensor:
        n = x.shape[-2]
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        if exists(focus_present_mask) and bool(focus_present_mask.all()):
            values = qkv[-1]
            return self.to_out(values)

        q, k, v = qkv
        *lead, _n, _ = q.shape

        q = q.view(*lead, n, self.heads, self.dim_head).permute(*range(len(lead)), -2, -3, -1)
        k = k.view(*lead, n, self.heads, self.dim_head).permute(*range(len(lead)), -2, -3, -1)
        v = v.view(*lead, n, self.heads, self.dim_head).permute(*range(len(lead)), -2, -3, -1)

        q = q * self.scale

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        sim = torch.einsum("... h i d, ... h j d -> ... h i j", q, k)
        if exists(pos_bias):
            sim = sim + pos_bias

        if exists(focus_present_mask) and not bool((~focus_present_mask).all()):
            b = focus_present_mask.shape[0]
            device = x.device
            attend_all = torch.ones((n, n), device=device, dtype=torch.bool)
            attend_self = torch.eye(n, device=device, dtype=torch.bool)
            fp = focus_present_mask.view(b, *([1] * (sim.dim() - 1)))
            base_shape = (1,) * (sim.dim() - 2) + (n, n)
            mask = torch.where(fp, attend_self.view(base_shape), attend_all.view(base_shape))
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("... h i j, ... h j d -> ... h i d", attn, v)
        out = out.permute(*range(out.dim() - 3), -2, -3, -1).contiguous()
        out = out.view(*lead, n, self.heads * self.dim_head)
        return self.to_out(out)

class TemporalAttention3D(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, rotary_emb: RotaryEmbedding):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, rotary_emb=rotary_emb)

    def forward(self, x: torch.Tensor, pos_bias: Optional[torch.Tensor] = None, focus_present_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, c, f, h, w = x.shape
        x_ = x.permute(0, 3, 4, 2, 1).reshape(b, h * w, f, c)  # (B,HW,F,C)
        out = self.attn(x_, pos_bias=pos_bias, focus_present_mask=focus_present_mask)
        out = out.reshape(b, h, w, f, c).permute(0, 4, 3, 1, 2).contiguous()
        return out

class SpatialFullAttention3D(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, rotary_emb=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, f, h, w = x.shape
        x_ = x.permute(0, 2, 3, 4, 1).reshape(b, f, h * w, c)  # (B,F,HW,C)
        out = self.attn(x_)
        out = out.reshape(b, f, h, w, c).permute(0, 4, 1, 2, 3).contiguous()
        return out

class Unet3D_VideoDiffusionPytorch(nn.Module):
    """
    Like reference Unet3D, but supports per-frame condition:
      cond: (B, F, cond_dim)
    """
    def __init__(
        self,
        dim: int,
        cond_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        dim_mults: Sequence[int] = (1, 2, 4, 8),
        channels: int = 3,
        attn_heads: int = 8,
        attn_dim_head: int = 32,
        init_dim: Optional[int] = None,
        init_kernel_size: int = 7,
        use_sparse_linear_attn: bool = True,
    ):
        super().__init__()
        self.channels = int(channels)

        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))
        self.time_rel_pos_bias = RelativePositionBias(heads=attn_heads, max_distance=32)

        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)
        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(self.channels, init_dim, (1, init_kernel_size, init_kernel_size), padding=(0, init_padding, init_padding))

        self.init_temporal_attn = Residual(PreNorm(init_dim, TemporalAttention3D(init_dim, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb)))

        dims = [init_dim, *[dim * int(m) for m in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.has_cond = exists(cond_dim)
        self.cond_dim = int(cond_dim) if exists(cond_dim) else 0
        self.null_cond_emb = nn.Parameter(torch.randn(1, self.cond_dim)) if self.has_cond else None

        self.cond_total_dim = time_dim + (self.cond_dim if self.has_cond else 0)

        block_klass_cond = lambda di, do: ResnetBlock(di, do, time_emb_dim=self.cond_total_dim)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                block_klass_cond(dim_out, dim_out),
                Residual(PreNorm(dim_out, SpatialLinearAttention(dim_out, heads=attn_heads, dim_head=attn_dim_head))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_out, TemporalAttention3D(dim_out, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb))),
                Downsample(dim_out) if not is_last else nn.Identity(),
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)
        self.mid_spatial_attn = Residual(PreNorm(mid_dim, SpatialFullAttention3D(mid_dim, heads=attn_heads, dim_head=attn_dim_head)))
        self.mid_temporal_attn = Residual(PreNorm(mid_dim, TemporalAttention3D(mid_dim, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb)))
        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_in),
                block_klass_cond(dim_in, dim_in),
                Residual(PreNorm(dim_in, SpatialLinearAttention(dim_in, heads=attn_heads, dim_head=attn_dim_head))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_in, TemporalAttention3D(dim_in, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb))),
                Upsample(dim_in) if not is_last else nn.Identity(),
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            ResnetBlock(dim * 2, dim, time_emb_dim=None),
            nn.Conv3d(dim, out_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,                 # (B,C,F,H,W)
        time: torch.Tensor,              # (B,)
        cond: Optional[torch.Tensor] = None,      # (B,F,cond_dim)
        null_cond_prob: float = 0.0,
        focus_present_mask: Optional[torch.Tensor] = None, # (B,)
        prob_focus_present: float = 0.0,
    ) -> torch.Tensor:
        assert not (self.has_cond and not exists(cond)), "cond must be passed if cond_dim specified"

        b, device = x.shape[0], x.device
        frames = x.shape[2]

        focus_present_mask = default(focus_present_mask, lambda: prob_mask_like((b,), prob_focus_present, device=device))
        time_rel_pos_bias = self.time_rel_pos_bias(frames, device=x.device)  # (heads,F,F)

        x = self.init_conv(x)
        x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)

        r = x.clone()

        t = self.time_mlp(time)  # (B,time_dim)
        t = t[:, None, :].expand(b, frames, t.shape[-1])  # (B,F,time_dim)

        if self.has_cond:
            if cond.dim() != 3 or cond.shape[0] != b or cond.shape[1] != frames or cond.shape[2] != self.cond_dim:
                raise ValueError(f"Expected cond shape (B,F,cond_dim)=({b},{frames},{self.cond_dim}), got {tuple(cond.shape)}")

            mask = prob_mask_like((b,), null_cond_prob, device=device)  # True => all-null
            null_emb = self.null_cond_emb.to(device).view(1, 1, self.cond_dim)  # (1,1,cond_dim)
            cond = torch.where(mask[:, None, None], null_emb, cond)
            t = torch.cat((t, cond), dim=-1)

        h = []
        for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
        x = self.mid_block2(x, t)

        for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        return self.final_conv(x)

class ImageConditioner(nn.Module):
    def __init__(self, cond_dim: int, image_emb_dim: int, tokens_per_emb: int):
        super().__init__()
        self.tokens_per_emb = int(tokens_per_emb)
        self.cond_dim_per_token = int(cond_dim)
        self.cond_dim_total = int(cond_dim) * int(tokens_per_emb)
        self.proj = nn.Linear(int(image_emb_dim), int(cond_dim))
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(int(cond_dim), int(cond_dim)))

    def forward(self, image_emb: torch.Tensor) -> torch.Tensor:
        if image_emb.dim() != 3:
            raise ValueError(f"Expected image_emb shape (B,tokens,dim), got {tuple(image_emb.shape)}")
        if image_emb.shape[1] != self.tokens_per_emb or image_emb.shape[2] != self.proj.in_features:
            raise ValueError(
                f"Expected image_emb shape (B,{self.tokens_per_emb},{self.proj.in_features}), got {tuple(image_emb.shape)}"
            )

        per_token = self.mlp(self.proj(image_emb))               # (B,P,cond_dim_per_token)
        return per_token.reshape(image_emb.shape[0], self.cond_dim_total)

class SlidingWindowModel(nn.Module):
    """
    cond_frames[b,t] = image_cond[b] if cond_mask[b,t]==1 else null_cond_emb
    then Unet3D runs with per-frame cond.
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
    ):
        super().__init__()
        self.cond_encoder = ImageConditioner(cond_dim=cond_dim, image_emb_dim=image_emb_dim, tokens_per_emb=tokens_per_emb)
        self.cond_dim_total = self.cond_encoder.cond_dim_total
        self.unet = Unet3D_VideoDiffusionPytorch(
            dim=unet_dim,
            cond_dim=self.cond_dim_total,
            out_dim=latent_channels,
            dim_mults=tuple(int(x) for x in dim_mults),
            channels=latent_channels,
            attn_heads=attn_heads,
            attn_dim_head=attn_dim_head,
            init_dim=init_dim,
            init_kernel_size=init_kernel_size,
            use_sparse_linear_attn=use_sparse_linear_attn,
        )

    def forward(
        self,
        volume: torch.Tensor,        # (B,C,T,H,W)
        timesteps: torch.Tensor,     # (B,)
        image_emb: torch.Tensor,     # (B,P,E)
        cond_mask: torch.Tensor,     # (B,T) float/bool
        null_cond_prob: float = 0.0,
        prob_focus_present: float = 0.0,
    ) -> torch.Tensor:
        B, _, T, _, _ = volume.shape
        if cond_mask.dim() != 2 or cond_mask.shape[0] != B or cond_mask.shape[1] != T:
            raise ValueError(f"Expected cond_mask shape (B,T)=({B},{T}), got {tuple(cond_mask.shape)}")

        img_cond = self.cond_encoder(image_emb)  # (B,cond_dim_total)
        cond_frames = img_cond[:, None, :].expand(B, T, self.cond_dim_total)  # (B,T,cond_dim_total)

        null_emb = self.unet.null_cond_emb.to(volume.device).view(1, 1, self.cond_dim_total).expand(B, T, self.cond_dim_total)
        m = (cond_mask > 0.5)[:, :, None]  # (B,T,1)
        cond_frames = torch.where(m, cond_frames, null_emb)

        return self.unet(
            volume,
            timesteps,
            cond=cond_frames,
            null_cond_prob=null_cond_prob,
            prob_focus_present=prob_focus_present,
        )

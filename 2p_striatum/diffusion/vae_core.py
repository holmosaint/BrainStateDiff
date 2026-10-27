from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class AutoencoderKLConfig:
    """Configuration shared by all AutoencoderKL wrappers."""

    ckpt_path: Optional[str] = None  # Path to load model checkpoint from
    loss_type: str = "gll"  # "gll", "RADICaL", or "mse"
    kl_weight: float = 0.0001  # Weight for KL divergence loss
    dropout: float = 0.2  # Dropout rate for regularization
    base_channels: int = 64
    ch_mult: Tuple[int, ...] = (1, 2)
    num_groups: int = 32  # For GroupNorm
    kernel_size: int = 3  # Global kernel size
    padding: Union[str, int] = 'auto'  # 'auto' or int
    stride: int = 2  # Stride for convolution layers
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
     # RADICaL (ZIG) params
    smin: float = 0.1          # gamma loc = minimum nonzero event 
    radical_eps: float = 1e-8 # numeric stability


class ResnetBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, groups=32, dropout=0.2, kernel_size=3, padding=1):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=1e-6)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=1e-6)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.shortcut(x)


class Downsample1D(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.conv(x)


class Upsample1D(nn.Module):
    def __init__(self, channels, scale_factor=2.0, kernel_size=3, padding=1):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        return self.conv(x)


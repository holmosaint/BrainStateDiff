"""Frozen stitched VAE and the original frame-independent encoding convention."""
from typing import Tuple, Optional
from dataclasses import asdict
import pickle
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from .vae_core import AutoencoderKLConfig
from .stitched_vae import StitchedAutoencoderKL_1D
AE_INTERP_MODE = "linear"
AE_INTERP_ALIGN_CORNERS = False


class _CheckpointUnpickler(pickle.Unpickler):
    """Map the one config class stored in the original stitched checkpoint."""

    def find_class(self, module, name):
        if (module, name) == ("pretraining.models.VAE.autoencoder_kl_motor", "AutoencoderKLConfig"):
            return AutoencoderKLConfig
        return super().find_class(module, name)


_CHECKPOINT_PICKLE = SimpleNamespace(
    __name__="stitched_vae_checkpoint", Unpickler=_CheckpointUnpickler, load=pickle.load)

class SessionBoundVAE(nn.Module):
    """Thin wrapper around StitchedAutoencoderKL_1D that binds a session_id.

    Exposes ``.encode(x)`` and ``.decode(z)`` with the same signatures as
    ``AutoencoderKL_1D`` so that ``encode_frames_independent`` and
    ``decode_frames_independent`` work without modification.
    """

    def __init__(self, stitched_vae: StitchedAutoencoderKL_1D, session_id: str):
        super().__init__()
        self.vae = stitched_vae
        self.session_id = session_id

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.vae.encode(x, self.session_id)

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.vae.decode(z, self.session_id)

    def forward(self, x: torch.Tensor, sample_posterior: bool = True):
        return self.vae(x, session_id=self.session_id, sample_posterior=sample_posterior)


@torch.no_grad()
def encode_frames_independent(
    ae: SessionBoundVAE,
    x: torch.Tensor,
    sample_posterior: bool = False,
) -> torch.Tensor:
    """Encode each timepoint independently.  x: (B, T, N) -> z: (B, T, D)."""
    B, T, N = x.shape
    x1 = x.reshape(B * T, 1, N).permute(0, 2, 1).contiguous()  # (B*T, N, 1)
    mean, logvar = ae.encode(x1)  # (B*T, D, L)
    if mean.shape[-1] != 1:
        mean = F.interpolate(mean, size=1, mode=AE_INTERP_MODE, align_corners=AE_INTERP_ALIGN_CORNERS)
        logvar = F.interpolate(logvar, size=1, mode=AE_INTERP_MODE, align_corners=AE_INTERP_ALIGN_CORNERS)
    if sample_posterior:
        std = torch.exp(0.5 * logvar)
        z = mean + torch.randn_like(std) * std
    else:
        z = mean
    return z.squeeze(-1).reshape(B, T, -1)


@torch.no_grad()
def decode_frames_independent(ae: SessionBoundVAE, z: torch.Tensor) -> torch.Tensor:
    """Decode each timepoint independently.  z: (B, T, D) -> recon_mu: (B, T, N)."""
    B, T, D = z.shape
    z1 = z.reshape(B * T, D, 1)
    recon_mu, _ = ae.decode(z1)
    if recon_mu.shape[-1] != 1:
        recon_mu = F.interpolate(recon_mu, size=1, mode=AE_INTERP_MODE, align_corners=AE_INTERP_ALIGN_CORNERS)
    return recon_mu.squeeze(-1).reshape(B, T, -1)


def load_frozen_stitched_vae(checkpoint, device="cpu"):
    """Read canonical checkpoint metadata and require an exact state_dict match.

    The checkpoint contains config (dict or original config dataclass), session_neuron_counts, and
    model_state_dict; latent width is determined by quant_conv.weight.
    """
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False,
                      pickle_module=_CHECKPOINT_PICKLE)
    config = (asdict(ckpt["config"]) if isinstance(ckpt["config"], AutoencoderKLConfig)
              else dict(ckpt["config"]))
    config["device"] = str(device)
    config["ch_mult"] = tuple(config["ch_mult"])
    state = ckpt["model_state_dict"]
    latent_dim = state["quant_conv.weight"].shape[0] // 2
    counts = ckpt["session_neuron_counts"]
    model = StitchedAutoencoderKL_1D(
        counts, latent_dim, AutoencoderKLConfig(**config),
        use_session_embedding="session_embedding.weight" in state,
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    return model, latent_dim, counts

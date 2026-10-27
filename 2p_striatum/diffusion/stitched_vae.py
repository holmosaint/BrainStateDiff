from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vae_core import (
    AutoencoderKLConfig,
    ResnetBlock1D,
    Downsample1D,
    Upsample1D,
)


# ==========================================
# Stitched Multi-Session VAE
# ==========================================
class StitchedAutoencoderKL_1D(nn.Module):
    """VAE with session-specific I/O layers and a shared backbone.

    Each session (identified by a string name) has its own ``encoder_in``
    and ``final_conv`` layers whose channel dimensions match the session's
    neuron count.  The encoder body, bottleneck, decoder body, and final
    normalization are shared across all sessions so the model learns a
    universal latent space.
    """

    def __init__(
        self,
        session_neuron_counts: Dict[str, int],
        latent_dim: int,
        config: AutoencoderKLConfig,
        use_session_embedding: bool = False,
    ):
        super().__init__()
        self.config = config
        self.device = config.device
        self.latent_dim = latent_dim
        self.loss_type = config.loss_type.lower()
        self.session_neuron_counts = dict(session_neuron_counts)
        self.session_names: List[str] = sorted(session_neuron_counts.keys())

        base_ch = config.base_channels
        ch_mult = config.ch_mult
        groups = config.num_groups
        drop = config.dropout
        k_size = config.kernel_size
        stride = config.stride

        if config.padding == 'auto':
            pad = k_size // 2
        else:
            pad = int(config.padding)

        # ---- SESSION-SPECIFIC I/O LAYERS ----
        self.encoder_ins = nn.ModuleDict()
        self.final_convs = nn.ModuleDict()

        for sid, n_neurons in session_neuron_counts.items():
            self.encoder_ins[sid] = nn.Conv1d(
                n_neurons, base_ch, kernel_size=k_size, padding=pad,
            )

            if self.loss_type == 'gll':
                final_out_ch = n_neurons * 2
            elif self.loss_type == 'radical':
                final_out_ch = n_neurons * 3
            else:
                final_out_ch = n_neurons

            # final_conv output channels are computed after we build decoder_body
            # so we store them now but create the layers after computing curr_ch
            self._final_out_chs = getattr(self, '_final_out_chs', {})
            self._final_out_chs[sid] = final_out_ch

        # ---- SHARED ENCODER BODY ----
        enc_blocks = []
        curr_ch = base_ch

        for i, mult in enumerate(ch_mult):
            out_ch = base_ch * mult
            enc_blocks.append(ResnetBlock1D(curr_ch, out_ch, groups, drop, kernel_size=k_size, padding=pad))
            enc_blocks.append(ResnetBlock1D(out_ch, out_ch, groups, drop, kernel_size=k_size, padding=pad))
            if stride > 1:
                enc_blocks.append(Downsample1D(out_ch, kernel_size=k_size, stride=stride, padding=pad))
            curr_ch = out_ch

        self.encoder_body = nn.Sequential(*enc_blocks)

        # ---- SHARED BOTTLENECK ----
        self.quant_conv = nn.Conv1d(curr_ch, 2 * latent_dim, kernel_size=1)
        self.post_quant_conv = nn.Conv1d(latent_dim, curr_ch, kernel_size=1)

        # ---- SHARED DECODER BODY ----
        dec_blocks = []
        for i, mult in enumerate(reversed(ch_mult)):
            out_ch = base_ch * mult
            if stride > 1:
                dec_blocks.append(Upsample1D(curr_ch, scale_factor=float(stride), kernel_size=k_size, padding=pad))
            dec_blocks.append(ResnetBlock1D(curr_ch, out_ch, groups, drop, kernel_size=k_size, padding=pad))
            dec_blocks.append(ResnetBlock1D(out_ch, out_ch, groups, drop, kernel_size=k_size, padding=pad))
            curr_ch = out_ch

        self.decoder_body = nn.Sequential(*dec_blocks)
        self.final_norm = nn.GroupNorm(num_groups=groups, num_channels=curr_ch, eps=1e-6)
        self.final_act = nn.SiLU()

        # Now create session-specific final_conv layers (need curr_ch from decoder)
        for sid in session_neuron_counts:
            self.final_convs[sid] = nn.Conv1d(
                curr_ch, self._final_out_chs[sid], kernel_size=k_size, padding=pad,
            )
        del self._final_out_chs

        # ---- OPTIONAL SESSION EMBEDDING ----
        self.session_embedding: Optional[nn.Embedding] = None
        if use_session_embedding:
            self.session_embedding = nn.Embedding(len(self.session_names), latent_dim)
            nn.init.normal_(self.session_embedding.weight, std=0.02)

    # ------------------------------------------------------------------
    # Core forward methods
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor, session_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode neural data to latent distribution parameters.

        Args:
            x: (B, N_session, T) neural data in channel-first format.
            session_id: which session's encoder_in to use.
        Returns:
            mean, logvar each (B, latent_dim, T').
        """
        h = self.encoder_ins[session_id](x)
        h = self.encoder_body(h)
        moments = self.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        return mean, logvar

    def decode(self, z: torch.Tensor, session_id: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Decode latent samples back to neural space.

        Args:
            z: (B, latent_dim, T') latent samples.
            session_id: which session's final_conv to use.
        Returns:
            recon_mu (B, N, T'), recon_logvar (B, k*N, T') or None.
        """
        h = self.post_quant_conv(z)
        h = self.decoder_body(h)
        h = self.final_act(self.final_norm(h))
        h = self.final_convs[session_id](h)

        if self.loss_type == 'radical':
            q_raw, k_raw, alpha_raw = torch.chunk(h, 3, dim=1)
            eps = self.config.radical_eps
            q = torch.sigmoid(q_raw).clamp(eps, 1.0 - eps)
            k_param = F.softplus(k_raw) + eps
            alpha = F.softplus(alpha_raw) + eps
            smin = torch.tensor(self.config.smin, device=h.device, dtype=h.dtype).view(1, 1, 1)
            recon_mu = q * (k_param * alpha + smin)
            recon_logvar = torch.cat([q, k_param, alpha], dim=1)
        elif self.loss_type == 'gll':
            recon_mu, recon_logvar = torch.chunk(h, 2, dim=1)
        else:
            recon_mu = h
            recon_logvar = None

        return recon_mu, recon_logvar

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(
        self,
        x: torch.Tensor,
        session_id: str,
        sample_posterior: bool = True,
    ):
        """Full forward pass for a single session's batch.

        Args:
            x: (B, T, N) neural data (time-major, as stored in dataloaders).
            session_id: selects per-session I/O layers.
            sample_posterior: if True reparameterize; else use mean.
        Returns:
            recon_mu (B, T, N), recon_logvar (B, T, k*N) or None,
            z_mean (B, D, T'), z_logvar (B, D, T').
        """
        if self.loss_type == 'radical':
            x = torch.nan_to_num(x, nan=0.0)

        target_len = x.shape[1]

        x = x.permute(0, 2, 1)  # (B, N, T)
        mean, logvar = self.encode(x, session_id)

        if sample_posterior:
            z = self.reparameterize(mean, logvar)
        else:
            z = mean

        # Optional session embedding injection (additive in latent space)
        if self.session_embedding is not None:
            sid_idx = self.session_names.index(session_id)
            emb = self.session_embedding(
                torch.tensor(sid_idx, device=z.device)
            )  # (latent_dim,)
            z = z + emb.unsqueeze(0).unsqueeze(-1)  # broadcast (1, D, 1)

        recon_mu, recon_logvar = self.decode(z, session_id)

        # Fix temporal shape mismatch from strided down/up-sampling
        if recon_mu.shape[-1] != target_len:
            recon_mu = F.interpolate(recon_mu, size=target_len, mode='linear', align_corners=False)
            if recon_logvar is not None:
                recon_logvar = F.interpolate(recon_logvar, size=target_len, mode='linear', align_corners=False)

        recon_mu = recon_mu.permute(0, 2, 1)  # (B, T, N)
        if recon_logvar is not None:
            recon_logvar = recon_logvar.permute(0, 2, 1)

        return recon_mu, recon_logvar, mean, logvar

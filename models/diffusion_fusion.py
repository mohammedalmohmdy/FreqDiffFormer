"""Diffusion latent fusion — TRAINING-TIME regularizer only.

Manuscript: diffusion is a training-time regularizer. At retrieval inference NO
iterative reverse-diffusion sampling is performed; the model uses the
deterministic fused latent from the Cross-Domain Transformer directly.

This module therefore exposes two clearly separated methods:
  - training_loss(z, cond): returns the diffusion reconstruction loss.
  - inference_latent(z):  returns z unchanged (no reverse sampling).
reverse() is intentionally NOT provided here.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .diffusion.ddpm import LatentDDPM


class DiffusionRegularizer(nn.Module):
    def __init__(
        self,
        latent_dim: int = 512,
        timesteps: int = 100,
        beta_start: float = 1.0e-4,
        beta_end: float = 2.0e-2,
        device: str = "cpu",
        unet_kwargs: dict | None = None,
    ):
        super().__init__()
        self.ddpm = LatentDDPM(latent_dim=latent_dim, timesteps=timesteps,
                               beta_start=beta_start, beta_end=beta_end,
                               device=device, unet_kwargs=unet_kwargs)

    def training_loss(self, z: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Diffusion reconstruction loss on the fused latent z."""
        return self.ddpm.forward_loss(z, cond=cond)

    @torch.no_grad()
    def inference_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Deterministic inference latent — NO reverse diffusion sampling."""
        return z


# Backward-compatible alias.
DiffusionLatentFusion = DiffusionRegularizer

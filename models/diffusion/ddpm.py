"""Latent DDPM used as a TRAINING-TIME regularizer.

Manuscript:
  - T = 100, linear beta (1e-4 .. 2e-2)
  - timesteps sampled uniformly during training
  - objective: MSE between the injected Gaussian noise and the predicted noise
  - diffusion is a training-time regularizer; NO iterative reverse diffusion at
    retrieval inference.

The reverse `sample()` method exists for research/diagnostic purposes only and is
NOT invoked anywhere on the retrieval/inference path (see scripts/eval.py and
models/diffusion_fusion.py). The retrieval forward uses the deterministic fused
latent directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .scheduler import LinearBetaSchedule, DEFAULT_TIMESTEPS, DEFAULT_BETA_START, DEFAULT_BETA_END
from .unet import ConditionalLatentUNet


class LatentDDPM(nn.Module):
    def __init__(
        self,
        latent_dim: int = 512,
        timesteps: int = DEFAULT_TIMESTEPS,
        beta_start: float = DEFAULT_BETA_START,
        beta_end: float = DEFAULT_BETA_END,
        device: str = "cpu",
        unet_kwargs: dict | None = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.timesteps = int(timesteps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.scheduler = LinearBetaSchedule(timesteps=timesteps,
                                           beta_start=beta_start,
                                           beta_end=beta_end,
                                           device=device)
        u = unet_kwargs or {}
        self.model = ConditionalLatentUNet(latent_dim=latent_dim, **u)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = args[0] if args else kwargs.get("device", "cpu")
        self.scheduler.to(device)
        return self

    def forward_loss(self, x0: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Diffusion reconstruction (denoising) loss.

        Args:
            x0:   (B, latent_dim) fused latent to regularize.
            cond: (B, M, cond_dim) conditioning tokens (cross-domain fused tokens).
                  If None, a single learnable-vector conditioning is emitted.
        Returns:
            scalar MSE(predicted_noise, injected_noise).
        """
        B = x0.size(0)
        t = torch.randint(0, self.timesteps, (B,), device=x0.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        xt = self.scheduler.q_sample(x0, t, noise)
        if cond is None:
            cond = x0.unsqueeze(1)  # (B, 1, latent_dim) degenerate conditioning
        pred_noise = self.model(xt, t, cond)
        loss = F_mse(pred_noise, noise)
        return loss

    @torch.no_grad()
    def sample(self, shape, cond=None, device=None):
        """RESEARCH-ONLY reverse diffusion sampling.

        NOT used for retrieval inference. Provided for diagnostics/inspection.
        """
        device = device if device is not None else self.scheduler.device
        x = torch.randn(shape, device=device)
        if cond is None:
            cond = x.unsqueeze(1)
        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            pred_noise = self.model(x, t_tensor, cond)
            beta_t = self.scheduler.beta[t].to(device)
            alpha_t = self.scheduler.alpha[t].to(device)
            alpha_bar_t = self.scheduler.alpha_bar[t].to(device)
            noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
            x = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise)
            x = x + torch.sqrt(beta_t) * noise
        return x


def F_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(pred, target)

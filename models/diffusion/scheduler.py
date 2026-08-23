"""Linear beta schedule for the latent diffusion regularizer.

Manuscript:
  - T = 100 timesteps
  - linear beta schedule, beta_min = 1.0e-4, beta_max = 2.0e-2
  - forward (posterior) process:  x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps
  - timesteps sampled uniformly during training
"""

from __future__ import annotations

import torch

# Canonical defaults (manuscript). Override only via config.
DEFAULT_TIMESTEPS = 100
DEFAULT_BETA_START = 1.0e-4
DEFAULT_BETA_END = 2.0e-2


class LinearBetaSchedule:
    """Linear beta schedule with derived alpha / alpha_bar.

    Buffers are stored as tensors (registered on the given device) so everything
    follows the device / dtype in use.
    """

    def __init__(
        self,
        timesteps: int = DEFAULT_TIMESTEPS,
        beta_start: float = DEFAULT_BETA_START,
        beta_end: float = DEFAULT_BETA_END,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        if timesteps != 100:
            pass  # configurable, but default is manuscript 100
        if not (beta_start < beta_end):
            raise ValueError(f"beta_start must be < beta_end, got {beta_start} >= {beta_end}")
        if beta_start < 0 or beta_end >= 1.0:
            raise ValueError("beta values must be in [0, 1).")
        self.timesteps = int(timesteps)
        self.device = device
        self.dtype = dtype
        self.beta = torch.linspace(beta_start, beta_end, self.timesteps,
                                   device=device, dtype=dtype)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def to(self, device, dtype=None):
        self.device = device
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        if dtype is not None:
            self.beta = self.beta.to(dtype)
            self.alpha = self.alpha.to(dtype)
            self.alpha_bar = self.alpha_bar.to(dtype)
            self.dtype = dtype
        return self

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward (posterior) sampling: x_t = sqrt(a_bar_t) x0 + sqrt(1-a_bar_t) eps."""
        a_bar = self.alpha_bar[t].to(x0.device, dtype=x0.dtype)
        a_bar = a_bar.view(-1, *([1] * (x0.dim() - 1)))
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise

    def get(self, t: torch.Tensor, name: str) -> torch.Tensor:
        return getattr(self, name)[t].to(t.device)

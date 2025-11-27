
import torch
import torch.nn as nn
from .scheduler import LinearBetaSchedule
from .unet import LatentUNet
import random

class LatentDDPM(nn.Module):
    def __init__(self, latent_dim=512, timesteps=50, device='cpu'):
        super().__init__()
        self.timesteps = timesteps
        self.device = device
        self.model = LatentUNet(latent_dim=latent_dim)
        self.scheduler = LinearBetaSchedule(timesteps=timesteps, device=device)

    def forward_loss(self, x0):
        """
        x0: (B, D)
        returns: predicted noise loss scalar
        """
        B = x0.size(0)
        t = torch.randint(0, self.timesteps, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.scheduler.q_sample(x0, t, noise)
        pred_noise = self.model(xt, t)
        loss = ((pred_noise - noise)**2).mean()
        return loss

    @torch.no_grad()
    def sample(self, shape, device=None):
        """
        Reverse sampling from pure noise to x0_hat.
        shape: (B, D)
        """
        device = device if device is not None else self.device
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            pred_noise = self.model(x, t_tensor)
            beta_t = self.scheduler.beta[t]
            alpha_t = self.scheduler.alpha[t]
            alpha_bar_t = self.scheduler.alpha_bar[t]
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            coef1 = 1.0 / torch.sqrt(alpha_t)
            coef2 = (beta_t / torch.sqrt(1.0 - alpha_bar_t))
            x = coef1 * (x - coef2 * pred_noise) + torch.sqrt(beta_t) * noise
        return x

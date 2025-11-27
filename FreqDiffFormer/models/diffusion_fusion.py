
import torch
import torch.nn as nn
from .diffusion.ddpm import LatentDDPM

class DiffusionLatentFusion(nn.Module):
    def __init__(self, latent_dim=512, timesteps=50, device='cpu'):
        super().__init__()
        self.ddpm = LatentDDPM(latent_dim=latent_dim, timesteps=timesteps, device=device)

    def forward(self, z):
        """
        For training: z is (B, D) latent to be denoised/regularized.
        Forward returns a denoised/refined latent (identity here) and/or uses ddpm during training.
        """
        # This wrapper uses the ddpm model for loss computation externally.
        return z

    def ddpm_loss(self, z):
        return self.ddpm.forward_loss(z)

    @torch.no_grad()
    def refine(self, shape, device=None):
        return self.ddpm.sample(shape, device=device)

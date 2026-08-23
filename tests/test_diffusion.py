"""Tests for the diffusion scheduler, conditional latent U-Net, and regularizer."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from models.diffusion.scheduler import LinearBetaSchedule, DEFAULT_TIMESTEPS, DEFAULT_BETA_START, DEFAULT_BETA_END
from models.diffusion.unet import ConditionalLatentUNet
from models.diffusion.ddpm import LatentDDPM
from models.diffusion_fusion import DiffusionRegularizer


def test_scheduler_defaults_match_manuscript():
    s = LinearBetaSchedule()
    assert s.timesteps == 100 == DEFAULT_TIMESTEPS
    assert abs(s.beta[0].item() - DEFAULT_BETA_START) < 1e-6
    assert abs(s.beta[-1].item() - DEFAULT_BETA_END) < 1e-6
    assert s.alpha_bar.shape[0] == 100


def test_scheduler_monotonic_alpha_bar():
    s = LinearBetaSchedule(100, 1e-4, 2e-2)
    assert torch.all(s.alpha_bar[1:] <= s.alpha_bar[:-1] + 1e-8)  # non-increasing


def test_unet_has_self_and_cross_attention():
    unet_kw = dict(base_dim=64, latent_grid=8, cond_dim=256, num_heads=4,
                   self_attn=True, cross_attn=True)
    u = ConditionalLatentUNet(latent_dim=512, **unet_kw)
    assert u.self_attn is not None
    assert all(blk.use_cross_attn for blk in [u.enc0, u.enc1, u.bot, u.dec0, u.dec1, u.dec2])
    out = u(torch.randn(2, 512), torch.randint(0, 100, (2,)), torch.randn(2, 10, 256))
    assert out.shape == (2, 512)


def test_unet_stride_and_back_identity():
    """U-Net preserves latent shape: (B, latent_dim) in -> (B, latent_dim) out."""
    u = ConditionalLatentUNet(latent_dim=512)
    out = u(torch.randn(4, 512), torch.randint(0, 100, (4,)), torch.randn(4, 12, 256))
    assert out.shape == (4, 512)


def test_ddpm_forward_loss_is_mse_on_noise():
    torch.manual_seed(0)
    ddpm = LatentDDPM(latent_dim=64, timesteps=100, beta_start=1e-4, beta_end=2e-2,
                      unet_kwargs=dict(cond_dim=256))
    cond = torch.randn(3, 5, 256)
    loss = ddpm.forward_loss(torch.randn(3, 64), cond)
    assert loss.dim() == 0 and torch.isfinite(loss)
    # backward works
    loss.backward()
    assert any(p.grad is not None for p in ddpm.parameters())


def test_diffusion_regularizer_inference_no_reverse_sampling():
    reg = DiffusionRegularizer(latent_dim=64, timesteps=100, beta_start=1e-4,
                                beta_end=2e-2)
    z = torch.randn(2, 64)
    out = reg.inference_latent(z)
    # inference must be the identity (deterministic fused latent), NOT reverse sampling
    assert torch.equal(out, z)

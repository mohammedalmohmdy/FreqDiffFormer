"""Tests for the composite loss: composition, ablation switches, components."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from utils.losses import (TripletLoss, FrequencyAlignmentLoss, DiffusionReconstructionLoss,
                           CompositeLoss)


def _fake_outputs(B=8, D=256, L=512):
    torch.manual_seed(0)
    return dict(
        latent=torch.randn(B, L, requires_grad=True),
        freq_pooled=torch.randn(B, D, requires_grad=True),
        spatial_pooled=torch.randn(B, D, requires_grad=True),
        freq_aligned=torch.randn(B, 196, D, requires_grad=True),
        spatial_aligned=torch.randn(B, 30, D, requires_grad=True),
    )


class _DummyModel:
    """Minimal stand-in providing diffusion_loss for CompositeLoss."""
    def __init__(self):
        self._cond = None

    def diffusion_loss(self, latent, cond=None):
        # deterministic tiny loss so the multiplier is observable
        return 2.0 * latent.pow(2).mean()


def test_total_loss_composition():
    out = _fake_outputs()
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    model = _DummyModel()
    fn = CompositeLoss(margin=0.2, lambda1=0.5, lambda2=0.1,
                       ablation=dict(use_triplet=True, use_diffusion=True,
                                     use_recon=True, use_freq_align=True))
    lb = fn(model, out, labels)
    # total == triplet + lambda1*diffusion + lambda2*freq_align
    # note: diffusion term here == lambda1 * (model.diffusion_loss) == 0.5 * 2.0 = 1.0*mse
    assert torch.allclose(lb.total, lb.triplet + lb.diffusion + lb.freq_align, atol=1e-5)


def test_ablation_no_triplet():
    out = _fake_outputs()
    model = _DummyModel()
    fn = CompositeLoss(ablation=dict(use_triplet=False, use_diffusion=True,
                                      use_recon=True, use_freq_align=True))
    lb = fn(model, out, torch.zeros(8, dtype=torch.long))
    assert lb.triplet.item() == 0.0
    assert lb.components["use_triplet"] is False


def test_ablation_no_freq_align():
    out = _fake_outputs()
    model = _DummyModel()
    fn = CompositeLoss(ablation=dict(use_triplet=True, use_diffusion=True,
                                      use_recon=True, use_freq_align=False))
    lb = fn(model, out, torch.zeros(8, dtype=torch.long))
    assert lb.freq_align.item() == 0.0


def test_ablation_no_recon():
    out = _fake_outputs()
    model = _DummyModel()
    fn = CompositeLoss(ablation=dict(use_triplet=True, use_diffusion=True,
                                      use_recon=False, use_freq_align=True))
    lb = fn(model, out, torch.zeros(8, dtype=torch.long))
    assert lb.diffusion.item() == 0.0


def test_ablation_no_diffusion():
    out = _fake_outputs()
    model = _DummyModel()
    fn = CompositeLoss(ablation=dict(use_triplet=True, use_diffusion=False,
                                      use_recon=True, use_freq_align=True))
    lb = fn(model, out, torch.zeros(8, dtype=torch.long))
    assert lb.diffusion.item() == 0.0


def test_freq_alignment_returns_always_lambda2_weighted():
    fn = FrequencyAlignmentLoss(lambda2=0.1)
    a = torch.randn(4, 8); b = torch.randn(4, 8)
    v = fn(a, b).item()
    # in range [0, 2*lambda2] since (1-cos) in [0,2]
    assert 0.0 <= v <= 0.2 + 1e-5


def test_triplet_loss_basic():
    fn = TripletLoss(margin=0.2)
    a = torch.zeros(2, 4); p = torch.zeros(2, 4); n = torch.ones(2, 4)
    # d_ap=0, d_an=1, loss = relu(0 - 1 + 0.2) = 0
    assert fn(a, p, n).item() == 0.0
    # reversed: loss positive
    a = torch.zeros(2, 4); p = torch.ones(2, 4); n = torch.zeros(2, 4)
    assert fn(a, p, n).item() > 0.0


def test_loss_is_differentiable():
    out = _fake_outputs()
    model = _DummyModel()
    fn = CompositeLoss()
    lb = fn(model, out, torch.zeros(8, dtype=torch.long))
    lb.total.backward()
    # at least some inputs require grad and got grad
    assert out["latent"].grad is not None or out["freq_pooled"].grad is not None

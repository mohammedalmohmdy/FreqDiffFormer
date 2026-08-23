"""Tests for the cross-domain transformer: real cross-attention, latent dim."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from models.cross_domain_transformer import CrossDomainTransformer


def test_cdt_is_not_stacked_transformer_encoder():
    # the model must expose cross-attention layers, not one TransformerEncoder
    cdt = CrossDomainTransformer(token_dim=256, latent_dim=512, num_heads=8, num_layers=2)
    assert hasattr(cdt, "freq_layers") and len(cdt.freq_layers) == 2
    # bidirectional cross-attention present
    assert cdt.spatial_layers is not None
    assert len(cdt.spatial_layers) == 2


def test_cdt_output_shapes():
    cdt = CrossDomainTransformer(token_dim=256, latent_dim=512, num_heads=8, num_layers=2)
    out = cdt(torch.randn(2, 196, 256), torch.randn(2, 30, 256))
    assert out["freq_aligned"].shape == (2, 196, 256)
    assert out["spatial_aligned"].shape == (2, 30, 256)
    assert out["latent"].shape == (2, 512)
    assert out["freq_pooled"].shape == (2, 256)
    assert out["spatial_pooled"].shape == (2, 256)


def test_cdt_changes_inputs():
    # cross-attention must modify the input token sequences (not identity).
    cdt = CrossDomainTransformer(token_dim=32, latent_dim=64, num_heads=4, num_layers=2)
    f = torch.randn(2, 16, 32); s = torch.randn(2, 8, 32)
    out = cdt(f, s)
    assert not torch.allclose(out["freq_aligned"], cdt.freq_domain_emb.expand(2, -1, -1) * 0 + f)
    assert not torch.allclose(out["spatial_aligned"], s)


def test_cdt_grad():
    cdt = CrossDomainTransformer(token_dim=64, latent_dim=128, num_heads=4, num_layers=1)
    out = cdt(torch.randn(2, 10, 64), torch.randn(2, 6, 64))
    out["latent"].pow(2).sum().backward()
    assert any(p.grad is not None for p in cdt.parameters())

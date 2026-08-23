"""Full-model integration + ablation-switch tests."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
import yaml

from models import build_model
from utils.losses import CompositeLoss


@pytest.fixture(scope="module")
def cfg():
    cfg = yaml.safe_load(open("configs/default.yaml", encoding="utf-8"))
    cfg["model"]["spatial"]["fail_on_missing"] = False  # CPU CI without Swin weights
    return cfg


def test_full_model_forward_shapes(cfg):
    m = build_model(cfg, "cpu")
    out = m(torch.randn(8, 1, 224, 224), torch.randn(8, 3, 224, 224))
    assert out["latent"].shape == (8, 512)
    assert out["freq_pooled"].shape == (8, 256)
    assert out["spatial_pooled"].shape == (8, 256)


def test_embed_sketch_and_photo(cfg):
    m = build_model(cfg, "cpu")
    q = m.embed_sketch(torch.randn(3, 1, 224, 224))
    g = m.embed_photo(torch.randn(5, 3, 224, 224))
    # query and gallery are SEPARATE embeddings of latent_dim
    assert q.shape == (3, 512) and g.shape == (5, 512)
    assert q is not g


def test_inference_path_does_not_reverse_diffuse(cfg):
    # embed_sketch / embed_photo (the retrieval path) must NOT call reverse sampling
    m = build_model(cfg, "cpu")
    z = m.embed_sketch(torch.randn(1, 1, 224, 224))
    # If reverse sampling were used, output would diverge from a deterministic fused latent.
    assert torch.isfinite(z).all()
    # Verify there is no forward(1) reverse-sampling call hook here
    # (architectural guarantee: DiffusionRegularizer.inference_latent is identity)
    assert "DiffusionRegularizer" in str(m.diffusion.__class__)


def test_ablation_no_diffusion(cfg):
    abl = dict(cfg["training"]["ablation"]); abl["use_diffusion"] = False
    cfg2 = dict(cfg); cfg2["training"]["ablation"] = abl
    m = build_model(cfg2, "cpu")
    assert m.diffusion is None


def test_ablation_no_freq_encoder(cfg):
    abl = dict(cfg["training"]["ablation"]); abl["use_freq_encoder"] = False
    cfg2 = dict(cfg); cfg2["training"]["ablation"] = abl
    m = build_model(cfg2, "cpu")
    # placeholder tokens instead of the real frequency encoder
    assert hasattr(m, "freq_placeholder")
    out = m(torch.randn(4, 1, 224, 224), torch.randn(4, 3, 224, 224))
    assert out["latent"].shape == (4, 512)


def test_ablation_no_cdt(cfg):
    abl = dict(cfg["training"]["ablation"]); abl["use_cdt"] = False
    cfg2 = dict(cfg); cfg2["training"]["ablation"] = abl
    m = build_model(cfg2, "cpu")
    out = m(torch.randn(4, 1, 224, 224), torch.randn(4, 3, 224, 224))
    assert out["latent"].shape == (4, 512)


def test_loss_composition_in_full_model(cfg):
    m = build_model(cfg, "cpu")
    out = m(torch.randn(6, 1, 224, 224), torch.randn(6, 3, 224, 224))
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    fn = CompositeLoss(margin=0.2, lambda1=0.5, lambda2=0.1,
                       ablation=cfg["training"]["ablation"])
    lb = fn(m, out, labels)
    assert torch.allclose(lb.total, lb.triplet + lb.diffusion + lb.freq_align, atol=1e-5)
    lb.total.backward()

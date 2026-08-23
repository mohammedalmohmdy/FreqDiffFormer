"""Tests for the frequency encoder (FEM): shapes, DCT, masking, compression."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from models.freq_encoder import (FreqEncoder, dct_2d, sign_log_compress,
                                  robust_min_max, _radial_masks)


def _cfg():
    return dict(in_channels=2, token_dim=256, num_tokens=196, token_grid=14,
                conv_channels=[64, 128, 256, 256], kernel_size=3,
                strides=[2, 2, 2, 1], paddings=[1, 1, 1, 1], norm="instance",
                activation="relu", use_sign_log=True, use_robust_minmax=True,
                low_freq_mask_ratio=0.25, high_freq_mask_ratio=0.25,
                positional_embedding="learnable", layer_norm=True, input_size=224)


def test_freq_encoder_output_shape():
    m = FreqEncoder(**_cfg())
    assert m.pos_emb.shape == (1, 196, 256)
    out = m(torch.randn(2, 1, 224, 224))
    assert out.shape == (2, 196, 256)


def test_freq_encoder_grad_flow():
    m = FreqEncoder(**_cfg())
    x = torch.randn(2, 1, 224, 224, requires_grad=True)
    m(x).pow(2).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_dct_2d_shape_and_energy():
    # DCT of a constant image should concentrate energy at the (0,0) coefficient.
    x = torch.full((1, 1, 64, 64), 0.5)
    c = dct_2d(x)
    assert c.shape == x.shape
    # the DC term at [0,0] dominates; the rest are ~0
    assert abs(c[0, 0, 0, 0].item()) > abs(c[0, 0, 5, 5].item()) * 1e3


def test_freq_masks_two_bands():
    masks = _radial_masks(224, 224, 0.25, 0.25, torch.device("cpu"))
    assert masks.shape == (2, 224, 224)
    assert masks[0].sum() > 0 and masks[1].sum() > 0


def test_sign_log_compress():
    x = torch.tensor([-3.0, -0.5, 0.0, 0.5, 3.0])
    y = sign_log_compress(x)
    assert torch.allclose(y[0], -torch.log1p(torch.tensor(3.0)))
    assert y[2].item() == 0.0


def test_robust_min_max_range():
    x = torch.randn(2, 2, 16, 16) * 100
    y = robust_min_max(x)
    assert y.min() >= -1e-5 and y.max() <= 1.0 + 1e-5


def test_rejects_batchnorm():
    cfg = _cfg(); cfg["norm"] = "batch"
    with pytest.raises(ValueError):
        FreqEncoder(**cfg)


def test_rejects_wrong_block_count():
    cfg = _cfg(); cfg["conv_channels"] = [64, 128]
    with pytest.raises(ValueError):
        FreqEncoder(**cfg)

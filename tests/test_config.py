"""Test that every manuscript-mandated config value is present and correct."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import pytest


CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "configs", "default.yaml")


def _load():
    return yaml.safe_load(open(CONFIG, encoding="utf-8"))


def test_latent_dim():
    assert _load()["model"]["latent_dim"] == 512


def test_input_size():
    assert _load()["model"]["input_size"] == 224


def test_diffusion_timesteps():
    assert _load()["diffusion"]["timesteps"] == 100


def test_beta_range():
    cfg = _load()["diffusion"]
    assert abs(cfg["beta_start"] - 1e-4) < 1e-12
    assert abs(cfg["beta_end"] - 2e-2) < 1e-9
    assert cfg["schedule"] == "linear"


def test_batch_size():
    assert _load()["training"]["batch_size"] == 32


def test_epochs():
    assert _load()["training"]["epochs"] == 50


def test_lr():
    assert abs(_load()["training"]["lr"] - 1e-4) < 1e-12


def test_margin():
    assert abs(_load()["training"]["margin"] - 0.2) < 1e-9


def test_lambdas():
    t = _load()["training"]
    assert abs(t["lambda1_diffusion"] - 0.5) < 1e-9
    assert abs(t["lambda2_freq_align"] - 0.1) < 1e-9


def test_optimizer():
    assert _load()["training"]["optimizer"] == "adamw"


def test_swin_backbone():
    assert _load()["model"]["spatial"]["backbone"] == "swin_tiny_patch4_window7_224"


def test_freq_encoder_params():
    f = _load()["model"]["freq"]
    assert f["conv_channels"] == [64, 128, 256, 256]
    assert f["strides"] == [2, 2, 2, 1]
    assert f["paddings"] == [1, 1, 1, 1]
    assert f["num_tokens"] == 196
    assert f["token_dim"] == 256
    assert f["token_grid"] == 14
    assert f["norm"] == "instance"
    assert f["use_sign_log"] is True
    assert f["use_robust_minmax"] is True
    assert f["positional_embedding"] == "learnable"
    assert f["layer_norm"] is True


def test_inference_no_reverse_sampling():
    assert _load()["diffusion"]["inference_reverse_sampling"] is False


def test_gallery_sizes():
    d = _load()["data"]["datasets"]
    assert d["sketchy"]["gallery_size"] == 500
    assert d["shoev2"]["gallery_size"] == 400
    assert d["chairv2"]["gallery_size"] == 250
    assert d["tu_berlin"]["gallery_size"] == 400
    assert d["sketchy"]["num_classes"] == 25
    assert d["shoev2"]["num_classes"] == 20
    assert d["chairv2"]["num_classes"] == 15
    assert d["tu_berlin"]["num_classes"] == 50


def test_stats_protocol():
    s = _load()["stats"]
    assert s["test"] == "wilcoxon_signed_rank"
    assert s["alpha"] == 0.01
    assert s["confidence_interval"] == 95
    assert s["per_dataset_observations"] == {
        "sketchy": 25, "shoev2": 20, "chairv2": 15, "tu_berlin": 50}


def test_timing_protocol():
    t = _load()["timing"]
    assert t["batch_size"] == 1
    assert t["warmup_iterations"] == 50
    assert t["measured_iterations"] == 300
    assert t["hardware"] == "RTX_A6000"


def test_ablation_switches_present():
    a = _load()["training"]["ablation"]
    for k in ["use_freq_encoder", "use_freq_align", "use_cdt", "use_diffusion",
              "use_triplet", "use_recon"]:
        assert k in a and a[k] is True

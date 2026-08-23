"""Tests for the retrieval pipeline: ranking, mAP@200, Top-1, per-class, caching."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from utils.retrieval import (compute_ranking, average_precision_at_k, evaluate_retrieval,
                              cache_gallery)


def test_average_precision_at_k_perfect():
    assert average_precision_at_k(np.array([1, 1, 1, 0, 0]), 5) == 1.0


def test_average_precision_at_k_zero_relevant():
    assert average_precision_at_k(np.array([0, 0, 0]), 3) == 0.0


def test_average_precision_at_k_class_0_relevant():
    # class label 0 must be treated as relevant when query is class 0
    rel = (np.array([0, 0, 0, 1, 1]) == 0).astype(np.int64)
    assert rel.tolist() == [1, 1, 1, 0, 0]
    assert average_precision_at_k(rel, 5) == 1.0


def test_compute_ranking_top1():
    q = np.array([[1.0, 0.0]])
    g = np.array([[0.1, 0.9], [1.0, 0.0]])
    order, _ = compute_ranking(q, g)
    assert order[0, 0] == 1  # most similar is index 1


def test_evaluate_retrieval_perfect_and_top1():
    g = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9], [1, 0], [0, 1]], dtype=float)
    gl = np.array([0, 0, 1, 1, 0, 1])
    q = np.array([[1, 0], [0, 1]], dtype=float); ql = np.array([0, 1])
    r = evaluate_retrieval(q, ql, g, gl, top_k=200)
    assert r["mAP@200"] == 1.0
    assert r["top1"] == 1.0
    assert set(r["per_class_mAP@200"].keys()) == {0, 1}


def test_evaluate_retrieval_per_class_keys():
    rng = np.random.default_rng(0)
    g = rng.standard_normal((40, 16)).astype(np.float32)
    gl = rng.integers(0, 4, size=40)
    q = g[:4].copy(); ql = gl[:4].copy()
    r = evaluate_retrieval(q, ql, g, gl, top_k=200)
    assert "per_class_mAP@200" in r
    assert all(k in r["per_class_mAP@200"] for k in ql.tolist())


def test_gallery_cache_roundtrip(tmp_path):
    import torch
    from models import build_model
    import yaml
    cfg = yaml.safe_load(open("configs/default.yaml", encoding="utf-8"))
    cfg["model"]["spatial"]["fail_on_missing"] = False
    model = build_model(cfg, "cpu")
    loader = [(torch.randn(2, 3, 224, 224), torch.tensor([0, 1]))]
    a, la = cache_gallery(model, loader, "cpu", "sketchy", "abc",
                          "/checkpoints/c.pth", str(tmp_path))
    assert a.shape == (2, 512) and la.shape == (2,)
    # second call reuses cache
    import numpy as np
    b, lb = cache_gallery(model, loader, "cpu", "sketchy", "abc",
                          "/checkpoints/c.pth", str(tmp_path))
    assert np.array_equal(a, b)

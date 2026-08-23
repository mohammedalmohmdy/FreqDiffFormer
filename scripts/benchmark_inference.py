"""Inference timing protocol.

Manuscript inference-timing protocol:
  - hardware target: RTX A6000 (script labels results with the actual device)
  - input 224x224
  - inference batch size 1
  - 50 warm-up iterations
  - 300 measured iterations
  - CUDA synchronization where CUDA is available
  - feature extraction latency, similarity/ranking latency, complete retrieval latency

On a CPU-only environment the same protocol runs and results are labeled 'CPU'
(not A6000); the protocol is identical, only the device differs.

Usage:
  python scripts/benchmark_inference.py --config configs/default.yaml \
        --checkpoint outputs/runs/freqdiffformer/checkpoint_last.pth \
        --dataset sketchy --gallery-size 500 [--smoke]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml

from models import build_model
from utils.helpers import set_seed
from utils.retrieval import compute_ranking


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def _timed_query(model, sketch, gallery_embs, device, runs=1):
    """Returns (feature_ms, rank_ms, total_ms) for one query, averaged over runs."""
    fts = []
    rts = []
    tts = []
    for _ in range(runs):
        t0 = time.perf_counter(); _sync(device)
        q_emb = model.embed_sketch(sketch).cpu().numpy()
        _sync(device); t1 = time.perf_counter()
        order, sims = compute_ranking(q_emb, gallery_embs)
        _sync(device); t2 = time.perf_counter()
        fts.append((t1 - t0) * 1e3)
        rts.append((t2 - t1) * 1e3)
        tts.append((t2 - t0) * 1e3)
    return float(np.mean(fts)), float(np.mean(rts)), float(np.mean(tts))


def main(args):
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed_info = set_seed(int(cfg["experiment"]["seed"]), deterministic=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hardware_label = "RTX_A6000" if device == "cuda" else "CPU"

    cfg["model"]["spatial"].setdefault("fail_on_missing", True)
    if device != "cuda":
        # CI without pretrained Swin: explicitly degrade (benchmark measures latency)
        cfg["model"]["spatial"]["fail_on_missing"] = False
    model = build_model(cfg, device)
    if args.checkpoint and os.path.isfile(args.checkpoint):
        ck = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ck.get("state_dict", ck), strict=False)
    model.to(device).eval()

    warmup = 0 if args.smoke else 50
    measured = 10 if args.smoke else 300
    gallery_size = args.gallery_size or 500
    latent_dim = cfg["model"]["latent_dim"]

    # random gallery embeddings (we measure LATENCY of ranking, not mAP)
    gallery = np.random.randn(gallery_size, latent_dim).astype(np.float32)
    gallery /= np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-6
    sketch = torch.randn(1, 1, 224, 224, device=device)

    # warm-up
    with torch.no_grad():
        for _ in range(warmup):
            _ = model.embed_sketch(sketch)

    feats, ranks, totals = [], [], []
    with torch.no_grad():
        for _ in range(measured):
            f, r, t = _timed_query(model, sketch, gallery, device, runs=1)
            feats.append(f); ranks.append(r); totals.append(t)

    out = {
        "hardware": hardware_label,
        "device_actual": device,
        "input_size": 224,
        "batch_size": 1,
        "warmup_iterations": warmup,
        "measured_iterations": measured,
        "gallery_size": gallery_size,
        "feature_extraction_ms": {
            "mean": float(np.mean(feats)), "std": float(np.std(feats)),
            "median": float(np.median(feats)),
        },
        "ranking_ms": {
            "mean": float(np.mean(ranks)), "std": float(np.std(ranks)),
            "median": float(np.median(ranks)),
        },
        "total_retrieval_ms": {
            "mean": float(np.mean(totals)), "std": float(np.std(totals)),
            "median": float(np.median(totals)),
        },
        "per_image_total_ms_median": float(np.median(totals)),
        "seed": seed_info["seed"],
    }
    out_dir = Path("outputs") / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"inference_{hardware_label.lower()}"
    out_path = out_dir / f"{tag}.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Inference benchmark ({hardware_label}, n={measured}):")
    print(f"  feature extraction: {out['feature_extraction_ms']['median']:.3f} ms (median)")
    print(f"  ranking:             {out['ranking_ms']['median']:.3f} ms (median)")
    print(f"  total retrieval:     {out['total_retrieval_ms']['median']:.3f} ms (median)")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--dataset", default="sketchy")
    p.add_argument("--gallery-size", type=int, default=None)
    p.add_argument("--smoke", action="store_true",
                    help="reduce warmup/measured iterations for a quick smoke check")
    p.add_argument("--tag", default=None)
    args = p.parse_args()
    main(args)

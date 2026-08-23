"""Training-time benchmark: diffusion overhead.

Measures the per-step training latency of the full model versus a no-diffusion
variant under the SAME training configuration, so the diffusion overhead can be
reported. Uses the manuscript training protocol defaults (batch 32) but accepts a
smoke override.

Outputs feature/loss latencies (with and without diffusion) under the same
config; reports mean/std/median ms/step.

Usage:
  python scripts/benchmark_training.py --config configs/default.yaml [--smoke] \
        [--checkpoint outputs/runs/.../checkpoint_last.pth]
"""

from __future__ import annotations

import argparse
import copy
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
from utils.losses import CompositeLoss


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def _bench(model, loss_fn, sketch, photo, labels, device, measured):
    times = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    with torch.enable_grad():
        # warm-up
        out = model(sketch, photo)
        lb = loss_fn(model, out, labels)
        lb.total.backward()
        optimizer.zero_grad()
        for _ in range(measured):
            _sync(device); t0 = time.perf_counter()
            out = model(sketch, photo)
            lb = loss_fn(model, out, labels)
            lb.total.backward()
            _sync(device); t1 = time.perf_counter()
            times.append((t1 - t0) * 1e3)
            optimizer.zero_grad()
    return times


def _stats(x):
    return {"mean": float(np.mean(x)), "std": float(np.std(x)), "median": float(np.median(x))}


def main(args):
    with open(args.config, encoding="utf-8") as f:
        cfg_full = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg_full)
    set_seed(int(cfg["experiment"]["seed"]), deterministic=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["model"]["spatial"]["fail_on_missing"] = device != "cuda"

    batch = 4 if args.smoke else int(cfg["training"]["batch_size"])  # 32 normally
    measured = 5 if args.smoke else 50
    latent = cfg["model"]["latent_dim"]
    sketch = torch.randn(batch, 1, 224, 224, device=device)
    photo = torch.randn(batch, 3, 224, 224, device=device)
    labels = torch.randint(0, 25, (batch,), device=device)

    # full model (with diffusion)
    m_full = build_model(cfg, device)
    loss_full = CompositeLoss(margin=cfg["training"]["margin"],
                               lambda1=cfg["training"]["lambda1_diffusion"],
                               lambda2=cfg["training"]["lambda2_freq_align"],
                               ablation=cfg["training"]["ablation"])
    t_full = _bench(m_full, loss_full, sketch, photo, labels, device, measured)

    # no-diffusion variant: same config but use_diffusion=False
    cfg_nd = copy.deepcopy(cfg)
    abl = dict(cfg_nd["training"]["ablation"]); abl["use_diffusion"] = False
    cfg_nd["training"]["ablation"] = abl
    m_nd = build_model(cfg_nd, device)
    loss_nd = CompositeLoss(margin=cfg["training"]["margin"],
                             lambda1=cfg["training"]["lambda1_diffusion"],
                             lambda2=cfg["training"]["lambda2_freq_align"],
                             ablation=abl)
    t_nd = _bench(m_nd, loss_nd, sketch, photo, labels, device, measured)

    times_full = _stats(t_full)
    times_nd = _stats(t_nd)
    overhead = times_full["median"] - times_nd["median"]

    out = {
        "device_actual": device,
        "batch_size": batch,
        "measured_steps": measured,
        "with_diffusion_ms_per_step": times_full,
        "without_diffusion_ms_per_step": times_nd,
        "diffusion_overhead_ms_per_step_median": float(overhead),
        "with_diffusion_has_trainable_diffusion": True,
        "without_diffusion_has_trainable_diffusion": False,
    }
    out_dir = Path("outputs") / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "training_diffusion_overhead.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Training step (batch={batch}, n={measured}):")
    print(f"  with diffusion    : {times_full['median']:.2f} ms/step (median)")
    print(f"  without diffusion : {times_nd['median']:.2f} ms/step (median)")
    print(f"  diffusion overhead: {overhead:+.2f} ms/step (median)")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    main(args)

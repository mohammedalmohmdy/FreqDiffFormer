"""FreqDiffFormer training.

Implements the manuscript training protocol:
  - AdamW, lr=1e-4, batch_size=32, 50 epochs
  - L_total = L_triplet + lambda1*L_diffusion + lambda2*L_frequency_alignment
  - timesteps sampled uniformly during training
  - diffusion as a TRAINING-TIME regularizer (no reverse sampling at retrieval)
  - ablation switches via config overlays (each removes ONLY one component)
  - deterministic seeding + run metadata recording

Usage:
  python scripts/train.py --config configs/default.yaml [--dataset sketchy]
                        [--overlay configs/ablation/no_diffusion.yaml]
                        [--run-name run0] [--seed 42]
                        [--max-train-batches N] [--validate-every 5]

Optional --max-train-batches / --validate-every let the pipeline run as a smoke
test on CPU without changing the real protocol defaults (config stays 50/32).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# allow running as a script from anywhere: put repo root on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import build_model
from utils.config import load_config
from utils.datasets import build_train_dataset, build_split_dataset, get_dataset_cfg
from utils.helpers import set_seed, save_checkpoint, run_metadata, save_metadata, config_hash
from utils.losses import CompositeLoss
from utils.transforms import default_transforms


def _move(batch, device):
    sketch, photo, label = batch
    return sketch.to(device), photo.to(device), label.to(device)


def _train_one_epoch(model, loader, loss_fn, optimizer, device, max_batches=None,
                    log_interval=20):
    model.train()
    running = dict(total=0.0, triplet=0.0, diffusion=0.0, freq_align=0.0)
    n = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        sketch, photo, label = _move(batch, device)
        outputs = model(sketch, photo)
        lb = loss_fn(model, outputs, label)
        optimizer.zero_grad()
        lb.total.backward()
        optimizer.step()
        running["total"] += lb.total.item()
        running["triplet"] += lb.triplet.item()
        running["diffusion"] += lb.diffusion.item()
        running["freq_align"] += lb.freq_align.item()
        n += 1
        if n % log_interval == 0:
            print(f"  step {n}: total={lb.total.item():.4f} "
                  f"(triplet={lb.triplet.item():.4f}, "
                  f"diff={lb.diffusion.item():.4f}, "
                  f"freq_align={lb.freq_align.item():.4f})")
    return {k: v / max(n, 1) for k, v in running.items()}


def _validate(model, loader, device, max_batches=None):
    """Lightweight validation loss on the val split (used only for lambda search).

    Not retrieval mAP; just a sanity/triplet-loss proxy.
    """
    model.eval()
    loss_fn = nn.TripletMarginLoss(margin=0.2)
    tot, n = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            sketch, photo, label = _move(batch, device)
            feats = model.embed_sketch(sketch)
            gal = model.embed_photo(photo)
            # proxy: distance same-class should be small -> use identity triplet
            d = nn.functional.pairwise_distance(feats, gal)
            tot += d.mean().item()
            n += 1
    return tot / max(n, 1)


def main(args):
    cfg = load_config(args.config, overlays=args.overlay)
    if args.dataset is not None:
        cfg["data"]["dataset"] = args.dataset
    if args.seed is not None:
        cfg["experiment"]["seed"] = args.seed

    seed = int(cfg["experiment"]["seed"])
    deterministic = bool(cfg["experiment"].get("deterministic", True))
    seed_info = set_seed(seed, deterministic=deterministic)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = cfg["data"]["dataset"]
    transform = default_transforms(cfg["data"]["image_size"])
    train_ds = build_train_dataset(cfg, dataset, transform, split="train")
    bs = int(cfg["training"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                             num_workers=int(cfg["data"]["num_workers"]),
                             drop_last=True)

    model = build_model(cfg, device)
    loss_fn = CompositeLoss(
        margin=float(cfg["training"]["margin"]),
        lambda1=float(cfg["training"]["lambda1_diffusion"]),
        lambda2=float(cfg["training"]["lambda2_freq_align"]),
        ablation=cfg["training"]["ablation"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg.get("training", {}).get("weight_decay", 0.01)),
    )

    # run output dir
    out_dir = os.path.join(cfg["logging"]["output_dir"], "runs",
                            args.run_name or cfg["experiment"]["name"])
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # validation loader
    val_every = args.validate_every
    val_loader = None
    if val_every and val_every > 0:
        try:
            val_ds = build_train_dataset(cfg, dataset, transform, split="val")
            val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                                    num_workers=int(cfg["data"]["num_workers"]))
        except Exception as e:
            print(f"[warn] no validation split loaded ({e}); skipping validation")

    epochs = int(cfg["training"]["epochs"])
    log_interval = int(cfg["logging"].get("log_interval", 20))
    history = []
    for epoch in range(epochs):
        stats = _train_one_epoch(model, train_loader, loss_fn, optimizer, device,
                                max_batches=args.max_train_batches,
                                log_interval=log_interval)
        line = (f"Epoch {epoch+1}/{epochs} "
                f"total={stats['total']:.4f} triplet={stats['triplet']:.4f} "
                f"diff={stats['diffusion']:.4f} freq_align={stats['freq_align']:.4f}")
        if val_loader is not None and (epoch + 1) % val_every == 0:
            v = _validate(model, val_loader, device, args.max_val_batches)
            line += f" val_proxy={v:.4f}"
        print(line)
        history.append({"epoch": epoch + 1, **stats})

    # checkpoint + metadata
    ckpt_path = os.path.join(out_dir, "checkpoint_last.pth")
    save_checkpoint({
        "epoch": epochs,
        "state_dict": model.state_dict(),
        "config": cfg,
    }, ckpt_path)

    meta = run_metadata(cfg, seed_info, extra={
        "dataset": dataset,
        "run_name": args.run_name or cfg["experiment"]["name"],
        "config_hash": config_hash(cfg),
        "epochs": epochs,
        "batch_size": bs,
        "loss_history": history,
    })
    save_metadata(meta, os.path.join(out_dir, "run_metadata.json"))
    print(f"Saved checkpoint -> {ckpt_path}")
    print(f"Saved metadata  -> {os.path.join(out_dir, 'run_metadata.json')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--overlay", nargs="*", default=None,
                    help="ablation/baseline config(s) merged on top of --config")
    p.add_argument("--dataset", default=None,
                    help="sketchy | shoev2 | chairv2 | tu_berlin")
    p.add_argument("--run-name", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-train-batches", type=int, default=None,
                    help="cap batches per epoch (smoke test only; does not change config)")
    p.add_argument("--max-val-batches", type=int, default=None)
    p.add_argument("--validate-every", type=int, default=0)
    args = p.parse_args()
    main(args)

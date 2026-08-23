"""FreqDiffFormer retrieval evaluation.

Manuscript retrieval protocol:
  - main benchmark evaluation uses the TEST split (not validation)
  - retrieval is query-sketch -> photo-gallery ranking
  - gallery photo embeddings are PRECOMPUTED OFFLINE (and cached)
  - cosine similarity ranking, mAP@200 (primary) and Top-1 (secondary)
  - dataset-specific gallery sizes (500/400/250/400)
  - NO iterative reverse-diffusion sampling at inference (uses deterministic
    fused latent from the Cross-Domain Transformer directly)

Outputs a JSON results file under outputs/<run>/results/eval_<dataset>.json
containing overall metrics and per-class mAP@200 (consumed by the stats and
Supplementary Table S1 scripts).

Usage:
  python scripts/eval.py --config configs/default.yaml
                        --checkpoint outputs/runs/freqdiffformer/checkpoint_last.pth
                        --dataset sketchy
                        [--split test|val]
                        [--no-cache]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# allow running as a script from anywhere: put repo root on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from models import build_model
from utils.config import load_config
from utils.datasets import build_split_dataset, get_dataset_cfg, DATASET_REGISTRY
from utils.helpers import config_hash, set_seed
from utils.retrieval import extract_embeddings, evaluate_retrieval, cache_gallery
from utils.transforms import default_transforms


def load_model_from_checkpoint(cfg, ckpt_path, device):
    cfg = dict(cfg)
    # In CI environments without pretrained Swin weights, allow a clearly-marked
    # fallback ONLY when the config opts in. The default manuscript pipeline keeps
    # fail_on_missing=true and never substitutes a lightweight backbone silently.
    if cfg["model"]["spatial"].get("fail_on_missing", True) and not _swin_available():
        # If we are loading a checkpoint that already trained a fallback backbone,
        # we must allow the same fallback at eval time; detected by checkpoint keys.
        allow = _ckpt_uses_fallback(ckpt_path)
        if allow:
            cfg["model"]["spatial"]["fail_on_missing"] = False
        else:
            raise RuntimeError(
                "Swin Transformer-Tiny is required by the manuscript and timm/weights "
                "are unavailable. Refusing to silently substitute a lightweight backbone "
                "at evaluation. Provide timm + pretrained weights, or set "
                "model.spatial.fail_on_missing=false explicitly.")
    model = build_model(cfg, device)
    if ckpt_path and os.path.isfile(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        state = ck.get("state_dict", ck)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[info] missing state keys (lenient load): {len(missing)}")
        if unexpected:
            print(f"[info] unexpected state keys (lenient load): {len(unexpected)}")
    else:
        print("[warn] no checkpoint found; using randomly initialized model")
    model.to(device).eval()
    return model


def _swin_available() -> bool:
    try:
        import timm  # noqa: F401
        return True
    except Exception:
        return False


def _ckpt_uses_fallback(ckpt_path: str) -> bool:
    """Detect whether a checkpoint was trained with the fallback dummy backbone."""
    if not ckpt_path or not os.path.isfile(ckpt_path):
        return False
    try:
        ck = torch.load(ckpt_path, map_location="cpu")
        keys = list(ck.get("state_dict", ck).keys())
        return any("spatial_encoder._feat_dim" in k or
                   "spatial_encoder.conv" in k for k in keys)
    except Exception:
        return False


def main(args):
    cfg = load_config(args.config, overlays=args.overlay)
    if args.dataset is not None:
        cfg["data"]["dataset"] = args.dataset

    seed_info = set_seed(int(cfg["experiment"]["seed"]),
                         deterministic=bool(cfg["experiment"].get("deterministic", True)))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = cfg["data"]["dataset"]
    dc = get_dataset_cfg(cfg, dataset)
    gallery_size = dc.get("gallery_size", DATASET_REGISTRY[dataset]["gallery_size"])
    transform = default_transforms(cfg["data"]["image_size"])
    top_k = int(cfg["eval"].get("top_k", 200))

    model = load_model_from_checkpoint(cfg, args.checkpoint, device)

    split = args.split or ("test" if cfg["eval"].get("use_test_split", True) else "val")
    if split != "test":
        print(f"[note] evaluating on split='{split}' (benchmark main eval uses 'test')")

    # gallery
    q_ds = build_split_dataset(cfg, dataset, "query", transform)
    g_ds = build_split_dataset(cfg, dataset, "gallery", transform,
                               restricted_gallery_size=gallery_size if args.restrict_gallery else None)
    if len(q_ds) == 0:
        raise RuntimeError(f"Query dataset '{dataset}' is empty. Provide split files "
                          f"under {dc['root']}/splits/")
    if len(g_ds) == 0:
        raise RuntimeError(f"Gallery dataset '{dataset}' is empty. Provide a gallery "
                          f"split under {dc['root']}/splits/gallery.csv")

    batch = max(1, int(cfg["training"]["batch_size"]) // 2 or 16)
    q_loader = DataLoader(q_ds, batch_size=batch, shuffle=False, num_workers=0)
    g_loader = DataLoader(g_ds, batch_size=batch, shuffle=False, num_workers=0)

    # precompute + cache gallery
    cache_dir = cfg["eval"].get("gallery_cache_dir", "./outputs/gallery_cache")
    chash = config_hash(cfg)
    if args.no_cache:
        g_embs, g_labels = extract_embeddings(model, g_loader, device, kind="photo")
    else:
        g_embs, g_labels = cache_gallery(model, g_loader, device, dataset, chash,
                                         args.checkpoint, cache_dir)

    q_embs, q_labels = extract_embeddings(model, q_loader, device, kind="sketch")

    results = evaluate_retrieval(q_embs, q_labels, g_embs, g_labels, top_k=top_k)
    results["dataset"] = dataset
    results["split"] = split
    results["gallery_size_used"] = int(g_embs.shape[0])
    results["seed"] = seed_info["seed"]

    out_dir = Path(cfg["logging"]["output_dir"]) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or dataset
    out_path = out_dir / f"eval_{tag}.json"
    # per-class CSV for downstream stats/S1
    per_class_path = out_path.with_name(f"per_class_{tag}.csv")
    import csv
    with open(per_class_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "mAP@200"])
        for cid in sorted(results["per_class_mAP@200"].keys()):
            w.writerow([cid, results["per_class_mAP@200"][cid]])
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    print(f"[{dataset}] mAP@{top_k}={results['mAP@200']:.4f} "
          f"Top-1={results['top1']:.4f} "
          f"(queries={results['num_queries']}, gallery={results['num_gallery']})")
    print(f"Saved {out_path}")
    print(f"Saved {per_class_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--overlay", nargs="*", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--split", default=None, help="test (default) or val")
    p.add_argument("--tag", default=None, help="output file tag (default=dataset)")
    p.add_argument("--no-cache", action="store_true",
                    help="do not reuse the gallery cache; recompute")
    p.add_argument("--restrict-gallery", action="store_true",
                    help="truncate gallery to the dataset gallery_size")
    args = p.parse_args()
    main(args)

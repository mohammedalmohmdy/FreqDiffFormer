"""lambda1 / lambda2 grid search on the Sketchy VALIDATION split.

Manuscript: lambda1 / lambda2 are FIXED (not learnable), selected on the Sketchy
validation split from:
   lambda1 (diffusion recon)  in {0.1, 0.5, 1.0}
   lambda2 (frequency align) in {0.01, 0.1, 0.5}
This script runs the full 3x3 grid, training each setting on Sketchy and
evaluating mAP@200 on the Sketchy VALIDATION split ONLY (never the test set).
It writes a CSV of (lambda1, lambda2, val_mAP@200, val_top1). It does NOT
hard-code the selected values; canonical defaults (0.5 / 0.1) remain in
configs/default.yaml.

Usage:
  python scripts/grid_search_lambdas.py --config configs/default.yaml \
        --grid-config configs/lambdas_grid.yaml \
        [--max-train-batches N]   # smoke mode
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml


def _run(cmd):
    print(">>>", " ".join(cmd))
    r = subprocess.run(cmd, cwd=os.getcwd())
    if r.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}")


def main(args):
    grid = yaml.safe_load(open(args.grid_config, encoding="utf-8"))
    base = yaml.safe_load(open(args.config, encoding="utf-8"))
    data_cfg = dict(base["data"])

    l1s = grid["grid"]["lambda1"]
    l2s = grid["grid"]["lambda2"]
    dataset = grid.get("dataset", "sketchy")
    override = grid.get("training_override", {})

    out_dir = Path("outputs") / "lambda_grid"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"lambda_grid_{dataset}.csv"

    rows = []
    for l1 in l1s:
        for l2 in l2s:
            tag = f"lam_p{int(l1*1000)}{int(l2*100)}_{dataset}".replace("+", "p")
            run_cfg_path = out_dir / f"{tag}.yaml"
            # build a synthetic config overlay for this cell of the grid
            overlay = {
                "experiment": {"name": tag},
                "training": {"lambda1_diffusion": float(l1), "lambda2_freq_align": float(l2),
                              **override},
            }
            with open(run_cfg_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(overlay, f)

            train_cmd = [sys.executable, "scripts/train.py",
                         "--config", args.config, "--overlay", str(run_cfg_path),
                         "--dataset", dataset, "--seed", "42", "--run-name", tag]
            if args.max_train_batches is not None:
                train_cmd += ["--max-train-batches", str(args.max_train_batches)]
            _run(train_cmd)

            ckpt = Path("outputs") / "runs" / tag / "checkpoint_last.pth"
            eval_cmd = [sys.executable, "scripts/eval.py",
                        "--config", args.config, "--overlay", str(run_cfg_path),
                        "--checkpoint", str(ckpt), "--dataset", dataset,
                        "--split", "val", "--tag", tag, "--no-cache"]
            _run(eval_cmd)

            rj = Path("outputs") / "results" / f"eval_{tag}.json"
            res = json.loads(rj.read_text(encoding="utf-8"))
            rows.append({"lambda1": l1, "lambda2": l2,
                          "val_mAP@200": res["mAP@200"], "val_top1": res["top1"]})
            print(f"[grid] lambda1={l1} lambda2={l2} -> "
                  f"val mAP@200={res['mAP@200']:.4f} top1={res['top1']:.4f}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["lambda1", "lambda2", "val_mAP@200", "val_top1"])
        w.writeheader()
        w.writerows(rows)
    print(f"Saved grid results -> {csv_path}")
    print("Canonical (fixed) defaults in configs/default.yaml: lambda1=0.5, lambda2=0.1")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--grid-config", default="configs/lambdas_grid.yaml")
    p.add_argument("--max-train-batches", type=int, default=None)
    args = p.parse_args()
    main(args)

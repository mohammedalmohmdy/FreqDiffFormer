"""Three independent runs with mean +/- SD aggregation.

Manuscript: main quantitative and ablation experiments use THREE INDEPENDENT runs;
reported numbers are mean +/- SD. SD is computed from actual per-run results,
never fabricated.

For each run this script:
  - trains a model with a configurable run seed
  - evaluates retrieval on the test split (mAP@200 + Top-1, per-class mAP)
  - writes per-run results JSON under outputs/<tag>/run{0,1,2}/
Then it aggregates mean/std across the runs and writes outputs/<tag>/aggregate.json.

Usage:
  python scripts/run_three.py --config configs/default.yaml \
        --seeds 42 43 44 --dataset sketchy \
        [--max-train-batches N]   # smoke mode (does NOT change config defaults)

The script imports train.train_main and eval.eval_main-equivalent functions; to
keep this as a single self-contained orchestrator, it shells out to the train
and eval scripts via subprocess so the EXACT same code paths are exercised.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run(cmd):
    print(">>>", " ".join(cmd))
    r = subprocess.run(cmd, cwd=os.getcwd())
    if r.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)} (returncode={r.returncode})")


def aggregate_results(run_json_paths):
    """Compute mean +/- SD of mAP@200 / Top-1 across runs (never fabricated)."""
    runs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in run_json_paths]
    metrics = {}
    for key in ("mAP@200", "top1"):
        vals = np.array([float(r[key]) for r in runs], dtype=np.float64)
        metrics[key] = {
            "values": vals.tolist(),
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1) if len(vals) > 1 else 0.0),
        }
    # per-class mean/std over runs (only if all runs have the same class set)
    per_class_runs = [r.get("per_class_mAP@200", {}) for r in runs]
    common = set(per_class_runs[0].keys())
    for pc in per_class_runs[1:]:
        common &= set(pc.keys())
    pc_agg = {}
    if common:
        for c in sorted(common, key=int):
            vals = np.array([float(pc[str(c)]) for pc in per_class_runs], dtype=np.float64)
            pc_agg[int(c)] = {"mean": float(vals.mean()),
                              "std": float(vals.std(ddof=1) if len(vals) > 1 else 0.0)}
    metrics["per_class_mAP@200_mean_sd"] = pc_agg
    return {"num_runs": len(runs), "runs": run_json_paths, "metrics": metrics}


def main(args):
    out_dir = Path(args.output_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = args.seeds
    if len(seeds) != 3:
        raise ValueError("Three independent runs are required by the manuscript.")

    run_jsons = []
    for i, seed in enumerate(seeds):
        run_name = f"{args.tag}/run{i}"
        # 1) train
        train_cmd = [sys.executable, "scripts/train.py",
                      "--config", args.config,
                      "--dataset", args.dataset,
                      "--seed", str(seed),
                      "--run-name", run_name]
        if args.overlay:
            for o in args.overlay:
                train_cmd += ["--overlay", o]
        if args.max_train_batches is not None:
            train_cmd += ["--max-train-batches", str(args.max_train_batches),
                          "--max-val-batches", str(args.max_train_batches),
                          "--validate-every", "0"]
        _run(train_cmd)

        ckpt = Path(args.output_dir) / "runs" / run_name / "checkpoint_last.pth"
        # 2) eval on test split
        eval_cmd = [sys.executable, "scripts/eval.py",
                    "--config", args.config,
                    "--checkpoint", str(ckpt),
                    "--dataset", args.dataset,
                    "--split", "test",
                    "--tag", run_name]
        if args.overlay:
            eval_cmd += ["--overlay"] + list(args.overlay)
        _run(eval_cmd)

        rj = Path(args.output_dir) / "results" / f"eval_{run_name}.json"
        if not rj.exists():
            raise RuntimeError(f"Run {i} eval output missing: {rj}")
        run_jsons.append(str(rj))

    agg = aggregate_results(run_jsons)
    agg_path = out_dir / "aggregate.json"
    agg_path.write_text(json.dumps(agg, indent=2, sort_keys=True), encoding="utf-8")
    print("\n=== THREE-RUN AGGREGATION ===")
    print(f"mAP@200: {agg['metrics']['mAP@200']['mean']:.4f} "
          f"+/- {agg['metrics']['mAP@200']['std']:.4f}  "
          f"(values={agg['metrics']['mAP@200']['values']})")
    print(f"Top-1:   {agg['metrics']['top1']['mean']:.4f} "
          f"+/- {agg['metrics']['top1']['std']:.4f}  "
          f"(values={agg['metrics']['top1']['values']})")
    print(f"Saved {agg_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--overlay", nargs="*", default=None)
    p.add_argument("--dataset", default="sketchy")
    p.add_argument("--seeds", type=int, nargs=3, default=[42, 43, 44])
    p.add_argument("--tag", default="freqdiffformer_3run")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--max-train-batches", type=int, default=None)
    args = p.parse_args()
    main(args)

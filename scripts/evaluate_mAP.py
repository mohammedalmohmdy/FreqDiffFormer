"""Legacy entry point — kept for backwards compatibility.

This script previously computed retrieval metrics on paired (sketch, photo)
batches. The manuscript requires proper query-sketch -> photo-gallery ranking,
which is implemented in `scripts/eval.py` (with gallery precompute/cache,
per-class mAP, dataset-specific gallery sizes). This shim forwards to that
pipeline and refuses to run the legacy degenerate "same-batch q==g" mode.

Usage (forwarded):
    python scripts/evaluate_mAP.py --config configs/default.yaml \
        --checkpoint outputs/runs/<name>/checkpoint_last.pth --dataset sketchy
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Forward to the real retrieval evaluator (test split by default).
print("[evaluate_mAP] Forwarding to scripts/eval.py (proper query->gallery retrieval).")
print("[evaluate_mAP] The legacy paired-batch mode is intentionally removed because "
      "it produced degenerate q_labels == g_labels rankings; use eval.py instead.")

from scripts import eval as eval_script  # noqa: E402

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Legacy mAP entry (forwards to eval.py)")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--checkpoint", default="outputs/checkpoint_last.pth")
    p.add_argument("--dataset", default="sketchy")
    p.add_argument("--k", type=int, default=200, help="top-k (default 200)")
    args = p.parse_args()
    # Build a compatible argument namespace for eval.main
    ns = argparse.Namespace(
        config=args.config, overlay=None, checkpoint=args.checkpoint,
        dataset=args.dataset, split=None, tag=None, no_cache=False,
        restrict_gallery=False,
    )
    eval_script.main(ns)

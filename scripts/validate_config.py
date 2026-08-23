"""Static validation of the FreqDiffFormer configuration.

Asserts every manuscript-mandated value is present and correct:

  latent_dim=512, diffusion timesteps=100, beta range=1e-4..2e-2,
  batch_size=32, epochs=50, lr=1e-4, margin=0.2, lambda1=0.5, lambda2=0.1.

Usage:
  python scripts/validate_config.py --config configs/default.yaml
  python scripts/validate_config.py --config configs/default.yaml --overlay configs/ablation/no_diffusion.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

# allow running as a script from anywhere: put repo root on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config


# (dotted path, expected, human name, check_fn)
ASSERTIONS = [
    ("model.latent_dim", 512, "latent_dim", lambda a, b: a == b),
    ("model.input_size", 224, "input_size", lambda a, b: a == b),
    ("model.freq.num_tokens", 196, "freq tokens", lambda a, b: a == b),
    ("model.freq.token_dim", 256, "freq token_dim", lambda a, b: a == b),
    ("model.freq.conv_channels", [64, 128, 256, 256], "freq conv channels",
     lambda a, b: list(a) == list(b)),
    ("model.freq.strides", [2, 2, 2, 1], "freq strides", lambda a, b: list(a) == list(b)),
    ("model.freq.paddings", [1, 1, 1, 1], "freq paddings", lambda a, b: list(a) == list(b)),
    ("model.freq.norm", "instance", "InstanceNorm", lambda a, b: a == b),
    ("model.spatial.backbone", "swin_tiny_patch4_window7_224", "Swin-Tiny",
     lambda a, b: a == b),
    ("diffusion.timesteps", 100, "diffusion T=100", lambda a, b: a == b),
    ("diffusion.beta_start", 1e-4, "beta_min=1e-4", lambda a, b: abs(a - b) < 1e-12),
    ("diffusion.beta_end", 2e-2, "beta_max=2e-2", lambda a, b: abs(a - b) < 1e-9),
    ("diffusion.schedule", "linear", "linear beta schedule", lambda a, b: a == b),
    ("diffusion.objective", "noise_mse", "MSE noise objective", lambda a, b: a == b),
    ("diffusion.inference_reverse_sampling", False, "no reverse sampling at inference",
     lambda a, b: a == b),
    ("training.epochs", 50, "epochs=50", lambda a, b: a == b),
    ("training.batch_size", 32, "batch_size=32", lambda a, b: a == b),
    ("training.lr", 1e-4, "lr=1e-4", lambda a, b: abs(a - b) < 1e-12),
    ("training.optimizer", "adamw", "AdamW", lambda a, b: a == b),
    ("training.margin", 0.2, "margin=0.2", lambda a, b: abs(a - b) < 1e-9),
    ("training.lambda1_diffusion", 0.5, "lambda1=0.5", lambda a, b: abs(a - b) < 1e-9),
    ("training.lambda2_freq_align", 0.1, "lambda2=0.1", lambda a, b: abs(a - b) < 1e-9),
]


def _get(cfg, path):
    cur = cfg
    for p in path.split("."):
        cur = cur[p]
    return cur


def main(args):
    cfg = load_config(args.config, overlays=args.overlay)
    failures = []
    for path, expected, name, ok in ASSERTIONS:
        try:
            val = _get(cfg, path)
        except (KeyError, TypeError):
            failures.append(f"MISSING {name} ({path})")
            continue
        if not ok(val, expected):
            failures.append(f"WRONG {name}: got {val!r}, expected {expected!r}")
        else:
            print(f"OK   {name} = {val!r}")
    if cfg["model"]["spatial"].get("fail_on_missing", True):
        print("OK   fail_on_missing=true (no silent backbone fallback)")
    else:
        print("WARN fail_on_missing=false (backbone fallback ENABLED)")
    if failures:
        print("\nCONFIG VALIDATION FAILED:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("\nAll mandatory config values present and correct.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--overlay", nargs="*", default=None)
    args = p.parse_args()
    main(args)

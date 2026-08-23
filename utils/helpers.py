"""Deterministic / reproducibility helpers + checkpoint utilities.

Implements the manuscript's reproducibility protocol:
  - seed (Python random, NumPy, torch, CUDA)
  - cudnn deterministic / benchmark settings
  - torch deterministic algorithms (warn-only, so uncommon CPU ops don't crash)
  - CUBLAS_WORKSPACE_CONFIG for deterministic CUDA matmul (where supported)
  - record seed + config into run metadata JSON
"""

from __future__ import annotations

import json
import os
import random
import socket
import sys
import time

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> dict:
    """Seed every RNG and enable deterministic settings where appropriate."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # required for fully deterministic CUDA matmul with deterministic algos
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            # older torch signatures
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
    return dict(seed=seed, deterministic=deterministic,
                torch_version=torch.__version__, cuda=torch.cuda.is_available())


def run_metadata(cfg: dict, seed_info: dict, extra: dict | None = None) -> dict:
    """Assemble reproducibility metadata for a run."""
    meta = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "experiment_name": cfg.get("experiment", {}).get("name"),
        **seed_info,
    }
    if extra:
        meta.update(extra)
    ablation = cfg.get("training", {}).get("ablation", {})
    if ablation:
        meta["ablation"] = ablation
    return meta


def save_metadata(meta: dict, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state, path)


def config_hash(cfg: dict) -> str:
    """Short stable hash of the config used to disambiguate runs/caches."""
    import hashlib
    blob = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(blob).hexdigest()[:12]

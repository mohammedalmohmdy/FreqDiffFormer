"""Config loading and merging (base + overlays)."""

from __future__ import annotations

import copy
import os
from typing import Any

import yaml


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay into base (overlay wins)."""
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(paths: list[str] | str, overlays: list[str] | None = None) -> dict:
    """Load one or more config YAMLs by deep-merge; later files override earlier.

    Args:
        paths:    base config path(s).
        overlays: ablation/baseline/equipment config path(s) merged on top.
    """
    if isinstance(paths, str):
        paths = [paths]
    cfg: dict = {}
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    for p in overlays or []:
        with open(p, "r", encoding="utf-8") as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    return cfg

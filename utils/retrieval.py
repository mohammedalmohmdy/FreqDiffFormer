"""Retrieval evaluation for FG-SBIR.

Implements the manuscript retrieval protocol:
  - extraction of query SKETCH embeddings (freq branch + cross-domain context)
  - precomputation + offline CACHING of gallery PHOTO embeddings
  - cosine similarity ranking of gallery photos per query sketch
  - mAP@200 (primary) and Top-1 (secondary)
  - dataset-specific gallery sizes (500/400/250/400)

mAP@K: for each query, take the top-K gallery photos by cosine similarity, then
average precision over the relevant (same-class) items within those top-K ranks.
Top-1: fraction of queries whose top-1 gallery photo shares the query's class.

Per-class mAP@200 is also produced (mean over queries of each class) because the
Wilcoxon statistical analysis and Supplementary Table S1 operate on per-class
results.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


def _cache_key(checkpoint_path: str, dataset: str, cfg_hash: str) -> str:
    base = f"{dataset}__{cfg_hash}"
    if checkpoint_path and os.path.isfile(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            ck = hashlib.md5(f.read()).hexdigest()[:12]
        base += f"__ckpt{ck}"
    return base


def extract_embeddings(model, loader, device, kind: str) -> Tuple[np.ndarray, np.ndarray]:
    """Extract embeddings. kind='sketch' or 'photo'."""
    model.eval()
    embs, labs = [], []
    with torch.no_grad():
        for batch in loader:
            if kind == "sketch":
                x, label = batch
                e = model.embed_sketch(x.to(device))
            else:
                x, label = batch
                e = model.embed_photo(x.to(device))
            embs.append(e.cpu().numpy())
            labs.append(label.numpy())
    return np.concatenate(embs, axis=0), np.concatenate(labs, axis=0)


def compute_ranking(q_embs: np.ndarray, g_embs: np.ndarray) -> np.ndarray:
    """Cosine similarity ranking. Returns gallery index array (Q, G) sorted desc."""
    qn = q_embs / (np.linalg.norm(q_embs, axis=1, keepdims=True) + 1e-12)
    gn = g_embs / (np.linalg.norm(g_embs, axis=1, keepdims=True) + 1e-12)
    sims = qn @ gn.T               # (Q, G)
    return np.argsort(-sims, axis=1), sims


def average_precision_at_k(ranked_labels: np.ndarray, k: int) -> float:
    """AP@k from a single query's ranked gallery labels (1=relevant,0=not)."""
    topk = ranked_labels[:k]
    relevant = int(topk.sum())
    if relevant == 0:
        return 0.0
    precisions = []
    cum = 0
    for rank, rel in enumerate(topk, start=1):
        if rel:
            cum += 1
            precisions.append(cum / rank)
    return float(np.mean(precisions)) if precisions else 0.0


def evaluate_retrieval(q_embs: np.ndarray, q_labels: np.ndarray,
                       g_embs: np.ndarray, g_labels: np.ndarray,
                       top_k: int = 200) -> dict:
    """Compute mAP@top_k, Top-1, and per-class mAP@top_k."""
    order, sims = compute_ranking(q_embs, g_embs)
    Q = order.shape[0]
    aps = []
    top1_correct = 0
    per_class_aps: dict[int, list[float]] = {}
    for i in range(Q):
        ql = q_labels[i]
        ranked_labels = g_labels[order[i]]
        # binary relevance: 1 where the gallery class equals the query class
        relevant = (ranked_labels == ql).astype(np.int64)
        ap = average_precision_at_k(relevant, top_k)
        aps.append(ap)
        if g_labels[order[i, 0]] == ql:
            top1_correct += 1
        per_class_aps.setdefault(int(ql), []).append(ap)
    map_at_k = float(np.mean(aps))
    top1 = float(top1_correct) / Q
    per_class_map = {c: float(np.mean(v)) for c, v in per_class_aps.items()}
    return {
        "mAP@200": map_at_k,
        "top1": top1,
        "top_k": top_k,
        "num_queries": Q,
        "num_gallery": g_embs.shape[0],
        "per_class_mAP@200": per_class_map,
    }


def cache_gallery(model, gallery_loader, device, dataset: str, cfg_hash: str,
                   checkpoint_path: str, cache_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute + cache gallery photo embeddings offline (reused at query time)."""
    os.makedirs(cache_dir, exist_ok=True)
    key = _cache_key(checkpoint_path, dataset, cfg_hash)
    path = os.path.join(cache_dir, f"gallery__{key}.npz")
    if os.path.isfile(path):
        d = np.load(path)
        return d["embeddings"], d["labels"]
    g_embs, g_labels = extract_embeddings(model, gallery_loader, device, kind="photo")
    np.savez(path, embeddings=g_embs, labels=g_labels)
    return g_embs, g_labels

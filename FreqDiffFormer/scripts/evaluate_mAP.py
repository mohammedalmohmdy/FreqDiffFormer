
"""
evaluate_mAP.py

Computes retrieval metrics (mAP@K and Top-1) for sketch->image retrieval.

Usage:
    python evaluate_mAP.py --config configs/default.yaml --checkpoint outputs/checkpoint_last.pth --dataset sketchy_mini --k 200

This script:
- loads models
- extracts embeddings for gallery (images) and queries (sketches)
- computes cosine similarity rankings
- computes mAP@K and Top-1
"""
import argparse
import yaml
import torch
import numpy as np
from utils.datasets import SketchyDataset
from utils.transforms import default_transforms
from torch.utils.data import DataLoader
from models.freq_encoder import DCTFrequencyEncoder
from models.spatial_encoder import SpatialEncoder
from models.cross_domain_transformer import CrossDomainTransformer
import os
from sklearn.metrics import average_precision_score

def load_models(cfg, ckpt_path, device='cpu'):
    freq = DCTFrequencyEncoder(in_ch=1, emb_dim=cfg['model']['freq_embedding_dim'])
    spatial = SpatialEncoder(out_dim=cfg['model']['freq_embedding_dim'])
    cdt = CrossDomainTransformer(dim=cfg['model']['freq_embedding_dim'])
    # load checkpoint if present
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        st = ck.get('state_dict', {})
        # attempt to load matching keys
        try:
            freq.load_state_dict(st.get('freq', {}))
            spatial.load_state_dict(st.get('spatial', {}))
            cdt.load_state_dict(st.get('cdt', {}))
        except Exception as e:
            print("Warning loading checkpoint:", e)
    return freq.to(device), spatial.to(device), cdt.to(device)

def extract_embeddings(freq, spatial, cdt, loader, device='cpu'):
    freq.eval(); spatial.eval(); cdt.eval()
    q_embs = []
    g_embs = []
    q_labels = []
    g_labels = []
    with torch.no_grad():
        for sketch, img, label in loader:
            sketch = sketch.to(device)
            img = img.to(device)
            f = freq(sketch)    # sketch embedding
            s = spatial(img)    # image embedding
            fa, sa = cdt(f, s)
            # final embeddings: average
            emb_q = fa.cpu().numpy()
            emb_g = sa.cpu().numpy()
            q_embs.append(emb_q)
            g_embs.append(emb_g)
            q_labels.append(label.numpy())
            g_labels.append(label.numpy())
    q_embs = np.vstack(q_embs)
    g_embs = np.vstack(g_embs)
    q_labels = np.concatenate(q_labels)
    g_labels = np.concatenate(g_labels)
    return q_embs, g_embs, q_labels, g_labels

def compute_map_at_k(q_embs, g_embs, q_labels, g_labels, k=200):
    # cosine similarity
    q_norm = q_embs / np.linalg.norm(q_embs, axis=1, keepdims=True)
    g_norm = g_embs / np.linalg.norm(g_embs, axis=1, keepdims=True)
    sims = q_norm.dot(g_norm.T)  # (Q, G)
    Q = sims.shape[0]
    APs = []
    top1 = 0
    for i in range(Q):
        sim_row = sims[i]
        idx_sorted = np.argsort(-sim_row)  # descending
        topk = idx_sorted[:k]
        relevant = (g_labels[topk] == q_labels[i]).astype(int)
        # compute average precision for this query based on top-k
        if relevant.sum() == 0:
            APs.append(0.0)
        else:
            # compute precision@i for relevant positions
            precisions = []
            cumulative = 0
            for rank, rel in enumerate(relevant, start=1):
                if rel:
                    cumulative += 1
                    precisions.append(cumulative / rank)
            APs.append(np.mean(precisions) if precisions else 0.0)
        # Top-1
        if g_labels[idx_sorted[0]] == q_labels[i]:
            top1 += 1
    mAP = float(np.mean(APs))
    top1_acc = float(top1 / Q)
    return mAP, top1_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--checkpoint', default='outputs/checkpoint_last.pth')
    parser.add_argument('--dataset', default='sketchy_mini')
    parser.add_argument('--k', type=int, default=200)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds_root = os.path.join(cfg['data']['root'], args.dataset)
    ds = SketchyDataset(csv_file=os.path.join(ds_root,'split_val.csv'), root=ds_root, transform=default_transforms())
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    freq, spatial, cdt = load_models(cfg, args.checkpoint, device=device)
    q_embs, g_embs, q_labels, g_labels = extract_embeddings(freq, spatial, cdt, loader, device=device)
    mAP, top1 = compute_map_at_k(q_embs, g_embs, q_labels, g_labels, k=args.k)
    print(f"mAP@{args.k}: {mAP:.4f}, Top-1: {top1:.4f}")

if __name__=='__main__':
    main()

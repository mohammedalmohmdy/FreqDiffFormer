# FreqDiffFormer — Frequency-Guided Transformer–Diffusion for Fine-Grained Sketch-Based Image Retrieval

A reproducible, manuscript-consistent implementation of **FreqDiffFormer**:
a hybrid frequency–transformer–diffusion architecture for Fine-Grained Sketch-Based
Image Retrieval (FG-SBIR).

This repository implements the full architecture, training, evaluation, statistical
analysis, ablation, and benchmarking pipeline described in the manuscript.
Results are produced by **actual execution** — no paper numbers are hard-coded.

---

## Manuscript configuration (all in `configs/default.yaml`)

| Parameter | Value |
|---|---|
| Input resolution | 224×224 |
| Frequency representation | DCT-based |
| Frequency encoder | 4 conv blocks (2→64→128→256→256), 3×3, InstanceNorm, ReLU, pool 14×14, 196 tokens, dim 256, learnable pos-emb, LayerNorm |
| Spatial backbone | Swin Transformer-Tiny |
| Cross-domain alignment | Bidirectional cross-attention transformer (not stacked TransformerEncoder) |
| Latent dimension | 512 |
| Diffusion timesteps | T = 100 (linear β: 1e-4 → 2e-2) |
| Diffusion denoiser | Lightweight conditional latent U-Net (self + cross attention) |
| Diffusion objective | MSE on injected vs. predicted noise |
| Diffusion role | Training-time regularizer only (no reverse sampling at retrieval) |
| Triplet margin | 0.2 |
| λ1 (diffusion recon) | 0.5 (fixed) |
| λ2 (freq alignment) | 0.1 (fixed) |
| Optimizer | AdamW, lr 1e-4 |
| Batch size | 32 |
| Epochs | 50 |
| Datasets | Sketchy, ShoeV2, ChairV2, TU-Berlin |
| Primary metric | mAP@200 (+ Top-1) |
| Gallery sizes | Sketchy 500, ShoeV2 400, ChairV2 250, TU-Berlin 400 |

Run `python scripts/validate_config.py --config configs/default.yaml` to verify
all manuscript-mandated values statically.

---

## Repository structure

```
FreqDiffFormer/
├── configs/
│   ├── default.yaml              # full manuscript configuration
│   ├── ablation/                 # one overlay per ablation (Tables 6–7)
│   ├── baselines/                # baseline placeholder configs (Table 2)
│   └── lambdas_grid.yaml         # λ1/λ2 grid-search on Sketchy val
├── models/
│   ├── __init__.py               # full FreqDiffFormer model + ablation switches
│   ├── freq_encoder.py           # DCT frequency encoder (FEM)
│   ├── spatial_encoder.py        # Swin-Tiny spatial backbone
│   ├── cross_domain_transformer.py  # bidirectional cross-attention
│   ├── diffusion_fusion.py       # training-time diffusion regularizer
│   └── diffusion/
│       ├── scheduler.py          # linear β schedule (T=100)
│       ├── unet.py               # conditional latent U-Net (self+cross attn)
│       └── ddpm.py               # latent DDPM (forward loss MSE)
├── baselines/
│   ├── registry.py               # Table 2 baseline registry
│   └── *.py                      # documented placeholders (no fabricated details)
├── scripts/
│   ├── train.py                  # training (full objective + ablations)
│   ├── eval.py                   # test-split retrieval (mAP@200, Top-1, per-class)
│   ├── validate_config.py        # static config validation
│   ├── run_three.py              # 3 independent runs + mean ± SD
│   ├── grid_search_lambdas.py    # λ1/λ2 grid search (Sketchy val only)
│   ├── stats_analysis.py         # Wilcoxon signed-rank + effect size + 95% CI
│   ├── gen_table_s1.py           # Supplementary Table S1 from real per-class results
│   ├── benchmark_inference.py    # inference timing (A6000 protocol)
│   └── benchmark_training.py     # training-time diffusion overhead
├── utils/
│   ├── datasets.py               # Sketchy/ShoeV2/ChairV2/TU-Berlin loaders
│   ├── transforms.py             # 224×224 transforms
│   ├── helpers.py                # deterministic seeding + metadata
│   ├── losses.py                 # triplet + diffusion + freq-align (ablation)
│   ├── retrieval.py              # gallery cache + cosine ranking + mAP@K
│   └── config.py                 # YAML load + overlay merge
├── tests/                        # automated validation (pytest)
├── AUDIT_REPORT.md               # pre-implementation audit
├── MANUSCRIPT_REPRODUCIBILITY.md # claim → file → param → script → output
└── requirements.txt
```

---

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python scripts/validate_config.py --config configs/default.yaml
```

## Exact commands

### 1. Train a dataset
```bash
python scripts/train.py --config configs/default.yaml --dataset sketchy --run-name freqdiffformer_sketchy
```
Repeat for each dataset: `shoev2`, `chairv2`, `tu_berlin`. Checkpoints and run
metadata (seed, config hash, loss history) are saved to `outputs/runs/<name>/`.

### 2. Validation (used only for λ selection on Sketchy)
```bash
python scripts/eval.py --config configs/default.yaml \
  --checkpoint outputs/runs/freqdiffformer_sketchy/checkpoint_last.pth \
  --dataset sketchy --split val --tag val_sketchy
```

### 3. Test retrieval (main benchmark — uses the TEST split)
```bash
python scripts/eval.py --config configs/default.yaml \
  --checkpoint outputs/runs/freqdiffformer_sketchy/checkpoint_last.pth \
  --dataset sketchy --split test --restrict-gallery \
  --tag freqdiffformer_sketchy
```
Gallery photo embeddings are precomputed offline and cached under
`outputs/gallery_cache/`. Per-class mAP@200 is written to
`outputs/results/per_class_<tag>.csv` for downstream statistics.

### 4. Three independent runs (mean ± SD)
```bash
python scripts/run_three.py --config configs/default.yaml \
  --dataset sketchy --seeds 42 43 44 --tag freqdiffformer_3run
```
Aggregated mean ± SD is written to `outputs/<tag>/aggregate.json`.

### 5. λ1/λ2 grid search (Sketchy validation split only)
```bash
python scripts/grid_search_lambdas.py --config configs/default.yaml \
  --grid-config configs/lambdas_grid.yaml
```
Writes `outputs/lambda_grid/lambda_grid_sketchy.csv` (all 3×3 combinations).
The canonical fixed values (λ1=0.5, λ2=0.1) remain in `configs/default.yaml`.

### 6. Ablation experiments (Tables 6–7)
```bash
# Remove frequency encoder
python scripts/train.py --config configs/default.yaml \
  --overlay configs/ablation/no_freq_encoder.yaml --dataset sketchy \
  --run-name abl_no_freq_encoder

# Remove frequency alignment loss
python scripts/train.py --config configs/default.yaml \
  --overlay configs/ablation/no_freq_align.yaml --dataset sketchy \
  --run-name abl_no_freq_align

# Remove Cross-Domain Transformer
python scripts/train.py --config configs/default.yaml \
  --overlay configs/ablation/no_cdt.yaml --dataset sketchy --run-name abl_no_cdt

# Remove diffusion module
python scripts/train.py --config configs/default.yaml \
  --overlay configs/ablation/no_diffusion.yaml --dataset sketchy \
  --run-name abl_no_diffusion

# Remove triplet loss
python scripts/train.py --config configs/default.yaml \
  --overlay configs/ablation/no_triplet.yaml --dataset sketchy \
  --run-name abl_no_triplet

# Remove reconstruction loss
python scripts/train.py --config configs/default.yaml \
  --overlay configs/ablation/no_recon.yaml --dataset sketchy \
  --run-name abl_no_recon
```
Each ablation removes ONLY the specified component. Evaluate each with `scripts/eval.py`.

### 7. Baseline experiments (Table 2)
```bash
# List baselines and their implementation status
python baselines/registry.py
```
Baselines (Siamese CNN, DSSA, DSH, SketchyGAN, StyleMeUp, CLIP-SBIR, DiffSketch)
are currently documented **placeholders** — their architecture details are not
verified in this repository and no results are fabricated. Each has a config
under `configs/baselines/` and a documented `MISSING_INFO` gap.

### 8. Statistical analysis (Wilcoxon signed-rank)
```bash
python scripts/stats_analysis.py \
  --freq outputs/results/per_class_freqdiffformer_sketchy.csv \
  --baseline outputs/results/per_class_diffsketch_sketchy.csv \
  --dataset sketchy \
  --out outputs/stats/stats_sketchy.json
```
Reports statistic, p-value, rank-biserial effect size, 95% CI, and paired
differences. Performed independently per dataset with α = 0.01 on per-class
mAP@200 observations.

### 9. Supplementary Table S1
```bash
python scripts/gen_table_s1.py \
  --datasets sketchy shoev2 chairv2 tu_berlin \
  --freq-tag freqdiffformer \
  --diffsketch-tag diffsketch \
  --out outputs/supplementary/Table_S1.csv
```
Generated from actual per-class evaluation CSVs. If DiffSketch results are
unavailable, those entries are marked `DIFFSKETCH_RESULTS_NOT_AVAILABLE` (never
fabricated).

### 10. Computational benchmarking
```bash
# Inference timing (RTX A6000 protocol: batch 1, 50 warmup, 300 measured)
python scripts/benchmark_inference.py --config configs/default.yaml \
  --checkpoint outputs/runs/freqdiffformer_sketchy/checkpoint_last.pth \
  --dataset sketchy --gallery-size 500

# Training-time diffusion overhead
python scripts/benchmark_training.py --config configs/default.yaml
```

### 11. Run the test suite
```bash
python -m pytest tests/ -q
```
Tests validate all manuscript-mandated config values, output embedding shapes,
loss composition, ablation switches, retrieval ranking, and statistical output
structure. They run on CPU-only torch (no GPU required).

---

## Datasets

- **Sketchy** — [Sketchy Official Website](https://sketchx.eecs.qmul.ac.uk/downloads/) · [Google Drive](https://drive.google.com/file/d/11GAr0jrtowTnR3otyQbNMSLPeHyvecdP/view)
- **ShoeV2 / ChairV2** — [Sketchy Official Website](https://sketchx.eecs.qmul.ac.uk/downloads/) · [Google Drive](https://drive.google.com/file/d/1frltfiEd9ymnODZFHYrbg741kfys1rq1/view)
- **TU-Berlin** — [TU-Berlin Official Website](https://www.tu-berlin.de/) · [Google Drive](https://drive.google.com/file/d/12VV40j5Nf4hNBfFy0AhYEtql1OjwXAUC/view)

Each dataset expects split files under `<root>/splits/`:
`train.csv`, `val.csv`, `test.csv`, `query.csv`, `gallery.csv`. Paths and root
directories are configurable in `configs/default.yaml -> data.datasets`.

---

## Reproducibility notes

- Deterministic seeding (Python, NumPy, torch, CUDA) is enforced by
  `utils/helpers.set_seed`. Each run records seed + config hash to a metadata JSON.
- Gallery embeddings are cached offline and reused across queries.
- Inference performs **zero** iterative reverse-diffusion steps.
- `λ1` and `λ2` are fixed (not learnable); the grid-search script evaluates the
  full manuscript grid on the Sketchy **validation** split and never uses the
  test set for selection.
- No results from Tables 5–12 or Supplementary Table S1 are hard-coded.

---

## Citation

```bibtex
@article{almohamadi2025freqdiffformer,
  title={FREQDIFFFORMER: Frequency-Guided Transformer--Diffusion Framework for
         Fine-Grained Sketch-Based Image Retrieval},
  author={Al-Mohamadi, Mohammed A. S. and Prabhakar, C. J.},
  journal={Multimedia Tools and Applications},
  year={2025}
}
```

Contact: almohmdy30@gmail.com · GitHub: https://github.com/mohammedalmohmdy

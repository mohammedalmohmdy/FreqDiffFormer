# FreqDiffFormer — Frequency-Guided Transformer–Diffusion for FG-SBIR

This repository contains a reproducible implementation skeleton of **FreqDiffFormer**:
a hybrid frequency–transformer–diffusion architecture for Fine-Grained Sketch-Based Image Retrieval (FG-SBIR).


---

## Repository structure

```
FreqDiffFormer/
├── configs/
│   └── default.yaml
├── data/
│   └── README.md
├── models/
│   ├── __init__.py
│   ├── freq_encoder.py
│   ├── spatial_encoder.py
│   ├── cross_domain_transformer.py
│   └── diffusion_fusion.py
├── scripts/
│   ├── train.py
│   └── eval.py
├── utils/
│   ├── datasets.py
│   ├── transforms.py
│   └── helpers.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Quick start (example)

1. Create a Python environment (Python 3.9+):
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Prepare your FG-SBIR datasets (Sketchy / TU-Berlin / ShoeV2 / ChairV2) under `data/` following `data/README.md`.

3. Train:
```bash
python scripts/train.py --config configs/default.yaml
```

4. Evaluate:
```bash
python scripts/eval.py --checkpoint outputs/checkpoint_best.pth --dataset sketchy
```





# FreqDiffFormer — Frequency-Guided Transformer–Diffusion for FG-SBIR

This repository contains a reproducible implementation skeleton of **FreqDiffFormer**:
a hybrid frequency–transformer–diffusion architecture for Fine-Grained Sketch-Based Image Retrieval (FG-SBIR).

**Note:** This is a full and realistic codebase skeleton (models, training, evaluation, config, and utilities).
You must provide datasets, pretrained weights (optional), and tune hyperparameters for state-of-the-art results.

**Paper:** FreqDiffFormer (see accompanying manuscript). fileciteturn1file0

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

---

## License
MIT — see LICENSE file.

---

If you want, I can also push this repository to a **new GitHub repository** for you (I will provide the commands and a git script).

## Notes & Reproduction Checklist

See `NOTES.md` for detailed developer notes and reproduction tips.

## Cover Letter & Suggested Reviewers
- `COVER_LETTER_ESWA.md`
- `SUGGESTED_REVIEWERS.md`

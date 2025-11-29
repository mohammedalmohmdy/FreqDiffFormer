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



Datasets
ShoeV2 / ChairV2
Sketchy Official Website
Google Drive Download

Sketchy
Sketchy Official Website
Google Drive Download

TU-Berlin
TU-Berlin Official Website
Google Drive Download

### 📂 Datasets

- **ShoeV2 / ChairV2**  
  [Sketchy Official Website](https://sketchx.eecs.qmul.ac.uk/downloads/)  
  [Google Drive Download](https://drive.google.com/file/d/1frltfiEd9ymnODZFHYrbg741kfys1rq1/view)

- **Sketchy**  
  [Sketchy Official Website](https://sketchx.eecs.qmul.ac.uk/downloads/)  
  [Google Drive Download](https://drive.google.com/file/d/11GAr0jrtowTnR3otyQbNMSLPeHyvecdP/view)

- **TU-Berlin**  
  [TU-Berlin Official Website](https://www.tu-berlin.de/)  
  [Google Drive Download](https://drive.google.com/file/d/12VV40j5Nf4hNBfFy0AhYEtql1OjwXAUC/view)


  Citation: If you use this code, please cite:

title = {FREQDIFFFORMER: FREQUENCY-GUIDED TRANSFORMER–DIFFUSION FRAMEWORK FOR FINE-GRAINED SKETCH-BASED IMAGE RETRIEVAL},

author = {Mohammed A. S. Al-Mohamadi and Prabhakar C. J.},

journal = {pattern analysis and applications}, year = {2025} }

License: This project is released under the MIT License.

Contact: almohmdy30@gmail.com GitHub: https://github.com/mohammedalmohmdy

# NOTES — Developer & Reproducibility

This file documents important developer notes, reproduction tips, and items reviewers often check.

## Important clarifications (do NOT remove)
- **Frequency processing:** The repository uses **FFT magnitude for visualization** inside `models/freq_encoder.py` for clarity. The paper uses **DCT-based encoding** for the Frequency Encoder (FEM) because of DCT's orthogonality and energy compaction. For exact reproduction, replace the FFT block with a DCT implementation (e.g. `scipy.fftpack.dct` or custom CUDA kernel).
- **Runtime claims:** The manuscript states **near real-time** inference (e.g., 9.4 ms per image on RTX A6000). Reported numbers depend on hardware and batch settings; we label them **near real-time** in the paper and README.
- **Diffusion module:** The included `SimpleLatentUNet` is a compact placeholder. For full fidelity to the paper, implement a timestep-conditioned DDPM in `models/diffusion_fusion.py`.
- **Data splits:** Provide CSV files `split_train.csv` and `split_val.csv` with `image_path,sketch_path,label` columns in each dataset folder under `data/`.

## Reproduction checklist (recommended before submission)
- [ ] Place datasets under `data/` following `data/README.md`.
- [ ] Install packages: `pip install -r requirements.txt`.
- [ ] Replace the diffusion placeholder with a full conditional DDPM if required.
- [ ] Set `configs/default.yaml` hyperparameters as in the manuscript.
- [ ] Train on at least one dataset and verify output shapes.

## Contact
If reviewers face issues, provide this repository link and contact: almohmdy30@gmail.com

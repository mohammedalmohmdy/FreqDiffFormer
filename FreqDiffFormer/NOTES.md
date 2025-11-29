# NOTES — Developer & Reproducibility

This file documents important developer notes, reproduction tips, and items reviewers often check.

## Important clarifications 
- **Frequency processing:** The repository uses **FFT magnitude for visualization** inside `models/freq_encoder.py` for clarity. The paper uses **DCT-based encoding** for the Frequency Encoder (FEM) because of DCT's orthogonality and energy compaction. For exact reproduction, replace the FFT block with a DCT implementation (e.g. `scipy.fftpack.dct` or custom CUDA kernel).
- **Runtime claims:** The manuscript states **near real-time** inference (e.g., 9.4 ms per image on RTX A6000). Reported numbers depend on hardware and batch settings; we label them **near real-time** in the paper and README.
- **Diffusion module:** The included `SimpleLatentUNet` is a compact placeholder. For full fidelity to the paper, implement a timestep-conditioned DDPM in `models/diffusion_fusion.py`.
- **Data splits:** Provide CSV files `split_train.csv` and `split_val.csv` with `image_path,sketch_path,label` columns in each dataset folder under `data/`.



## Contact
If reviewers face issues, provide this repository link and contact: almohmdy30@gmail.com

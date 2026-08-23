# NOTES — Developer & Reproducibility

## Implementation status (post-audit)

This repository is now a **full, manuscript-consistent implementation** of
FreqDiffFormer — not a skeleton.

### Key decisions and clarifications

- **Frequency processing:** Uses a **differentiable, pure-torch 2D DCT-II**
  (matrix-based, orthonormal) so gradients flow through the frequency transform.
  No scipy/numpy detour for the forward pass. The DCT origin is at index [0,0];
  radial masks separate low/high-frequency bands correctly.
- **Frequency encoder:** Two-channel (low + high) DCT structural representation,
  frequency masking, sign-log compression, robust min-max (median/MAD), four
  InstanceNorm conv blocks, 14×14 adaptive pool → 196 tokens, learnable
  positional embedding, LayerNorm.
- **Cross-Domain Transformer:** Genuine **bidirectional cross-attention**
  (frequency attends to spatial AND spatial attends to frequency), with
  learnable domain embeddings. NOT a stacked TransformerEncoder.
- **Diffusion denoiser:** Lightweight **conditional latent U-Net** with encoder-
  decoder skips, FiLM timestep conditioning, bottleneck self-attention, and
  cross-attention over the cross-domain fused tokens at every block.
- **Diffusion at inference:** The retrieval path uses
  `DiffusionRegularizer.inference_latent()` which is the identity — **no
  iterative reverse diffusion**. Reverse `sample()` exists for research only.
- **Retrieval embeddings:** Query (sketch) and gallery (photo) embeddings are
  computed separately via `embed_sketch` / `embed_photo`. The absent modality is
  fed as a zero-context token so cross-attention over zero contributes ~0 and the
  residual path yields each domain's aligned embedding.
- **Runtime claims:** The manuscript reports near real-time inference (e.g.
  9.4 ms/image on RTX A6000). Our benchmark script implements the exact protocol
  (batch 1, 50 warmup, 300 measured, CUDA sync). On CPU-only environments it
  runs the same protocol and labels results as CPU — it does NOT hard-code or
  claim the A6000 number.
- **Baselines:** All seven baselines are documented **placeholders** — their
  architecture details are not verified in this repository and no results are
  fabricated. Each has a config and a documented `MISSING_INFO` gap. The
  registry (`baselines/registry.py`) flags `implemented=False` for all.
- **λ selection:** The grid-search script evaluates the full manuscript grid on
  the Sketchy **validation** split. Canonical defaults (0.5 / 0.1) are fixed in
  `configs/default.yaml` and are **not** overwritten by the search.
- **Data splits:** Split files are configurable per dataset (`data.datasets.*`).
  The repository does not alter splits to improve results. Main benchmark
  evaluation uses the **test** split; validation is used only for λ selection.

### What the audit could NOT implement (documented placeholders)

1. Baseline architecture details (Siamese CNN / DSSA / DSH / SketchyGAN /
   StyleMeUp / CLIP-SBIR / DiffSketch) — not present in the repo or brief.
2. DiffSketch per-class results for Supplementary Table S1 — the S1 generator
   leaves the DiffSketch column empty with `DIFFSKETCH_RESULTS_NOT_AVAILABLE`
   until real results are provided.
3. RTX A6000 timing — verification environment is CPU-only; the benchmark script
   runs the exact protocol and labels the actual device used.

## Contact
If reviewers face issues, provide this repository link and contact: almohmdy30@gmail.com

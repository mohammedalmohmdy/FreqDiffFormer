# AUDIT_REPORT — FreqDiffFormer Repository (Pre-Implementation)

This audit was produced **before** modifying any source code, as required by the task brief.
It is machine-readable (fixed-key records) and documents, for every code component:

`code component → current implementation → manuscript requirement → gap → planned fix`.

A final `REPRODUCIBILITY_AUDIT.md` (PASS / PARTIAL / BLOCKED) is produced **after** implementation.

---

## 0. Environment observed during audit

- Python interpreters: `C:\Python314\python.exe` (3.14.6, **no** torch/timm), `C:\Program Files\Python312\python.exe` (3.12.4).
- Verification interpreter used: **Python 3.12** with `torch 2.2.1+cpu` (CUDA unavailable), `timm 1.0.7`, `scipy 1.14.1`, `scikit-learn 1.6.1`, `PyYAML`, `tqdm`, `matplotlib`, `opencv-python`, `einops 0.8.2`, `pytest 9.1.1`.
- Implication: code must run on CPU for verification and **never assume CUDA is present**; CUDA-specific paths (deterministic algos, timing protocol) must be guarded so the pipeline still executes (and tests still pass) without a GPU, while strictly using CUDA sync where CUDA is available.

---

## 1. Component-level records

Each record: `COMPONENT | CURRENT | MANUSCRIPT_REQ | GAP | PLANNED_FIX`.
`STATUS_AFTER` is filled in `REPRODUCIBILITY_AUDIT.md`.

```yaml
records:
  - component: configs/default.yaml
    current: "epochs=2, batch_size=2, lr=1e-4, lambda_ddpm=0.1, latent_dim=256, ddpm_timesteps=10, dataset=sketchy_mini, output_dir=./outputs"
    manuscript_req: "epochs=50, batch_size=32, lr=1e-4, latent_dim=512, T=100, beta_min=1e-4, beta_max=2e-2, margin=0.2, lambda1=0.5 (diffusion recon), lambda2=0.1 (freq alignment), input 224x224, datasets Sketchy/ShoeV2/ChairV2/TU-Berlin, mAP@200 + Top-1, gallery sizes 500/400/250/400, AdamW"
    gap: "All core hyperparameters wrong/missing; lambdas and beta schedule not exposed; dataset/gallery/timing/seed-protocol not configurable."
    planned_fix: "Replace with a full config exposing every manuscript parameter; add layered configs (ablation, baseline, datasets, lambda-grid, timing). Add a config validator script."
    status_after: ""

  - component: models/freq_encoder.py (DCTFrequencyEncoder)
    current: "DCT-II (scipy or FFT fallback); single static high-freq mask keeping top-left k_h x k_w block; 2 Conv2d(3x3) with BatchNorm2d + ReLU; AdaptiveAvgPool2d(1,1) -> Linear to emb_dim. Output (B, emb_dim)."
    manuscript_req: "DCT-based frequency representation; low/high-frequency structural info, frequency masking, sign-logarithmic compression, robust min-max scaling; 4 conv blocks channels 2->64->128->256->256, 3x3 kernels; blocks 1-3 stride2 pad1, block4 stride1 pad1; InstanceNorm; ReLU; adaptive avg pool to 14x14; 196 frequency tokens; token dim 256; learnable positional embedding; LayerNorm."
    gap: "Mask is single and in wrong corner (DCT-II origin is index [0,0]); no low/high structural split; no sign-log compression; no robust min-max; only 2 conv blocks (need 4); BatchNorm instead of InstanceNorm; global 1x1 pooling instead of 14x14; no tokenization/positional embedding/LayerNorm; input channel expects 1 but manuscript channels are 2->64 implying a 2-channel frequency input."
    planned_fix: "Rewrite as FreqEncoder: build a 2-channel DCT structural representation (low-freq + high-freq) from a grayscale sketch; apply frequency masks, sign(x)*log(1+|x|) compression and robust min-max (subtract median, divide by robust std via MAD); 4 conv blocks (InstanceNorm2d(affine=False) + ReLU), channels 2->64->128->256->256 with stated strides/padding; AdaptiveAvgPool2d(14); flatten to (B,196,256) tokens; learnable (1,196,256) positional embedding add; LayerNorm(256). Output tokens (B,196,256)."
    status_after: ""

  - component: models/spatial_encoder.py (SpatialEncoder)
    current: "timm swin_tiny_patch4_window7_224 features_only -> AdaptiveAvgPool2d(1,1) -> Linear(feat_dim, out_dim). Falls back to a 2-conv CNN if timm missing."
    manuscript_req: "Spatial backbone: Swin Transformer-Tiny; input 224x224; latent dim 512 downstream."
    gap: "Backbone name is correct but: silent CNN fallback (brief forbids silently substituting a lightweight architecture where manuscript requires Swin); must fail loudly if timm/Swin unavailable; must emit patch tokens (not pooled vec) so the Cross-Domain Transformer can do real cross-attention; output projection target should be the latent/token dim."
    planned_fix: "Use timm Swin-Tiny; fail loudly with a clear error if timm or the model is unavailable (no silent fallback). Expose token feature map and project to a configurable token dim (default 256) matching freq tokens; also keep a pooled projection path for the latent. Document pretrained usage."
    status_after: ""

  - component: models/cross_domain_transformer.py (CrossDomainTransformer)
    current: "nn.TransformerEncoder over torch.stack([sketch,img],dim=0); output (out[0],out[1]). No real cross-attention."
    manuscript_req: "Cross-domain Transformer alignment must implement genuine cross-domain/cross-attention interaction, not merely stacking two embeddings and applying a generic TransformerEncoder."
    gap: "This is exactly the forbidden pattern (stack + TransformerEncoder). No query/key/value cross-attention between the two domains."
    planned_fix: "Replace with CrossDomainTransformer: two transformer layers where (a) frequency tokens attend to spatial tokens via MultiheadAttention(Q=freq,K=V=spatial) and (b) spatial tokens attend to frequency tokens, i.e. bidirectional cross-attention (a-la cross-encoder). Provide learnable domain embeddings + positional info. Output aligned fused tokens; pool to latent_dim(512) for downstream heads."
    status_after: ""

  - component: models/diffusion/scheduler.py (LinearBetaSchedule)
    current: "linspace(beta_start,beta_end,timesteps); alpha=1-beta; alpha_bar=cumprod(alpha). Constants only."
    manuscript_req: "Linear beta schedule, beta_min=1e-4, beta_max=2e-2, T=100."
    gap: "Defaults are correct but not enforced to manuscript values; not exposed via config; quantities not buffer-registered for device correctness."
    planned_fix: "Hard-default to (beta_start=1e-4,beta_end=2e-2,timesteps=100) but read from config; store as registered buffers; expose getters. Validator asserts range."
    status_after: ""

  - component: models/diffusion/unet.py (LatentUNet)
    current: "MLP: Linear->ReLU->Linear->ReLU->Linear with timestep sinusoidal embedding added once. Name says U-Net but is not a U-Net, no self/cross-attention."
    manuscript_req: "Lightweight conditional latent U-Net denoiser; must include self-attention and cross-attention as described; timestep conditioning."
    gap: "It is an MLP, not a U-Net; no self-attention; no cross-attention; conditioning is a single biased add."
    planned_fix: "Implement ConditionalLatentUNet operating on a token/latent grid: sinusoidal timestep MLP conditioning via layernorm-modulation (FiLM-style); a small encoder-decoder (U-Net) with residual blocks, a self-attention at the bottleneck and cross-attention over a conditioning sequence (the cross-domain fused tokens); skip connections between encoder and decoder mirror blocks (linear on the 1D latent). Output shape = input latent shape (B, latent_dim)."
    status_after: ""

  - component: models/diffusion/ddpm.py (LatentDDPM)
    current: "forward_loss samples t with torch.randint, q_sample, MSE(pred_noise,noise). sample() performs full reverse diffusion loop (T steps). timesteps default 50."
    manuscript_req: "T=100; uniform t sampling during training; objective MSE between injected Gaussian noise and predicted noise; diffusion is a TRAINING-TIME regularizer; NO iterative reverse diffusion at retrieval inference; lambda1=0.5 (fixed, not learnable)."
    gap: "Default timesteps 50 not 100; reverse sampling present and reachable from inference (must be excluded from retrieval path); lambdas not owned here (in loss module)."
    planned_fix: "Default timesteps=100, beta 1e-4..2e-2 (from config). forward_loss unchanged in spirit (uniform t, MSE on noise). Keep sample() but mark it RESEARCH-ONLY and ensure the retrieval/inference code path NEVER calls it. lambda1/lambda2 live in utils/losses.py per brief."
    status_after: ""

  - component: models/diffusion_fusion.py (DiffusionLatentFusion)
    current: "wraps LatentDDPM; forward() is identity (returns z); refine() calls ddpm.sample()."
    manuscript_req: "Diffusion latent fusion/training pathway used as a TRAINING-TIME regularizer; inference must NOT perform iterative reverse diffusion."
    gap: "Identity forward and a refine() that does reverse sampling is not a real fusion pathway; no clear training/inference separation."
    planned_fix: "Replace with DiffusionRegularizer: training-time computes diffusion recon loss on the fused latent (forward_loss); provide a no-op/no-sampling inference projection. Ensure eval/inference uses the deterministic fused latent directly. Document that reverse sampling is intentionally not used for retrieval."
    status_after: ""

  - component: utils/datasets.py (SketchyDataset)
    current: "Single SketchyDataset reading CSV columns image_path,sketch_path,label; assumes one shared layout; no ShoeV2/ChairV2/TU-Berlin; no train/val/test split distinction; no separate query/gallery."
    manuscript_req: "Dataset classes for Sketchy, ShoeV2, ChairV2, TU-Berlin; configurable paths and split files; explicit train/val/test handling; retrieval query-sketch -> photo-gallery."
    gap: "Only one dataset; one layout; no split files; gallery/query not separated."
    planned_fix: "Add a registry of dataset configs (root, split_dir, photo_root, sketch_root, gallery_size, num_classes). Implement FGSBIRDataset base + per-dataset adapters; query-only (sketch) and gallery-only (photo) datasets for retrieval; train/val/test split loaded from configurable CSV/text files. Do not hard-code paths."
    status_after: ""

  - component: utils/transforms.py
    current: "Resize(224), ToTensor, ImageNet-normalize for photo; grayscale for sketch + Normalize(0.5,0.5)."
    manuscript_req: "Input resolution 224x224."
    gap: "Resolution is correct; frequency transform not included (sketch for DCT path should be 1-channel). OK otherwise."
    planned_fix: "Keep 224 resize for both sketches and photos; provide separate transforms for the frequency (sketch, 1ch) and spatial (photo, 3ch) branches; make resolution configurable."
    status_after: ""

  - component: utils/helpers.py (set_seed)
    current: "random, np, torch.manual_seed, cuda.manual_seed_all. save_checkpoint."
    manuscript_req: "Seed (python/random/numpy/torch/CUDA), deterministic settings where appropriate, record seed in experiment metadata."
    gap: "No cudnn deterministic/benchmark flags; no metadata recording; no deterministic-algo setting (where supported)."
    planned_fix: "set_seed(seed, deterministic=True): set all RNGs; torch.use_deterministic_algorithms(True, warn_only=True); cudnn.deterministic=True; cudnn.benchmark=False; CUBLAS_WORKSPACE_CONFIG env; return/record seed+git+config to a run metadata JSON."
    status_after: ""

  - component: utils/losses (does not exist)
    current: "No dedicated loss module. train.py builds triplet in-batch manually and adds lambda_ddpm*ddpm_loss; no frequency-alignment loss; no ablation switches; uses 'recon_loss' variable name for triplet (misleading)."
    manuscript_req: "L_total = L_triplet + lambda1*L_diffusion + lambda2*L_frequency_alignment; implement each independently; ablation switches (no freq align, no diffusion, no triplet, no recon); margin 0.2; lambda1=0.5 (diffusion recon) fixed; lambda2=0.1 (freq alignment) fixed; frequency alignment loss exactly as described."
    planned_fix: "Create utils/losses.py with TripletLoss(margin=0.2, in-batch semi-hard mining), DiffusionReconstructionLoss (wraps LatentDDPM.forward_loss, weight lambda1), FrequencyAlignmentLoss (weight lambda2) computing alignment between the frequency branch and the spatial/cross-domain output per the manuscript description. Provide a CompositeLoss that toggles components via ablation flags. Each component loss is independently computable."
    status_after: ""

  - component: scripts/train.py
    current: "Builds freq/spatial/cdt/diffusion; trains on sketchy_mini split_train.csv; in-batch triplet; loss = recon + lambda_ddpm*ddpm; saves checkpoint_last. No val, no ablation switches, no metadata, no per-run output dir, no save seed/run metadata."
    manuscript_req: "AdamW, lr=1e-4, batch=32, epochs=50; full objective; lambda grid-search (Sketchy val) selectable; ablation switches; per-run save; record metadata; deterministic."
    gap: "Missing validation, ablation switches, metadata, three-run support, correct objective, correct defaults."
    planned_fix: "Rewrite as config-driven trainer: build full model + CompositeLoss with ablation toggles from config; AdamW; uniform-batch training; per-epoch optional validation on Sketchy val (used only for lambda selection); checkpoint + metadata (seed, config hash, components on/off) into outputs/<run_name>; CLI for dataset, config, run index."
    status_after: ""

  - component: scripts/eval.py
    current: "Loads freq/spatial/cdt; extracts (fa+sa)/2 embeddings over the val split (paired sketch+photo per row); prints embedding shape only. Not retrieval."
    manuscript_req: "Test-split retrieval: query sketch embeddings + precomputed offline gallery photo embeddings; cosine similarity; rank gallery; mAP@200 + Top-1; dataset-specific gallery sizes; cache+reuse gallery embeddings."
    gap: "This is paired-embedding extraction, not retrieval; uses val not test; no gallery precompute/cache; no ranking/metrics."
    planned_fix: "Rewrite as retrieval evaluator on the test split: build a QueryDataset(sketches) and GalleryDataset(photos); precompute + cache gallery photo embeddings to disk (keyed by checkpoint+dataset); compute query sketch embeddings; cosine similarity; rank at dataset gallery size; mAP@200 and Top-1; cache reused if checkpoint unchanged."
    status_after: ""

  - component: scripts/evaluate_mAP.py
    current: "Paired extraction=correct for a paired setting but: query and gallery are the SAME batch rows (q_labels == g_labels == label), gallery size == batch labels, not the manuscript gallery; uses val split; k=200 default. So ranking is degenerate (each sketch's own photo in same batch)."
    manuscript_req: "query->gallery ranking over real gallery; dataset gallery sizes (500/400/250/400)."
    gap: "Same as eval.py: not real retrieval; no gallery precompute."
    planned_fix: "Replace/consolidate into the new retrieval evaluator (scripts/eval.py) with proper gallery. Provide backward-compat thin shim that errors clearly if used in the legacy paired mode."
    status_after: ""

  - component: scripts (missing)
    current: "Only train.py, eval.py, evaluate_mAP.py exist."
    manuscript_req: "Validation; three-run aggregation; lambda grid search (Sketchy val); Wilcoxon stats (statistic, p, effect size, 95% CI, paired diffs); Supplementary Table S1 generation from actual per-class results; computational inference benchmark (A6000, batch1, 50 warmup, 300 measured, CUDA sync, feature+rank+total latency); training-time diffusion-overhead benchmark; config validation."
    gap: "All missing."
    planned_fix: "Add scripts/validate_config.py; run_three.py; grid_search_lambdas.py; stats_analysis.py; gen_table_s1.py; benchmark_inference.py; benchmark_training.py. Each consumes/produces real artifacts."
    status_after: ""

  - component: tests/ (test_ddpm.py, test_freq_encoder.py)
    current: "Two ad-hoc print-based scripts; no pytest assertions; check only ddpm loss/sample and freq encoder old shape."
    manuscript_req: "Automated validation tests for: latent_dim=512, T=100, beta range, batch_size=32, epochs=50, lr=1e-4, margin=0.2, lambda1=0.5, lambda2=0.1, output embedding shapes, retrieval ranking, loss composition, ablation switches, statistical output structure."
    gap: "No assertions; missing the listed checks."
    planned_fix: "Replace with pytest test modules under tests/: test_config.py, test_freq_encoder.py, test_cross_domain.py, test_diffusion.py, test_losses.py, test_retrieval.py, test_stats.py. Use tiny synthetic tensors (CPU) and strict asserts."
    status_after: ""

  - component: baselines/ (does not exist)
    current: "No baseline implementations."
    manuscript_req: "Baselines: Siamese CNN, DSSA, DSH, SketchyGAN, StyleMeUp, CLIP-SBIR, DiffSketch for Table 2; do NOT invent architecture details; where details unavailable, create explicit configuration placeholders + document missing info rather than fabricating; do not claim a baseline is implemented unless its code exists; baseline experiment registry for Table 2 settings."
    gap: "Entirely missing."
    planned_fix: "Add baselines/registry.py (programmatic registry of Table 2 settings: name, dataset, gallery size, metric, citation, config placeholder) and per-baseline placeholder modules with explicit MISSING_INFO notes. Where a baseline is a known published architecture we can sketch, provide a minimal interface but clearly flag internals as not-verified. No fabricated results."
    status_after: ""

  - component: README.md
    current: "Describes the repo as a 'reproducible implementation skeleton'; quick-start for train/eval only; dataset download links."
    manuscript_req: "Describe the ACTUAL implemented repository; exact commands for training/validation/test/three-runs/baselines/ablations/stats/benchmarks/S1."
    gap: "Calls itself a skeleton; missing all the new commands."
    planned_fix: "Rewrite README to describe the real pipeline + exact commands + dataset layout + config structure + reproducibility notes. Keep citation block + dataset download links."
    status_after: ""

  - component: MANUSCRIPT_REPRODUCIBILITY.md (does not exist)
    current: "Missing."
    manuscript_req: "Machine-readable checklist mapping manuscript claim -> source file -> config param -> experiment script -> expected output/table."
    gap: "Missing."
    planned_fix: "Add a checklist table covering each manuscript claim (architecture, losses, scheduling, datasets, metrics, stats, timing, ablations, baselines, S1)."
    status_after: ""
```

---

## 2. Manuscript requirements that CANNOT be implemented from currently-available information

These will be implemented as **explicit placeholders / configuration flags** rather than fabricated scientific content, per the scientific-safety rules:

```yaml
unverifiable_or_placeholder:
  - claim: "Hyperparameter selection grid (lambda1 in {0.1,0.5,1.0}, lambda2 in {0.01,0.1,0.5}) selected on Sketchy validation split."
    block: "Selection RESULTS are not re-implemented (correctly: never fabricated). The grid-search MECHANISM is fully implementable and runs on the Sketchy val split; selected values are stored in a results file, not hard-coded as 'the chosen ones'."
    resolution: "Implement scripts/grid_search_lambdas.py that evaluates the full 3x3 grid on Sketchy val and writes a CSV of (lambda1,lambda2,val_metric). The fixed defaults lambda1=0.5,lambda2=0.1 remain the canonical config."
  - claim: "Seven baselines (Siamese CNN, DSSA, DSH, SketchyGAN, StyleMeUp, CLIP-SBIR, DiffSketch) with their reported Table 2 numbers."
    block: "Exact architecture details for DSSA/DSH/SketchyGAN/StyleMeUp/CLIP-SBIR/DiffSketch are not provided in this repository nor in the task brief. Reported numbers must NOT be hard-coded."
    resolution: "Provide baselines/registry.py + per-baseline placeholder modules with explicit MISSING_INFO fields and a 'configured but not verified' status flag. No baseline is claimed implemented unless its code exists; placeholder code = interface + registry entry, not results."
  - claim: "Swin Transformer-Tiny pretrained weights for the spatial backbone."
    block: "Offline pretraining weights require download (internet may be unavailable)."
    resolution: "Use timm's pretrained flag; if download unavailable, fail loudly with a clear error and an instruction to obtain weights or set spatial.pretrained=false. No silent CNN fallback in the final pipeline."
  - claim: "Per-class mAP@200 for DiffSketch (for Supplementary Table S1 paired differences)."
    block: "DiffSketch per-class results are not present in the repo."
    resolution: "scripts/gen_table_s1.py reads BOTH FreqDiffFormer and DiffSketch per-class CSV files emitted by real evaluation runs; if the DiffSketch CSV is absent it produces S1 with the DiffSketch column empty and a clear 'DIFFSKETCH RESULTS NOT AVAILABLE' status rather than fabricating values."

  - claim: "RTX A6000 inference timing (9.4 ms/image reported)."
    block: "Verification environment is CPU-only (no CUDA, no A6000)."
    resolution: "scripts/benchmark_inference.py implements the exact protocol (batch1, 50 warmup, 300 measured, torch.cuda.synchronize where CUDA available, feature/rank/total latency). On CPU it runs the same protocol and clearly labels results as CPU (not A6000). It does NOT hard-code or claim the A6000 number."
```

---

## 3. Scientific-safety guardrails adopted for implementation

```yaml
guardrails:
  - "No value from Tables 5-12 is hard-coded anywhere; all results come from real execution."
  - "Supplementary Table S1 values are generated from real per-class CSVs, never hard-coded."
  - "Dataset splits are not altered to improve results; splits come from configurable split files."
  - "Baselines are never tuned with test-set information; any baseline placeholder explicitly forbids test-set tuning."
  - "lambda1/lambda2 selection uses ONLY the Sketchy validation split."
  - "A baseline module is marked implemented only when non-placeholder code exists."
  - "No silent lightweight architecture substitution: Swin is required; missing timm/Swin raises a clear error."
  - "Missing optional dependencies raise a clear error instead of silent fallback."
  - "Diffusion reverse sampling is unreachable from the retrieval/inference path."
  - "Manuscript scientific terminology is preserved verbatim in code/config names."
```

---

End of pre-implementation audit. Implementation proceeds against this plan.

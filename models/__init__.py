"""FreqDiffFormer full model with ablation switches.

Wires together:
  - Frequency encoder  (FEM)             -> frequency tokens
  - Spatial encoder    (Swin-Tiny)        -> spatial tokens
  - Cross-Domain Transformer (CDT)        -> cross-domain aligned tokens + latent
  - Diffusion regularizer                 -> training-only denoising loss

Retrieval embedding protocol (defined here, used by scripts/eval.py):
  The Cross-Domain Transformer is genuinely bidirectional and requires both
  modalities. For standalone query (sketch) / gallery (photo) embedding we feed
  the absent modality as a zero token context, so cross-attention over a zero
  context contributes ~0 and the residual path yields each domain's aligned
  pooled tokens. The shared `to_latent` head maps to the 512-D latent used for
  cosine ranking. This is an inference embedding choice (documented); it does
  NOT modify any manuscript claim and adds no fabricated learnable parameters.

Ablation switches (from config -> training.ablation):
  use_freq_encoder: if False, the FEM is replaced by a learnable non-frequency
                    token placeholder (architecture stays structurally intact).
  use_cdt:          if False, the CDT is bypassed; freq and spatial tokens are
                    pooled independently and projected to the latent.
  use_diffusion, use_freq_align, use_triplet, use_recon: handled by the loss
                    module (utils/losses.py), not here.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .freq_encoder import FreqEncoder
from .spatial_encoder import SpatialEncoder
from .cross_domain_transformer import CrossDomainTransformer
from .diffusion_fusion import DiffusionRegularizer


class FreqDiffFormer(nn.Module):
    def __init__(
        self,
        freq_cfg: dict,
        spatial_cfg: dict,
        cdt_cfg: dict,
        diffusion_cfg: dict,
        latent_dim: int = 512,
        input_size: int = 224,
        ablation: dict | None = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size
        abl = ablation or {}
        self.use_freq_encoder = abl.get("use_freq_encoder", True)
        self.use_cdt = abl.get("use_cdt", True)
        self.use_diffusion = abl.get("use_diffusion", True)

        self.freq_encoder = FreqEncoder(
            in_channels=freq_cfg["in_channels"],
            token_dim=freq_cfg["token_dim"],
            num_tokens=freq_cfg["num_tokens"],
            token_grid=freq_cfg["token_grid"],
            conv_channels=freq_cfg["conv_channels"],
            kernel_size=freq_cfg["kernel_size"],
            strides=freq_cfg["strides"],
            paddings=freq_cfg["paddings"],
            norm=freq_cfg["norm"],
            activation=freq_cfg["activation"],
            use_sign_log=freq_cfg["use_sign_log"],
            use_robust_minmax=freq_cfg["use_robust_minmax"],
            low_freq_mask_ratio=freq_cfg["low_freq_mask_ratio"],
            high_freq_mask_ratio=freq_cfg["high_freq_mask_ratio"],
            positional_embedding=freq_cfg["positional_embedding"],
            layer_norm=freq_cfg["layer_norm"],
            input_size=input_size,
        )
        token_dim = freq_cfg["token_dim"]

        # Ablation placeholder: learnable non-frequency tokens (no DCT content).
        if not self.use_freq_encoder:
            self.freq_placeholder = nn.Parameter(torch.zeros(1, freq_cfg["num_tokens"], token_dim))
            nn.init.trunc_normal_(self.freq_placeholder, std=0.02)

        self.spatial_encoder = SpatialEncoder(
            backbone=spatial_cfg["backbone"],
            pretrained=spatial_cfg["pretrained"],
            token_dim=token_dim,
            fail_on_missing=spatial_cfg["fail_on_missing"],
            allow_fallback=not spatial_cfg["fail_on_missing"],
        )

        self.cdt = CrossDomainTransformer(
            token_dim=token_dim,
            latent_dim=latent_dim,
            num_heads=cdt_cfg["num_heads"],
            num_layers=cdt_cfg["num_layers"],
            bidirectional=cdt_cfg["bidirectional"],
            ffn_ratio=cdt_cfg["ffn_ratio"],
        )

        # Independent pooling path used when use_cdt=False.
        self.to_latent_no_cdt = nn.Linear(token_dim, latent_dim)

        if self.use_diffusion:
            uk = diffusion_cfg.get("unet", {})
            self.diffusion = DiffusionRegularizer(
                latent_dim=latent_dim,
                timesteps=diffusion_cfg["timesteps"],
                beta_start=diffusion_cfg["beta_start"],
                beta_end=diffusion_cfg["beta_end"],
                device=device,
                unet_kwargs=uk,
            )
        else:
            self.diffusion = None

    # ------------------------------------------------------------------ helpers
    def _freq_tokens(self, sketch: torch.Tensor) -> torch.Tensor:
        if self.use_freq_encoder:
            return self.freq_encoder(sketch)
        B = sketch.size(0)
        return self.freq_placeholder.expand(B, -1, -1)

    def _spatial_tokens(self, photo: torch.Tensor) -> torch.Tensor:
        return self.spatial_encoder(photo)

    def _fuse(self, freq_t: torch.Tensor, spat_t: torch.Tensor):
        if self.use_cdt:
            return self.cdt(freq_t, spat_t)
        # bypass CDT: pool each domain independently and sum into latent
        fp = freq_t.mean(dim=1)
        sp = spat_t.mean(dim=1)
        latent = self.to_latent_no_cdt(fp) + self.to_latent_no_cdt(sp)
        return dict(
            freq_aligned=freq_t,
            spatial_aligned=spat_t,
            latent=latent,
            freq_pooled=fp,
            spatial_pooled=sp,
        )

    # ------------------------------------------------------------ training fwd
    def forward(self, sketch: torch.Tensor, photo: torch.Tensor) -> dict:
        """Paired training forward.

        Returns a dict with: latent (B,latent_dim), freq_aligned, spatial_aligned,
        freq_pooled, spatial_pooled.
        """
        freq_t = self._freq_tokens(sketch)
        spat_t = self._spatial_tokens(photo)
        return self._fuse(freq_t, spat_t)

    # ------------------------------------------------------- retrieval embeddings
    @torch.no_grad()
    def embed_sketch(self, sketch: torch.Tensor) -> torch.Tensor:
        """Query embedding (B, latent_dim) for a batch of sketches."""
        self.eval()
        freq_t = self._freq_tokens(sketch)
        B = freq_t.size(0)
        Ns_placeholder = 1  # one zero context token for the absent spatial domain
        zeros_spat = torch.zeros(B, Ns_placeholder, freq_t.size(-1),
                                 device=freq_t.device, dtype=freq_t.dtype)
        out = self._fuse(freq_t, zeros_spat)
        return out["latent"]

    @torch.no_grad()
    def embed_photo(self, photo: torch.Tensor) -> torch.Tensor:
        """Gallery embedding (B, latent_dim) for a batch of photos."""
        self.eval()
        spat_t = self._spatial_tokens(photo)
        B = spat_t.size(0)
        Nf_placeholder = 1  # one zero context token for the absent frequency domain
        zeros_freq = torch.zeros(B, Nf_placeholder, spat_t.size(-1),
                                 device=spat_t.device, dtype=spat_t.dtype)
        out = self._fuse(zeros_freq, spat_t)
        return out["latent"]

    # ---------------------------------------------------------- diffusion loss
    def diffusion_loss(self, latent: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        if not self.use_diffusion or self.diffusion is None:
            return torch.zeros((), device=latent.device, dtype=latent.dtype)
        return self.diffusion.training_loss(latent, cond=cond)


def build_model(cfg: dict, device: str = "cpu") -> FreqDiffFormer:
    """Build the full FreqDiffFormer from a parsed config dict."""
    m = cfg["model"]
    return FreqDiffFormer(
        freq_cfg=m["freq"],
        spatial_cfg=m["spatial"],
        cdt_cfg=m["cdt"],
        diffusion_cfg=cfg["diffusion"],
        latent_dim=m["latent_dim"],
        input_size=m["input_size"],
        ablation=cfg["training"]["ablation"],
        device=device,
    )

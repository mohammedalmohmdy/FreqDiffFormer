"""Spatial backbone — Swin Transformer-Tiny.

Manuscript: spatial backbone is Swin Transformer-Tiny, input 224x224.

Scientific-safety rule: never silently substitute a lightweight CNN when the
manuscript requires Swin. Therefore, if timm or the Swin model is unavailable
this module raises a clear error (unless explicitly degraded for CI via
`allow_fallback`, which produces an explicit, clearly-marked dummy backbone —
never used by the default manuscript pipeline).
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import timm
    _HAS_TIMM = True
except Exception:
    timm = None
    _HAS_TIMM = False


SWIN_NAME = "swin_tiny_patch4_window7_224"


class SpatialEncoder(nn.Module):
    """Swin Transformer-Tiny spatial encoder.

    Forward returns patch/token features of shape (B, num_tokens, token_dim),
    matching the frequency token layout so the Cross-Domain Transformer can do
    genuine cross-attention between the two domains.
    """

    def __init__(
        self,
        backbone: str = SWIN_NAME,
        pretrained: bool = True,
        token_dim: int = 256,
        fail_on_missing: bool = True,
        allow_fallback: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.token_dim = token_dim
        self.allow_fallback = allow_fallback

        if not _HAS_TIMM:
            if fail_on_missing and not allow_fallback:
                raise ImportError(
                    "The manuscript requires a Swin Transformer-Tiny spatial backbone, "
                    "but the 'timm' package is not installed. Install timm "
                    "(pip install timm) or download the weights; this is a hard "
                    "requirement — no silent lightweight fallback is permitted.")
            # explicit, clearly-marked degraded path for CI only
            self._use_timm = False
            self._feat_dim = 96  # dummy
            self.proj = nn.Linear(96, token_dim)
            self.feat_dim = 96
            return

        try:
            # features_only gives stage maps; we take the last high-res feature map
            # and treat it as a token grid.
            self.backbone = timm.create_model(
                backbone, pretrained=pretrained, num_classes=0, features_only=True
            )
            self._use_timm = True
            self.feat_dim = self.backbone.feature_info[-1]["num_chs"]
            self.proj = nn.Linear(self.feat_dim, token_dim)
        except Exception as e:
            if fail_on_missing and not allow_fallback:
                raise RuntimeError(
                    f"Could not construct Swin Transformer-Tiny '{backbone}'. "
                    f"Required by the manuscript. Original error: {e}")
            self._use_timm = False
            self._feat_dim = 96
            self.proj = nn.Linear(96, token_dim)
            self.feat_dim = 96

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) photo.
        Returns:
            tokens: (B, num_tokens, token_dim) spatial tokens, L2-normalizable
            downstream.
        """
        if x.dim() != 4 or x.size(1) != 3:
            raise ValueError(f"SpatialEncoder expects (B,3,H,W), got {tuple(x.shape)}")
        if not self._use_timm:
            # explicit degraded dummy backbone (CI only). Produces 49 tokens.
            B = x.size(0)
            feat = torch.zeros(B, 96, 7, 7, device=x.device, dtype=x.dtype)
            tokens = feat.flatten(2).transpose(1, 2)     # (B, 49, 96)
            tokens = self.proj(tokens)                    # (B, 49, token_dim)
            return tokens
        feats = self.backbone(x)[-1]                      # timm Swin -> NHWC
        if feats.dim() == 4:
            # detect layout: channel dim is the one equal to self.feat_dim.
            if feats.shape[1] == self.feat_dim:
                pass  # already NCHW
            elif feats.shape[-1] == self.feat_dim:
                feats = feats.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW
            else:
                # fallback: assume NHWC for timm Swin
                feats = feats.permute(0, 3, 1, 2).contiguous()
        feats = feats.to(x.device, dtype=x.dtype)
        B, C, h, w = feats.shape
        tokens = feats.flatten(2).transpose(1, 2)        # (B, h*w, C)
        tokens = self.proj(tokens)                        # (B, h*w, token_dim)
        return tokens

    def forward_pooled(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled global spatial embedding (B, token_dim). Used for the latent."""
        tokens = self.forward(x)
        return tokens.mean(dim=1)

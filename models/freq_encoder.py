"""Frequency Encoder (FEM) — manuscript-consistent implementation.

FREQDIFFFORMER: Frequency-Guided Transformer-Diffusion for FG-SBIR.

Frequency representation is DCT-based. The encoder implements the manuscript
processing pipeline:
  - 2D DCT-II (orthonormal) of the grayscale sketch
  - low/high-frequency structural information (two frequency bands)
  - frequency masking (radius-based low-pass / high-pass)
  - sign-logarithmic compression:  sign(x) * log(1 + |x|)
  - robust min-max scaling (median / MAD centring, then min-max to [0, 1])
  - four convolutional blocks (InstanceNorm + ReLU)
        channels 2 -> 64 -> 128 -> 256 -> 256
        3x3 kernels; blocks 1-3 stride 2 padding 1; block 4 stride 1 padding 1
  - adaptive average pooling to 14x14
  - 196 frequency tokens of dimension 256
  - learnable positional embedding
  - LayerNorm

Output:  frequency tokens  (B, 196, 256)

The DCT is implemented in pure torch (matrix form) so it is differentiable and
device-portable; it does not require scipy and never leaves the autograd graph.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Differentiable 2D DCT-II (orthonormal)
# ---------------------------------------------------------------------------

def _dct_basis(N: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return the orthonormal DCT-II basis matrix D of shape (N, N).

    XFreq = D @ X  performs the 1D DCT-II along the last axis.
    D[k, n] = sqrt(2/N) * cos(pi * (2n+1) * k / (2N)), with D[0, n] = sqrt(1/N).
    """
    n = torch.arange(N, device=device, dtype=dtype).unsqueeze(0)   # (1, N)
    k = torch.arange(N, device=device, dtype=dtype).unsqueeze(1)   # (N, 1)
    D = math.sqrt(2.0 / N) * torch.cos(math.pi * (2 * n + 1) * k / (2 * N))
    D[0, :] = math.sqrt(1.0 / N)
    return D


def dct_2d(x: torch.Tensor) -> torch.Tensor:
    """Functional differentiable 2D DCT-II on (B, C, H, W)."""
    if x.dim() == 3:
        x = x.unsqueeze(1)
        squeeze = True
    else:
        squeeze = False
    B, C, H, W = x.shape
    Dh = _dct_basis(H, x.device, x.dtype)
    Dw = _dct_basis(W, x.device, x.dtype)
    # DCT along height: out = Dh @ x   -> (B, C, H, W)
    out = torch.matmul(Dh, x)
    # DCT along width:  out = out @ Dw^T
    out = torch.matmul(out, Dw.transpose(0, 1))
    return out.squeeze(1) if squeeze else out


# ---------------------------------------------------------------------------
# Frequency masks
# ---------------------------------------------------------------------------

def _radial_masks(H: int, W: int, low_ratio: float, high_ratio: float,
                  device: torch.device) -> torch.Tensor:
    """Return a (2, H, W) mask tensor.

    Channel 0 (low band):  keep coefficients within low_ratio * min(H, W) of the
        DCT origin ([0,0]).
    Channel 1 (high band):  keep coefficients farther than high_ratio * min(H, W)
        from the origin (high-pass / complement of the inner disk).

    Masks are binary (kept = 1). The DCT-II origin is at index [0, 0].
    """
    yy = torch.arange(H, device=device).view(H, 1).float()
    xx = torch.arange(W, device=device).view(1, W).float()
    # distance from the top-left origin in index units; normalized by min side.
    side = float(min(H, W))
    dist = torch.sqrt(yy ** 2 + xx ** 2) / side
    low = (dist <= low_ratio).float()
    high = (dist > high_ratio).float()
    return torch.stack([low, high], dim=0)  # (2, H, W)


# ---------------------------------------------------------------------------
# Sign-logarithmic compression and robust min-max scaling
# ---------------------------------------------------------------------------

def sign_log_compress(x: torch.Tensor) -> torch.Tensor:
    """sign(x) * log(1 + |x|).  Compresses the wide dynamic range of DCT energy."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def robust_min_max(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Robust min-max scaling.

    For each sample-channel over the spatial plane:
      1. centre by the median and scale by the median absolute deviation (MAD);
      2. rescale the resulting values to [0, 1] with a per-cell min-max.

    More robust to outliers than plain min-max, matching the manuscript's
    "robust min-max scaling".
    Input/output shape: (B, C, H, W).
    """
    B, C, H, W = x.shape
    flat = x.reshape(B, C, -1)                 # (B, C, H*W)
    med = flat.median(dim=-1, keepdim=True).values
    mad = (flat - med).abs().median(dim=-1, keepdim=True).values
    robust = (flat - med) / (mad + eps)
    mn = robust.min(dim=-1, keepdim=True).values
    mx = robust.max(dim=-1, keepdim=True).values
    scaled = (robust - mn) / (mx - mn + eps)
    return scaled.reshape(B, C, H, W)


# ---------------------------------------------------------------------------
# Frequency encoder (FEM)
# ---------------------------------------------------------------------------

class FreqEncoder(nn.Module):
    """DCT frequency encoder producing 196 frequency tokens of dim 256.

    Args mirror configs/default.yaml -> model.freq.
    """

    def __init__(
        self,
        in_channels: int = 2,
        token_dim: int = 256,
        num_tokens: int = 196,
        token_grid: int = 14,
        conv_channels=(64, 128, 256, 256),
        kernel_size: int = 3,
        strides=(2, 2, 2, 1),
        paddings=(1, 1, 1, 1),
        norm: str = "instance",
        activation: str = "relu",
        use_sign_log: bool = True,
        use_robust_minmax: bool = True,
        low_freq_mask_ratio: float = 0.25,
        high_freq_mask_ratio: float = 0.25,
        positional_embedding: str = "learnable",
        layer_norm: bool = True,
        input_size: int = 224,
    ):
        super().__init__()
        if norm != "instance":
            raise ValueError(
                f"FreqEncoder requires InstanceNorm per manuscript (got norm='{norm}'). "
                "BatchNorm is intentionally not supported here."
            )
        if activation != "relu":
            raise ValueError(f"FreqEncoder requires ReLU per manuscript (got '{activation}').")
        if len(conv_channels) != 4:
            raise ValueError("Manuscript specifies exactly four convolutional blocks; "
                             f"got {len(conv_channels)}.")
        if len(strides) != 4 or len(paddings) != 4:
            raise ValueError("strides/paddings must have length 4 (one per conv block).")

        self.in_channels = in_channels
        self.token_dim = token_dim
        self.num_tokens = num_tokens
        self.token_grid = token_grid
        self.use_sign_log = use_sign_log
        self.use_robust_minmax = use_robust_minmax
        self.low_freq_mask_ratio = low_freq_mask_ratio
        self.high_freq_mask_ratio = high_freq_mask_ratio
        self.input_size = input_size

        # Frequency masks are deterministic buffers (depend on input resolution).
        masks = _radial_masks(input_size, input_size, low_freq_mask_ratio,
                              high_freq_mask_ratio, torch.device("cpu"))
        # (2, H, W) -> broadcast against (B, 2, H, W)
        self.register_buffer("freq_masks", masks)

        # --- four convolutional blocks --------------------------------------
        in_ch = in_channels
        blocks = []
        for i in range(4):
            out_ch = conv_channels[i]
            blocks.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                                    stride=strides[i], padding=paddings[i]))
            blocks.append(nn.InstanceNorm2d(out_ch, affine=True,
                                             track_running_stats=False))
            blocks.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        self.convs = nn.Sequential(*blocks)
        self.out_channels = in_ch  # 256

        # adaptive average pool to 14x14 -> 196 tokens
        self.pool = nn.AdaptiveAvgPool2d((token_grid, token_grid))

        if self.out_channels != token_dim:
            # project channels to token_dim (256 == 256 in manuscript, identity-like)
            self.proj = nn.Linear(self.out_channels, token_dim)
        else:
            self.proj = nn.Identity()

        if positional_embedding == "learnable":
            self.pos_emb = nn.Parameter(torch.zeros(1, num_tokens, token_dim))
            nn.init.trunc_normal_(self.pos_emb, std=0.02)
        else:
            self.register_buffer("pos_emb", torch.zeros(1, num_tokens, token_dim))

        self.layer_norm = nn.LayerNorm(token_dim) if layer_norm else nn.Identity()

    def frequency_preprocess(self, sketch: torch.Tensor) -> torch.Tensor:
        """sketch (B, 1, H, W) grayscale -> 2-channel frequency representation (B, 2, H, W)."""
        if sketch.dim() == 3:
            sketch = sketch.unsqueeze(1)
        if sketch.size(1) != 1:
            # if multi-channel arrives, average to grayscale
            sketch = sketch.mean(dim=1, keepdim=True)
        B, C, H, W = sketch.shape
        # DCT-based frequency representation (per channel, here single channel).
        freq = dct_2d(sketch)  # (B, 1, H, W)

        # apply low/high-frequency structural masks -> 2 channels
        masks = self.freq_masks.to(device=freq.device, dtype=freq.dtype)  # (2, H, W)
        masks = masks.unsqueeze(0).expand(B, -1, -1, -1)                  # (B, 2, H, W)
        structural = freq * masks                                          # (B, 2, H, W)

        # sign-logarithmic compression
        if self.use_sign_log:
            structural = sign_log_compress(structural)

        # robust min-max scaling
        if self.use_robust_minmax:
            structural = robust_min_max(structural)

        return structural

    def forward(self, sketch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sketch: (B, 1, H, W) grayscale sketch tensor, any normalization.
        Returns:
            tokens: (B, num_tokens, token_dim) = (B, 196, 256) frequency tokens.
        """
        x = self.frequency_preprocess(sketch)          # (B, 2, H, W)
        x = self.convs(x)                              # (B, 256, h, w)
        x = self.pool(x)                               # (B, 256, 14, 14)
        B, Ch, Gh, Gw = x.shape
        assert Gh == self.token_grid and Gw == self.token_grid, (
            f"expected pooled grid {self.token_grid}x{self.token_grid}, got {Gh}x{Gw}")
        x = x.flatten(2).transpose(1, 2)                # (B, 196, 256)
        x = self.proj(x)                               # (B, 196, token_dim)
        x = x + self.pos_emb                            # add learnable positional embedding
        x = self.layer_norm(x)                          # LayerNorm over token_dim
        return x


# Backward-compatible alias (old code imports DCTFrequencyEncoder).
DCTFrequencyEncoder = FreqEncoder

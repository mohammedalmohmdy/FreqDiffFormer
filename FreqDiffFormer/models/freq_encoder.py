
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    # prefer scipy's dct for correctness/performance if available
    from scipy.fftpack import dct as scipy_dct
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

"""
DCT-based Frequency Encoder
- Applies 2D DCT (type-II) to the input sketch image (grayscale)
- Extracts high-frequency coefficients (configurable) and feeds them through a small conv encoder
- If scipy is available it will use scipy.fftpack.dct for the transform; otherwise a torch implementation is used.
"""

def dct_1d(x, norm='ortho'):
    # x: (..., N)
    # Use scipy if available
    if _HAS_SCIPY:
        import numpy as _np
        arr = x.detach().cpu().numpy()
        y = scipy_dct(arr, type=2, axis=-1, norm=norm)
        return torch.from_numpy(y).to(x.device, dtype=x.dtype)
    # Torch implementation (using FFT trick)
    N = x.shape[-1]
    v = torch.cat([x, x.flip(dims=[-1])], dim=-1)
    V = torch.fft.rfft(v, n=None, dim=-1)
    k = torch.arange(N, device=x.device).reshape([*( [1]*(x.dim()-1) ), -1])
    W_real = torch.cos(-torch.pi * k / (2*N))
    W_imag = torch.sin(-torch.pi * k / (2*N))
    V = V[..., :N]
    # reconstruct DCT-II coefficients (approx)
    return (V.real * W_real - V.imag * W_imag)

def dct_2d(x):
    """
    x: (B, C, H, W) or (B, H, W)
    returns tensor of same shape with DCT applied per channel
    """
    squeeze_channel = False
    if x.dim() == 3:  # (B, H, W)
        x = x.unsqueeze(1)
        squeeze_channel = True
    B, C, H, W = x.shape
    # apply 1D DCT over last dim then over second last
    # prefer scipy dctn if available for correctness
    if _HAS_SCIPY:
        # use numpy implementation per-sample for simplicity
        import numpy as _np
        out = torch.empty_like(x)
        for b in range(B):
            for c in range(C):
                arr = x[b,c].detach().cpu().numpy()
                # apply dct over axis=0 then axis=1 (type-II, norm='ortho')
                tmp = scipy_dct(arr, type=2, axis=0, norm='ortho')
                tmp = scipy_dct(tmp, type=2, axis=1, norm='ortho')
                out[b,c] = torch.from_numpy(tmp).to(x.device, dtype=x.dtype)
        if squeeze_channel:
            out = out.squeeze(1)
        return out
    # Torch fallback: separable 1D DCT using FFT-based approximation
    # First along width, then along height
    # we implement a relatively accurate DCT-II using real FFT trick
    # Note: This fallback is approximate but suitable when scipy not available.
    # Apply to each row
    # reshape to (B*C*H, W)
    x_reshaped = x.permute(0,1,2,3).contiguous().view(B*C*H, W)
    # pad and use rfft trick
    # mirror padding
    v = torch.cat([x_reshaped, x_reshaped.flip(dims=[-1])], dim=-1)
    V = torch.fft.rfft(v, dim=-1)
    N = W
    k = torch.arange(N, device=x.device).reshape(1, -1)
    W_real = torch.cos(-torch.pi * k / (2*N))
    W_imag = torch.sin(-torch.pi * k / (2*N))
    V = V[..., :N]
    Xw = (V.real * W_real - V.imag * W_imag)
    Xw = Xw.view(B, C, H, W)
    # now DCT along height: treat as (B*C*W, H)
    Xh_reshaped = Xw.permute(0,1,3,2).contiguous().view(B*C*W, H)
    v2 = torch.cat([Xh_reshaped, Xh_reshaped.flip(dims=[-1])], dim=-1)
    V2 = torch.fft.rfft(v2, dim=-1)
    N2 = H
    k2 = torch.arange(N2, device=x.device).reshape(1, -1)
    W2_real = torch.cos(-torch.pi * k2 / (2*N2))
    W2_imag = torch.sin(-torch.pi * k2 / (2*N2))
    V2 = V2[..., :N2]
    Xh = (V2.real * W2_real - V2.imag * W2_imag)
    Xh = Xh.view(B, C, W, H).permute(0,1,3,2)
    return Xh

class DCTFrequencyEncoder(nn.Module):
    def __init__(self, in_ch=1, emb_dim=256, use_topk=0.5):
        """
        in_ch: input channels (1 for sketches)
        emb_dim: output embedding dimension
        use_topk: fraction of high-frequency coefficients to retain (0-1)
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.use_topk = use_topk
        # small conv stack to process spectral maps
        mid = max(64, emb_dim//2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, mid, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.proj = nn.Linear(mid, emb_dim)

    def forward(self, x):
        """
        x: (B, 1, H, W) float tensor in range [0,1] or normalized
        returns: (B, emb_dim)
        """
        # Ensure single-channel
        if x.dim() == 3:
            x = x.unsqueeze(1)
        # Apply 2D-DCT
        X = dct_2d(x)
        # Optionally keep high-frequency coefficients by masking low-freq center
        B, C, H, W = X.shape
        # create mask that zeros out low-frequency coefficients (keep topk fraction)
        k_h = int(H * self.use_topk)
        k_w = int(W * self.use_topk)
        mask = torch.zeros((H,W), device=X.device)
        # keep the corners (high-frequency)
        mask[:k_h, :k_w] = 1.0
        mask = mask.unsqueeze(0).unsqueeze(0)  # 1,1,H,W
        X_masked = X * mask
        out = self.conv(X_masked)
        out = out.view(out.size(0), -1)
        out = self.proj(out)
        return out

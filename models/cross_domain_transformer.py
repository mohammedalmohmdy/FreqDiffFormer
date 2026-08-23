"""Cross-Domain Transformer alignment.

The manuscript requires *genuine cross-domain / cross-attention interaction*, not
merely stacking two embeddings and applying a generic TransformerEncoder. This
module implements a bidirectional cross-attention cross-encoder:

  - frequency tokens  F  (B, Nf, D)   from the FEM
  - spatial tokens    S  (B, Ns, D)   from Swin-Tiny

  For each layer:
    F' = LayerNorm(F + CrossAttn(Q=F, K=S, V=S))   # frequency attends to spatial
    S' = LayerNorm(S + CrossAttn(Q=S, K=F, V=F))   # spatial attends to frequency
    followed by standard feed-forward sublayers (with residuals + LayerNorm).

Domain-conditioned by learnable domain embeddings added to F / S so each branch
knows which modality it is. The fused representation is projected to the latent
dimension (512) for downstream heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionLayer(nn.Module):
    """One cross-attention + FFN block. Q-domain attends over K/V-domain."""

    def __init__(self, dim: int, num_heads: int, ffn_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                                batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim * ffn_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * ffn_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # Q-domain tokens attend over K/V = the other domain. Post-norm residuals.
        attn_out, _ = self.cross_attn(self.norm1(q), self.norm1(kv), self.norm1(kv),
                                       need_weights=False)
        q = q + attn_out
        q = q + self.ffn(self.norm2(q))
        return q


class CrossDomainTransformer(nn.Module):
    """Bidirectional cross-attention cross-encoder (genuine cross-domain).

    Returns two aligned token sequences (freq, spatial) and a pooled fused latent
    projected to `latent_dim`.
    """

    def __init__(
        self,
        token_dim: int = 256,
        latent_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        bidirectional: bool = True,
        ffn_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.latent_dim = latent_dim
        self.bidirectional = bidirectional

        # learnable domain embeddings distinguish the two modalities
        self.freq_domain_emb = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.spatial_domain_emb = nn.Parameter(torch.zeros(1, 1, token_dim))
        nn.init.trunc_normal_(self.freq_domain_emb, std=0.02)
        nn.init.trunc_normal_(self.spatial_domain_emb, std=0.02)

        # we build num_layers cross-attention blocks for each direction
        self.freq_layers = nn.ModuleList([
            CrossAttentionLayer(token_dim, num_heads, ffn_ratio, dropout)
            for _ in range(num_layers)
        ])
        if bidirectional:
            self.spatial_layers = nn.ModuleList([
                CrossAttentionLayer(token_dim, num_heads, ffn_ratio, dropout)
                for _ in range(num_layers)
            ])
        else:
            self.spatial_layers = None

        self.norm = nn.LayerNorm(token_dim)
        # fuse both domains -> latent
        self.to_latent = nn.Linear(token_dim, latent_dim)

    def forward(self, freq_tokens: torch.Tensor, spatial_tokens: torch.Tensor):
        """
        Args:
            freq_tokens:    (B, Nf, token_dim) from the FEM.
            spatial_tokens: (B, Ns, token_dim) from Swin-Tiny.
        Returns:
            freq_aligned:   (B, Nf, token_dim)    frequency tokens aligned to spatial
            spatial_aligned:(B, Ns, token_dim)   spatial tokens aligned to frequency
            latent:         (B, latent_dim)      fused cross-domain latent
            freq_pooled:    (B, token_dim)       mean-pooled aligned freq tokens
            spatial_pooled:(B, token_dim)       mean-pooled aligned spatial tokens
        """
        f = freq_tokens + self.freq_domain_emb
        s = spatial_tokens + self.spatial_domain_emb

        for i, layer in enumerate(self.freq_layers):
            f = layer(f, s)                     # frequency attends to spatial
            if self.spatial_layers is not None:
                s = self.spatial_layers[i](s, f)  # spatial attends to frequency

        f = self.norm(f)
        s = self.norm(s) if self.spatial_layers is not None else s

        freq_pooled = f.mean(dim=1)            # (B, token_dim)
        spatial_pooled = s.mean(dim=1)
        latent = self.to_latent(freq_pooled + spatial_pooled)   # (B, latent_dim)
        return dict(
            freq_aligned=f,
            spatial_aligned=s,
            latent=latent,
            freq_pooled=freq_pooled,
            spatial_pooled=spatial_pooled,
        )

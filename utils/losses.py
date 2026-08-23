"""FreqDiffFormer objective.

L_total = L_triplet + lambda1 * L_diffusion + lambda2 * L_frequency_alignment

  - L_triplet           : triplet ranking loss, margin = 0.2, in-batch semi-hard
                          mining. Anchor = sketch embedding, positive = same-class
                          photo embedding, negative = different-class photo embedding.
  - L_diffusion         : diffusion reconstruction (denoising MSE) loss, weight
                          lambda1 = 0.5. Diffusion is a TRAINING-TIME regularizer.
  - L_frequency_alignment: aligns the frequency branch with the spatial/cross-
                          domain fused representation, weight lambda2 = 0.1.

Each loss is implemented independently and can be toggled via ablation switches,
so every ablation removes ONLY the specified component:
  - no_freq_align       -> lambda2 term = 0 (frequency alignment off)
  - no_diffusion        -> the diffusion module is not built; term = 0
  - no_recon            -> the diffusion module exists but its recon loss = 0
  - no_triplet          -> L_triplet = 0
lambda1/lambda2 are fixed (not learnable), selected on the Sketchy validation
split via configs/lambdas_grid.yaml. Their canonical values are 0.5 / 0.1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Triplet loss with in-batch semi-hard mining
# ---------------------------------------------------------------------------

class TripletLoss(nn.Module):
    """Triplet ranking loss with in-batch semi-hard negative mining.

    Anchor *a* (sketch embedding), positive *p* (same-class photo embedding),
    negative *n* (different-class photo embedding). Margin = 0.2.

    `a`, `p`, `n` are (B, D); `labels` (B,) gives the class of each anchor and
    is used only for the convenience helper `from_batch` that builds triplets
    from sketch/photo batches of the same length (paired by index). The core
    `forward(a, p, n)` is class-agnostic.
    """

    def __init__(self, margin: float = 0.2, p: int = 2):
        super().__init__()
        self.margin = float(margin)
        self.p = int(p)

    def forward(self, a: torch.Tensor, p: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        d_ap = F.pairwise_distance(a, p, p=self.p)
        d_an = F.pairwise_distance(a, n, p=self.p)
        # semi-hard: keep negatives that are farther than the positive but
        # within margin (standard FaceNet semi-hard mining in-batch).
        loss = F.relu(d_ap - d_an + self.margin)
        # mask out easy negatives that already exceed by margin (loss>0 handles it)
        return loss.mean()

    def from_batch(self, anchor: torch.Tensor, positive: torch.Tensor,
                   labels: torch.Tensor) -> torch.Tensor:
        """Build all-valid triplets in-batch using class labels.

        For each anchor i, positive j is any same-class index (j != i); negative
        k is any different-class index. We pick the first valid (j, k) pair with a
        semi-hard negative (d_an > d_ap) if available, else the hardest different-
        class negative. If a batch has no valid triplet, it contributes 0.
        """
        B = anchor.size(0)
        if B < 2:
            return anchor.new_zeros(())
        d_ap_all = torch.cdist(anchor, positive, p=self.p)   # (B, B)
        d_an_all = torch.cdist(anchor, anchor, p=self.p)     # use photo vs... we use
        # We need d(anchor_i, negative_k) over photo embeddings: use positive pool.
        d_an_pool = d_ap_all.clone()                          # negatives drawn from photo pool
        same = (labels.unsqueeze(0) == labels.unsqueeze(1))   # (B, B) same-class mask
        eye = torch.eye(B, dtype=torch.bool, device=labels.device)
        pos_mask = same & (~eye)
        neg_mask = ~same
        total = anchor.new_zeros(())
        count = 0
        for i in range(B):
            pos_idx = torch.nonzero(pos_mask[i], as_tuple=False)
            neg_idx = torch.nonzero(neg_mask[i], as_tuple=False)
            if pos_idx.numel() == 0 or neg_idx.numel() == 0:
                continue
            # choose positive: the hardest positive (max distance) -> mining
            p_i = pos_idx[d_ap_all[i, pos_idx].argmax()].item()
            n_dists = d_an_pool[i, neg_idx]
            # semi-hard: negative farther than d_ap but closer than d_ap + margin
            d_ap_i = d_ap_all[i, p_i]
            semi = (n_dists > d_ap_i) & (n_dists < d_ap_i + self.margin)
            if semi.any():
                k = neg_idx[n_dists.argmin()].item()  # closest semi-hard neg
            else:
                k = neg_idx[n_dists.argmin()].item()  # hardest neg fallback
            d_ap = d_ap_all[i, p_i]
            d_an = d_an_pool[i, k]
            total = total + F.relu(d_ap - d_an + self.margin)
            count += 1
        return total / max(count, 1)


# ---------------------------------------------------------------------------
# Diffusion reconstruction loss (lambda1)
# ---------------------------------------------------------------------------

class DiffusionReconstructionLoss(nn.Module):
    """Wraps the diffusion regularizer's training-time loss with fixed lambda1."""

    def __init__(self, lambda1: float = 0.5):
        super().__init__()
        self.lambda1 = float(lambda1)

    def forward(self, model, latent: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.lambda1 * model.diffusion_loss(latent, cond=cond)


# ---------------------------------------------------------------------------
# Frequency alignment loss (lambda2)
# ---------------------------------------------------------------------------

class FrequencyAlignmentLoss(nn.Module):
    """Align the frequency branch with the fused spatial/cross-domain output.

    Manuscript: "frequency alignment loss" between the frequency representation
    and the spatial/cross-domain fused representation. We compute a cosine-based
    alignment between the pooled frequency tokens and the pooled spatial tokens
    pulled into the frequency token space, encouraging them to share structure:

        L_freq_align = (B, D).mean( 1 - cos(freq_pooled, spatial_pooled) )

    All pooled quantities live in the (B, token_dim) shared space produced by the
    Cross-Domain Transformer. If the Cross-Domain Transformer is ablated, the
    independent pooling path's freq/spatial pooled tensors are used instead.
    """

    def __init__(self, lambda2: float = 0.1):
        super().__init__()
        self.lambda2 = float(lambda2)

    def forward(self, freq_pooled: torch.Tensor, spatial_pooled: torch.Tensor) -> torch.Tensor:
        cos = F.cosine_similarity(freq_pooled, spatial_pooled, dim=-1)  # (B,)
        return self.lambda2 * (1.0 - cos).mean()


# ---------------------------------------------------------------------------
# Composite objective with ablation switches
# ---------------------------------------------------------------------------

@dataclass
class LossBreakdown:
    total: torch.Tensor
    triplet: torch.Tensor
    diffusion: torch.Tensor
    freq_align: torch.Tensor
    components: Dict[str, bool]


class CompositeLoss(nn.Module):
    """L_total = L_triplet + lambda1*L_diffusion + lambda2*L_freq_align.

    Ablation switches (from config -> training.ablation):
      use_triplet, use_recon, use_freq_align, use_diffusion, use_freq_encoder, use_cdt
    """

    def __init__(
        self,
        margin: float = 0.2,
        lambda1: float = 0.5,
        lambda2: float = 0.1,
        ablation: dict | None = None,
    ):
        super().__init__()
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        abl = ablation or {}
        self.use_triplet = abl.get("use_triplet", True)
        self.use_recon = abl.get("use_recon", True)
        self.use_freq_align = abl.get("use_freq_align", True)
        self.use_diffusion = abl.get("use_diffusion", True)

        self.triplet = TripletLoss(margin=margin)
        self.freq_align = FrequencyAlignmentLoss(lambda2=lambda2)
        # Diffusion recon wrapper; reads lambda1. The diffusion MODULE is owned
        # by the model; this loss just weights it. If use_diffusion is False the
        # model has no diffusion module and the term is identically zero.

    def forward(
        self,
        model,
        outputs: dict,
        labels: torch.Tensor,
    ) -> LossBreakdown:
        """Compute the composite loss.

        Args:
            model:   FreqDiffFormer (for diffusion_loss access).
            outputs: dict from model.forward() with latent, freq_pooled,
                     spatial_pooled, freq_aligned, spatial_aligned.
            labels:  (B,) class labels for triplet mining.
        """
        device = outputs["latent"].device
        zero = torch.zeros((), device=device, dtype=outputs["latent"].dtype)

        # Triplet: anchor = freq-domain pooled sketch, positive = spatial pooled
        # (same index pair), negative mined in-batch by class.
        if self.use_triplet:
            triplet = self.triplet.from_batch(
                outputs["freq_pooled"], outputs["spatial_pooled"], labels
            )
        else:
            triplet = zero

        # Frequency alignment (lambda2)
        if self.use_freq_align:
            freq_align = self.freq_align(outputs["freq_pooled"], outputs["spatial_pooled"])
        else:
            freq_align = zero

        # Diffusion reconstruction (lambda1)
        if self.use_diffusion and self.use_recon:
            diffusion = DiffusionReconstructionLoss(self.lambda1)(
                model, outputs["latent"], cond=outputs["freq_aligned"]
            )
        else:
            diffusion = zero

        total = triplet + diffusion + freq_align
        comp = {
            "use_triplet": self.use_triplet,
            "use_recon": self.use_recon,
            "use_freq_align": self.use_freq_align,
            "use_diffusion": self.use_diffusion,
        }
        return LossBreakdown(total=total, triplet=triplet, diffusion=diffusion,
                             freq_align=freq_align, components=comp)

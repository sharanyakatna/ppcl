"""
PPCL-Stable: EMA variance routing + cross-block decorrelation.

Vanilla PPCL picks plasticity dimensions from instantaneous batch variance,
which is noisy early in training and can flip across batches. Keeping an EMA
of per-dimension variance and selecting from that is much more stable in
practice (same reserve fraction r, just smoother routing).

The optional Frobenius penalty on cross-correlation between the stability and
plasticity blocks pushes them to be more orthogonal, which helps when both
objectives are pulling on the same coordinates simultaneously.

Safe to import without CUDA or Colab.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


class VarianceEMASubspace:
    """Per-task running EMA of projector variance for stable low/high splits."""

    __slots__ = ("dim", "momentum", "var_ema")

    def __init__(self, dim: int, momentum: float = 0.99):
        self.dim = int(dim)
        self.momentum = float(momentum)
        self.var_ema: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.var_ema = None

    def low_variance_indices(self, z1: torch.Tensor, z2: torch.Tensor, reserve_frac: float):
        """Return indices of the num_r lowest EMA-variance (plasticity) dimensions."""
        d = z1.shape[1]
        num_r = max(1, int(d * reserve_frac))
        batch_var = torch.cat([z1, z2], dim=0).var(dim=0).detach()
        if self.var_ema is None:
            self.var_ema = batch_var.clone()
        else:
            if self.var_ema.device != batch_var.device:
                self.var_ema = self.var_ema.to(batch_var.device)
            if self.var_ema.shape[0] != batch_var.shape[0]:
                self.var_ema = batch_var.clone()
            self.var_ema.mul_(self.momentum).add_(batch_var, alpha=1.0 - self.momentum)
        _, low_idx = self.var_ema.topk(num_r, largest=False)
        return low_idx


def ppcl_loss_stable_subspace(
    z1: torch.Tensor,
    z2: torch.Tensor,
    var_state: VarianceEMASubspace,
    reserve: float,
    temp: float,
    unif_weight: float,
    ntxent_fn,
    orth_lambda: float = 0.02,
):
    """
    PPCL with EMA variance routing and optional decorrelation penalty.

    The decorrelation term penalizes cross-correlation between the high-var
    (stability) and low-var (plasticity) blocks using a Frobenius norm on
    their batch cross-correlation matrix. Helps when both losses interfere.

    Returns (loss, er_diag, Lc_item, Lu_item, Lorth_item).
    er_diag is always 0.0 (skipping SVD on the fast path).
    """
    z1f, z2f = z1.float(), z2.float()
    d = z1f.shape[1]
    low_idx = var_state.low_variance_indices(z1f, z2f, reserve)
    hmask = torch.ones(d, dtype=torch.bool, device=z1f.device)
    hmask[low_idx] = False
    high_idx = torch.where(hmask)[0]

    Lc = ntxent_fn(z1f[:, high_idx], z2f[:, high_idx], temp)
    zf = F.normalize(torch.cat([z1f[:, low_idx], z2f[:, low_idx]], dim=0).float(), dim=1)
    # Wang & Isola (2020) uniformity: log E[exp(-2||u-v||^2)]; O(N^2) in batch size
    Lu = torch.pdist(zf, p=2).pow(2).mul(-2).exp().mean().clamp(min=1e-8).log()

    Lorth = z1f.new_tensor(0.0)
    if orth_lambda and orth_lambda > 0:
        zh1 = z1f[:, high_idx]
        zl1 = z1f[:, low_idx]
        b = zh1.size(0)
        if b > 1 and zh1.size(1) > 0 and zl1.size(1) > 0:
            zh = zh1 - zh1.mean(0, keepdim=True)
            zl = zl1 - zl1.mean(0, keepdim=True)
            c = (zh.T @ zl) / float(b)
            Lorth = (c.pow(2).sum()) * float(orth_lambda)

    loss = Lc + unif_weight * Lu + Lorth
    return loss, 0.0, Lc.item(), Lu.item(), float(Lorth.detach().item())

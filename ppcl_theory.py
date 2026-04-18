"""
Theory helpers for PPCL + BYOL.

The EMA target in BYOL is a Polyak average of the online weights. A clean
ablation of whether that anchoring is what actually helps continual SSL is
to replace it with explicit L2 regularization toward an EMA shadow copy.
See simclr_ema_l2 in core.run_method. Cite BYOL + Polyak literature
appropriately in the paper; the claim lives somewhere between a lemma and
an empirical story depending on how far we push the formalism.

Safe to import without CUDA or Colab.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def weight_modules_l2_squared_sum(mod_a: nn.Module, mod_b: nn.Module) -> torch.Tensor:
    """Sum of squared parameter differences between two modules with the same structure."""
    total = None
    for pa, pb in zip(mod_a.parameters(), mod_b.parameters()):
        d = (pa - pb).float().pow(2).sum()
        total = d if total is None else total + d
    if total is None:
        return torch.tensor(0.0)
    return total


def byol_online_target_enc_proj_l2(model) -> torch.Tensor:
    """||theta_online - theta_target||_2 over encoder + projector."""
    s = weight_modules_l2_squared_sum(model.enc, model.t_enc) + weight_modules_l2_squared_sum(
        model.proj, model.t_proj
    )
    return torch.sqrt(s.clamp(min=1e-20))


def simclr_shadow_enc_proj_l2(model, shadow_enc: nn.Module, shadow_proj: nn.Module) -> torch.Tensor:
    """||theta_online - theta_shadow||_2 for the explicit L2 ablation."""
    s = weight_modules_l2_squared_sum(model.enc, shadow_enc) + weight_modules_l2_squared_sum(
        model.proj, shadow_proj
    )
    return torch.sqrt(s.clamp(min=1e-20))


@torch.no_grad()
def ema_update_modules_(target: nn.Module, online: nn.Module, momentum: float) -> None:
    """In-place EMA: target <- tau*target + (1-tau)*online."""
    tau = float(momentum)
    for tp, op in zip(target.parameters(), online.parameters()):
        tp.data.mul_(tau).add_(op.data, alpha=1.0 - tau)

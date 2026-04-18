"""Lightweight tests for metrics, losses, and model (no GPU required)."""
import os
import sys

import pytest
import torch

# Repo root on path for `import core` / `ppcl_subspace`
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC TESTS
# ═══════════════════════════════════════════════════════════════════════════════
def test_forward_transfer_mean():
    from core import compute_fwd_transfer_mean

    assert compute_fwd_transfer_mean([]) == 0.0
    assert compute_fwd_transfer_mean([10.0, 20.0]) == 15.0


def test_forgetting_non_trivial():
    from core import compute_forgetting

    # Two tasks: no drop on task 0 -> forgetting 0
    acc = [[50.0], [60.0, 55.0]]
    assert compute_forgetting(acc) == 0.0
    # Clear regression on early tasks by the final row
    acc2 = [[90.0], [90.0, 85.0], [50.0, 60.0, 70.0]]
    assert compute_forgetting(acc2) > 0


def test_backward_transfer():
    from core import compute_backward_transfer

    acc = [[50.0], [60.0, 40.0], [55.0, 45.0, 30.0]]
    bwt = compute_backward_transfer(acc)
    assert bwt == pytest.approx(((45 - 40) + (55 - 50)) / 2)


# ═══════════════════════════════════════════════════════════════════════════════
# LOSS GRADIENT-FLOW TESTS
# ═══════════════════════════════════════════════════════════════════════════════
def test_ntxent_masked_fill_no_inplace_diag():
    from core import ntxent

    z1 = torch.randn(4, 8, requires_grad=True)
    z2 = torch.randn(4, 8, requires_grad=True)
    loss = ntxent(z1, z2, temp=0.5)
    loss.backward()
    assert z1.grad is not None


def test_barlow_loss_gradient_flow():
    from core import barlow_loss

    z1 = torch.randn(8, 32, requires_grad=True)
    z2 = torch.randn(8, 32, requires_grad=True)
    loss = barlow_loss(z1, z2)
    loss.backward()
    assert z1.grad is not None and z2.grad is not None
    assert torch.isfinite(loss).item()


def test_vicreg_loss_gradient_flow():
    from core import vicreg_loss

    z1 = torch.randn(8, 32, requires_grad=True)
    z2 = torch.randn(8, 32, requires_grad=True)
    loss = vicreg_loss(z1, z2)
    loss.backward()
    assert z1.grad is not None and z2.grad is not None
    assert torch.isfinite(loss).item()


def test_simsiam_loss_gradient_flow():
    from core import simsiam_loss

    p1 = torch.randn(8, 32, requires_grad=True)
    z2 = torch.randn(8, 32)
    p2 = torch.randn(8, 32, requires_grad=True)
    z1 = torch.randn(8, 32)
    loss = simsiam_loss(p1, z2, p2, z1)
    loss.backward()
    assert p1.grad is not None and p2.grad is not None


# ═══════════════════════════════════════════════════════════════════════════════
# PPCL-STABLE TESTS
# ═══════════════════════════════════════════════════════════════════════════════
def test_variance_ema_subspace_and_loss():
    from ppcl_subspace import VarianceEMASubspace, ppcl_loss_stable_subspace

    def ntxent(a, b, temp):
        a = torch.nn.functional.normalize(a.float(), dim=1)
        b = torch.nn.functional.normalize(b.float(), dim=1)
        return (a - b).pow(2).sum(dim=1).mean()

    b, d = 8, 32
    z1 = torch.randn(b, d, requires_grad=True)
    z2 = torch.randn(b, d, requires_grad=True)
    st = VarianceEMASubspace(d, momentum=0.9)
    loss, _, _, _, _ = ppcl_loss_stable_subspace(
        z1, z2, st, reserve=0.125, temp=0.5, unif_weight=0.05, ntxent_fn=ntxent, orth_lambda=0.01
    )
    loss.backward()
    assert z1.grad is not None and z2.grad is not None


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL TESTS
# ═══════════════════════════════════════════════════════════════════════════════
def test_sslmodel_forward_shapes():
    """Verify SSLModel output shapes for encoder, projector, and predictor."""
    from core import SSLModel

    model = SSLModel(use_pred=True, use_ema=True, ema_mom=0.996, in_size=32, proj_dim=128)
    x = torch.randn(4, 3, 32, 32)
    h, z, p = model(x)
    assert h.shape == (4, 512), f"Encoder output shape mismatch: {h.shape}"
    assert z.shape == (4, 128), f"Projector output shape mismatch: {z.shape}"
    assert p.shape == (4, 128), f"Predictor output shape mismatch: {p.shape}"
    # Target forward
    zt = model.forward_target(x)
    assert zt.shape == (4, 128), f"Target output shape mismatch: {zt.shape}"


def test_determinism_same_seed_same_loss():
    """Same seed must produce identical loss on the first batch."""
    from core import SSLModel, ntxent, set_global_determinism

    losses = []
    for _ in range(2):
        set_global_determinism(42)
        model = SSLModel(in_size=32, proj_dim=128)
        x = torch.randn(4, 3, 32, 32)
        _, z1, _ = model(x)
        _, z2, _ = model(x)
        loss = ntxent(z1, z2, temp=0.5)
        losses.append(loss.item())
    assert losses[0] == pytest.approx(losses[1], abs=1e-5), \
        f"Determinism violated: {losses[0]} != {losses[1]}"


# ═══════════════════════════════════════════════════════════════════════════════
# SAFETY TESTS
# ═══════════════════════════════════════════════════════════════════════════════
def test_lump_mixup_no_inplace():
    """lump_mixup must not modify the original tensor."""
    from core import lump_mixup
    import numpy as np

    np.random.seed(0)
    v1 = torch.randn(8, 3, 32, 32)
    v2 = torch.randn(8, 3, 32, 32)
    v1_original = v1.clone()
    buf = torch.randn(8, 3, 32, 32)
    out1, _ = lump_mixup(v1, v2, buf, None, alpha=0.4)
    # v1 should be unchanged (the function clones internally)
    assert torch.equal(v1, v1_original), "lump_mixup modified input tensor in-place"


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA / INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════
def test_core_schema_and_stable_method():
    import core as core_mod

    assert core_mod.RESULT_SCHEMA_VERSION == "v2.3"
    assert "fwd_transfer_per_task" in core_mod.RESULT_KEYS
    assert "backward_transfer" in core_mod.RESULT_KEYS
    assert "peak_memory_mb" in core_mod.RESULT_KEYS


# ═══════════════════════════════════════════════════════════════════════════════
# NeurIPS v2 — NEW LOSS AND METHOD TESTS
# ═══════════════════════════════════════════════════════════════════════════════
def test_ppcl_loss_soft_gradient():
    """Soft-PPCL produces gradients (continuous weighting, no hard partition)."""
    from core import ppcl_loss_soft

    z1 = torch.randn(16, 128, requires_grad=True)
    z2 = torch.randn(16, 128, requires_grad=True)
    loss, er, lc, lu = ppcl_loss_soft(z1, z2, temp=0.5, unif_weight=0.05)
    loss.backward()
    assert z1.grad is not None, "z1 has no gradient"
    assert z2.grad is not None, "z2 has no gradient"
    assert torch.isfinite(loss).item(), f"Non-finite loss: {loss.item()}"
    assert er == 0.0, "erank diagnostic should be 0.0 (fast path)"


def test_byol_no_pred_loss():
    """BYOL without predictor produces finite loss."""
    from core import SSLModel

    model = SSLModel(use_pred=False, use_ema=True, in_size=32, proj_dim=128)
    x = torch.randn(4, 3, 32, 32)
    _, z, _ = model(x)
    tz = model.forward_target(x)
    z_n = torch.nn.functional.normalize(z, dim=1)
    tz_n = torch.nn.functional.normalize(tz.detach(), dim=1)
    loss = 2 - (z_n * tz_n).sum(1).mean()
    assert torch.isfinite(loss).item(), f"Non-finite loss: {loss.item()}"


def test_simclr_pred_loss():
    """SimCLR + Predictor loss is finite and produces gradients."""
    from core import SSLModel, ntxent

    model = SSLModel(use_pred=True, use_ema=False, in_size=32, proj_dim=128)
    x1 = torch.randn(4, 3, 32, 32)
    x2 = torch.randn(4, 3, 32, 32)
    _, z1, p1 = model(x1)
    _, z2, p2 = model(x2)
    loss = 0.5 * (ntxent(p1, z2.detach()) + ntxent(p2, z1.detach()))
    loss.backward()
    assert torch.isfinite(loss).item(), f"Non-finite loss: {loss.item()}"


def test_imagenet100_model_shape():
    """Standard ResNet-18 for 224x224 produces correct output shapes."""
    from core import SSLModel

    model = SSLModel(use_pred=False, use_ema=False, in_size=224,
                     proj_dim=256, proj_hid=2048)
    x = torch.randn(2, 3, 224, 224)
    h, z, _ = model(x)
    assert h.shape == (2, 512), f"Encoder output shape: {h.shape}"
    assert z.shape == (2, 256), f"Projector output shape: {z.shape}"


def test_ppcl_enc_gradient_flow():
    """Encoder-level PPCL applies loss to 512-d encoder features."""
    from core import SSLModel, ppcl_loss, ntxent

    model = SSLModel(use_pred=False, use_ema=False, in_size=32, proj_dim=128)
    x1 = torch.randn(4, 3, 32, 32)
    x2 = torch.randn(4, 3, 32, 32)
    h1, z1, _ = model(x1)
    h2, z2, _ = model(x2)
    loss_enc, _, _, _ = ppcl_loss(h1, h2, reserve=0.10, temp=0.5, unif_weight=0.05)
    loss_proj = ntxent(z1, z2, temp=0.5)
    loss = loss_proj + loss_enc
    loss.backward()
    assert torch.isfinite(loss).item(), f"Non-finite loss: {loss.item()}"


def test_uniformity_loss_numerical_stability():
    """Uniformity loss must not produce NaN/Inf even with large feature distances."""
    import torch.nn.functional as F

    # Case 1: Large distances (exp(-2*d^2) → 0)
    z_far = torch.randn(16, 128) * 100  # Very spread out
    z_far = F.normalize(z_far, dim=1)
    u = torch.pdist(z_far, p=2).pow(2).mul(-2).exp().mean().clamp(min=1e-8).log()
    assert torch.isfinite(u).item(), f"Uniformity NaN/Inf with large distances: {u.item()}"

    # Case 2: Near-identical features (collapsed)
    z_same = torch.ones(16, 128) + torch.randn(16, 128) * 1e-6
    z_same = F.normalize(z_same, dim=1)
    u2 = torch.pdist(z_same, p=2).pow(2).mul(-2).exp().mean().clamp(min=1e-8).log()
    assert torch.isfinite(u2).item(), f"Uniformity NaN/Inf with collapsed features: {u2.item()}"

    # Case 3: Actual PPCL loss with edge-case inputs
    from core import ppcl_loss
    z1 = torch.randn(8, 64) * 50
    z2 = torch.randn(8, 64) * 50
    loss, _, lc, lu = ppcl_loss(z1, z2, reserve=0.10, temp=0.5, unif_weight=0.05)
    assert torch.isfinite(loss).item(), f"PPCL loss non-finite: {loss.item()}"


def test_sslmodel_with_ema_pred_shapes():
    """SSLModel with both EMA and predictor (BYOL config) has correct shapes."""
    from core import SSLModel

    model = SSLModel(use_pred=True, use_ema=True, in_size=32, proj_dim=128)
    x = torch.randn(2, 3, 32, 32)
    h, z, p = model(x)
    zt = model.forward_target(x)
    assert h.shape == (2, 512)
    assert z.shape == (2, 128)
    assert p.shape == (2, 128)
    assert zt.shape == (2, 128)
    # EMA update should not raise
    model.update_ema()

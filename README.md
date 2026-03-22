

# PPCL: Plasticity-Preserving Continual Learning

**Status: Work in progress — targeting NeurIPS 2026**

---

## What is this?

PPCL investigates plasticity collapse in continual self-supervised learning. In standard continual SSL, by Task 5 the encoder is using only 2-3 effective dimensions out of 512. There is no room to absorb new information. PPCL addresses this directly.

---

## The Problem

When training an SSL model sequentially on tasks, representations collapse over time. The model forgets not just old tasks — it loses the capacity to learn new ones. We call this **plasticity collapse**.

This is distinct from catastrophic forgetting. Forgetting means old knowledge is overwritten. Plasticity collapse means the model's ability to learn anything new degrades. Both happen simultaneously in continual SSL.

---

## The Approach

PPCL partitions the projection head output (128 dims) into two subspaces:

- **Stability dimensions (90%)** — trained with standard InfoNCE contrastive loss. These learn discriminative features for current and past tasks.
- **Plasticity reserve dimensions (10%)** — trained with a uniformity loss. These dimensions are kept spread out and active, preserving the model's capacity to absorb future tasks.

**PPCL Loss:**
```
L_PPCL = InfoNCE(z_stability) + λ × Uniformity(z_plasticity)
```

**Two novel contributions:**

**Plasticity Health Score (PHS)** — a leading indicator derived from spectral rank that predicts plasticity collapse 1-3 tasks before it occurs. Allows intervention before performance drops.

**Spectral Plasticity Regulariser (SPR)** — a loss that acts directly on representation geometry to prevent rank collapse before it starts. Preventive rather than reactive.

---

## Key Theoretical Finding

BYOL's EMA momentum is mathematically equivalent to an implicit L2 regularizer on encoder parameters:
```
L_total = L_BYOL + λ_implicit × ||θ - θ_init||²
```

where λ_implicit is proportional to (1-τ). This explains why BYOL outperforms contrastive methods in continual settings — not because of architectural differences, but because of implicit regularization. PPCL makes this tradeoff explicit and tunable.

---

## Built On

This work extends ACE (Adaptive Confidence-Based Encoder Freezing):
[github.com/sharanyakatna/ace](https://github.com/sharanyakatna/ace)

ACE showed that model confidence can serve as a control signal for continual learning. PPCL addresses the limitation ACE revealed: coarse layer-level control with no mechanism to detect representation degradation before it causes forgetting.

---

## Experiments

Benchmarked on:
- CIFAR-100 (10 tasks × 10 classes, 20 tasks × 5 classes)
- TinyImageNet (20 tasks)

18 baseline methods including SimCLR, BYOL, EWC-SSL, LwF-SSL, CaSSLe, DER++, Replay-SSL, VICReg, Barlow Twins, SimSiam.

**Results coming — experiments in progress.**

---

## Author

Sharanya Katna
sharanyakatna13@gmail.com
[github.com/sharanyakatna](https://github.com/sharanyakatna)

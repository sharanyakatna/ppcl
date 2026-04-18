

# PPCL: Plasticity-Preserving Continual Learning

**Status: Active research (targeting NeurIPS 2026**)

## What is this?
PPCL investigates plasticity collapse in continual self-supervised learning. In standard continual SSL, by Task 5 the encoder is using only 2-3 effective dimensions out of 512. There is no room to absorb new information. PPCL addresses this directly.
## The Problem
When training an SSL model sequentially on tasks, representations collapse over time. The model forgets not just old tasks — it loses the capacity to learn new ones. We call this plasticity collapse.
This is distinct from catastrophic forgetting. Forgetting means old knowledge is overwritten. Plasticity collapse means the model's ability to learn anything new degrades. Both happen simultaneously in continual SSL.
## The Approach
PPCL partitions the projection head output (128 dims) into two subspaces:
- **Stability dimensions (80%)** — trained with standard SimCLR contrastive loss. These learn discriminative features for current and past tasks.
- **Plasticity reserve dimensions (10%)** — trained with a uniformity loss. These dimensions are kept spread out and active, preserving the model's capacity to absorb future tasks.
**PPCL Loss:**
$$L_{PPCL}=\text{SimCLR}(z_{stability})+\lambda\times\text{Uniformity}(z_{plasticity})$$
**Two novel contributions:**
- **Plasticity Health Score (PHS)** — a leading indicator derived from spectral rank that predicts plasticity collapse 1-2 tasks before it occurs. Allows intervention before performance drops.
- **Spectral Plasticity Regulariser (SPR)** — acts directly on representation geometry to prevent rank collapse before it starts. Preventive rather than reactive.
## Key Theoretical Finding
BYOL's EMA momentum is mathematically equivalent to an implicit L2 regulariser on encoder parameters:
$$L_{EMA}=L_{task}+L_{stability}\times(\theta-\alpha\times\theta_{target})^2$$
where $\theta_{target}$ is proportional to $\theta_{prev}$. This explains why BYOL outperforms continual methods in continual settings — not because of architectural differences, but because of implicit L2 regularisation. PPCL makes this tradeoff explicit and tunable.This finding has not been directly investigated in the continual SSL literature, to the best of our knowledge.
## Preliminary Baseline Results
The following results are from ACE, the supervised continual learning framework on which PPCL is built. They establish the baseline performance that PPCL extends into the self-supervised setting.
| Method | Avg Accuracy | Forgetting | Std Dev |
|--------|-------------|------------|---------|
| Naive (No Freeze) | 13.1% | 1.1% | 0.4% |
| Fixed Freeze (40%) | 13.2% | 0.7% | 0.3% |
| **ACE (Ours)** | **13.8%** | **0.5%** | **0.2%** |
> These results are for the ACE baseline (supervised continual learning). Full PPCL results on continual SSL benchmarks are in progress.
## Built On
This work extends ACE (Adaptive Confidence-Based Encoder Freezing): [github.com/sharanyakatna/ace](https://github.com/sharanyakatna/ace)
ACE showed that model confidence can serve as a control signal for continual learning. PPCL addresses the limitation ACE revealed: coarse layer-level control with no mechanism to detect representation degradation before it causes forgetting.
## Experiments
**Benchmarks:**
- CIFAR-100 (10 tasks × 10 classes, 20 tasks × 5 classes)
- TinyImageNet (20 tasks)
**10 baseline methods including SimCLR, BYOL, EWC, LwF, GDI, CaSSLe, DNN+, Replay-SSL, VICReg, Barlow Twins, SimSiam.**
Results coming — paper in progress.

## Author

Sharanya Katna
sharanyakatna13@gmail.com
[github.com/sharanyakatna](https://github.com/sharanyakatna)

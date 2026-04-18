# PPCL: Plasticity-Preserving Contrastive Learning

Continual self-supervised learning that prevents projector collapse across tasks.

## Idea

In continual SSL, some projector dimensions lose variance over time (they "collapse"). PPCL splits the projector into two parts each batch:

- **Plasticity reserve** (low-variance dims): uniformity loss revives spread here
- **Active subspace** (high-variance dims): standard NT-Xent contrastive objective

The split is determined by a top-k on batch variance (stop-grad). `ppcl_stable` uses an EMA of variance for more stable routing across batches.

## Setup

```bash
pip install -r requirements.txt
```

Tested on PyTorch 2.2.2 + CUDA. CPU works but is slow for the full benchmark.

## Usage

```python
# In Colab or a script
%run core.py
%run experiments.py

# Quick sanity check
phase1()  # ~1 hr: PPCL vs SimCLR blocker gate

# Full benchmark
phase2(best_r=0.10)  # ~6 hrs: 18 methods x 3 seeds
```

## Files

- `core.py` — models, losses, training loop, evaluation
- `ppcl_subspace.py` — EMA variance routing for PPCL-Stable
- `ppcl_theory.py` — weight-space helpers for BYOL ablations
- `experiments.py` — all experiment phases
- `test_ppcl_metrics.py` — unit tests (no GPU required)

## Tests

```bash
pytest test_ppcl_metrics.py -v
```

## Methods included

SimCLR, BYOL, Barlow Twins, VICReg, SimSiam, EWC-SSL, LwF-SSL, Replay-SSL,
CaSSLe, DER-SSL, LUMP, PNR, PPCL (+ stable/soft/adaptive/enc/mom variants),
supervised baseline, oracle (joint training).

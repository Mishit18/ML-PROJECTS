# 100/100 Quality Checklist

This project is considered resume-ready only when all boxes below are complete.

## Implementation

- [x] Forward noising process implemented from closed form.
- [x] Linear and cosine schedules implemented.
- [x] Residual U-Net implemented from PyTorch primitives.
- [x] Sinusoidal timestep embeddings injected into residual blocks.
- [x] Multi-head self-attention at CIFAR-appropriate resolutions.
- [x] DDPM ancestral sampler implemented.
- [x] DDIM deterministic sampler implemented.
- [x] EMA weights implemented.
- [x] AMP training implemented.
- [x] Config validation and tests added.

## Training

- [x] Main RTX 4060 run launched.
- [ ] Main run completed to target training budget.
- [ ] Final EMA checkpoint saved.
- [ ] Final 64-image sample grid saved.
- [ ] One-batch overfit sanity check recorded.

## Evaluation

- [ ] 50k-sample FID computed for EMA DDIM-50.
- [ ] 50k-sample Inception Score computed for EMA DDIM-50.
- [ ] Raw-vs-EMA evaluation completed.
- [ ] DDPM-1000 vs DDIM-50 speed comparison completed.
- [ ] DDIM 25/50/100 sampler benchmark completed.

## Ablations

- [ ] Cosine vs linear schedule.
- [ ] Medium vs small U-Net.
- [ ] EMA vs raw weights.
- [ ] DDPM 1000 vs DDIM 50.

## Reporting

- [ ] `results/REPORT.md` generated with final numbers.
- [ ] `results/final_metrics.csv` populated.
- [ ] Resume bullets filled with measured values only.
- [ ] README updated with final sample grid and metrics.

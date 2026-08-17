# Experiment Report

## Current Main Run

| Field | Value |
| --- | --- |
| Config | `configs/cifar10_rtx4060_best.yaml` |
| Dataset | CIFAR-10 train split, random horizontal flip |
| Objective | epsilon-prediction MSE |
| Schedule | cosine |
| Parameters | 71.03M |
| Target train steps | 300,000 optimizer steps |
| Current logged step | 300,000 |
| Current logged epoch | 769 |
| Latest logged loss | 0.040992 |
| Final evaluated checkpoint | `runs/cifar10_rtx4060_best/checkpoint_0770.pt` |
| Status | Training and EMA DDIM-50 evaluation complete |

## Final Metrics

The final evaluation used the EMA checkpoint, DDIM with 50 denoising steps, and 50,000 generated samples.

| Run | Weights | Sampler | Steps | Samples | FID | IS mean | IS std | Samples/sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| best cosine | EMA | DDIM | 50 | 50,000 | 10.0958 | 8.7801 | 0.0958 | 9.999 |
| best cosine | EMA | DDPM | 1000 | 50,000 | pending | pending | pending | pending |
| best cosine | raw | DDIM | 50 | 50,000 | pending | pending | pending | pending |
| linear ablation | EMA | DDIM | 50 | 50,000 | pending | pending | pending | pending |
| small model | EMA | DDIM | 50 | 50,000 | pending | pending | pending | pending |

## Reproduction Commands

Generate qualitative samples:

```powershell
python sample.py --run-dir runs/cifar10_rtx4060_best --sampler ddim --ddim-steps 50 --num-samples 64
```

Run final evaluation:

```powershell
python evaluate.py --run-dir runs/cifar10_rtx4060_best --sampler ddim --ddim-steps 50 --weights ema --num-samples 50000
python evaluate.py --run-dir runs/cifar10_rtx4060_best --sampler ddim --ddim-steps 50 --weights raw --num-samples 50000
python benchmark_sampler.py --run-dir runs/cifar10_rtx4060_best --ddim-steps 25 50 100 --include-ddpm
```

## Resume Bullets

- Implemented DDPM from scratch in PyTorch, including closed-form forward
  noising, epsilon-prediction reverse denoising, cosine noise scheduling, EMA
  checkpoints, and DDIM/DDPM samplers.
- Built a 71.03M-parameter residual U-Net with GroupNorm/SiLU blocks,
  sinusoidal timestep conditioning, encoder-decoder skip connections, and
  multi-head attention at 16x16 and 8x8 resolutions.
- Completed 300,000 optimizer steps and evaluated 50,000 EMA DDIM-50 samples,
  achieving FID 10.0958 and Inception Score 8.7801 +/- 0.0958 at approximately
  10 samples per second.

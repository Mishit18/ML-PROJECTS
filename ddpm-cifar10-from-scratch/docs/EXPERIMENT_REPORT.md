# Experiment Report Template

## Main Run

| Field | Value |
| --- | --- |
| Config | `configs/cifar10_rtx4060_best.yaml` |
| Dataset | CIFAR-10 train split, random horizontal flip |
| Objective | epsilon-prediction MSE |
| Schedule | cosine |
| Parameters | fill after `python -c "..."` or from evaluation output |
| Train steps | 300,000 |
| Weights evaluated | EMA |
| Sampler | DDIM 50 and DDPM 1000 |

## Final Metrics

| Run | Weights | Sampler | Steps | Samples | FID | IS mean | IS std | Samples/sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| best cosine | EMA | DDIM | 50 | 50,000 | TBD | TBD | TBD | TBD |
| best cosine | EMA | DDPM | 1000 | 50,000 | TBD | TBD | TBD | TBD |
| best cosine | raw | DDIM | 50 | 50,000 | TBD | TBD | TBD | TBD |
| linear ablation | EMA | DDIM | 50 | 50,000 | TBD | TBD | TBD | TBD |
| small model | EMA | DDIM | 50 | 50,000 | TBD | TBD | TBD | TBD |

## Resume Bullets

Replace placeholders only with measured values.

- Implemented DDPM from scratch in PyTorch, including closed-form forward noising, epsilon-prediction reverse denoising, cosine noise scheduling, EMA checkpoints, and DDIM/DDPM samplers; achieved FID `[X]` on CIFAR-10 using 50k generated samples.
- Built a `[Z]`M-parameter residual U-Net with GroupNorm/SiLU blocks, sinusoidal timestep conditioning, encoder-decoder skip connections, and multi-head attention at `[Y]`; trained for `[N]` optimizer steps with AMP on a single GPU.
- Implemented deterministic DDIM sampling and benchmarked `[25, 50, 100]` denoising-step schedules against 1000-step ancestral DDPM; achieved `[X]`x sampling speedup at FID delta `< [Y]`.
- Ran controlled ablations over noise schedule, EMA weights, sampler type, and model size; cosine schedule reduced FID by `[X]%`, while EMA improved FID by `[Y]%` over raw weights.

## Commands

```powershell
python monitor.py --run-dir runs/cifar10_rtx4060_best
python sample.py --run-dir runs/cifar10_rtx4060_best --sampler ddim --ddim-steps 50 --num-samples 64
python evaluate.py --run-dir runs/cifar10_rtx4060_best --sampler ddim --ddim-steps 50 --weights ema --num-samples 50000
python evaluate.py --run-dir runs/cifar10_rtx4060_best --sampler ddim --ddim-steps 50 --weights raw --num-samples 50000
python benchmark_sampler.py --run-dir runs/cifar10_rtx4060_best --ddim-steps 25 50 100 --include-ddpm
```

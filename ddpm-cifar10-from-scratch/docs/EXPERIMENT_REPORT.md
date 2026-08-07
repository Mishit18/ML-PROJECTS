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
| Current logged step | 106,500 |
| Current logged epoch | 273 |
| Latest logged loss | 0.058331 |
| Latest saved checkpoint | `runs/cifar10_rtx4060_best/checkpoint_0270.pt` |
| Latest sample grid | `runs/cifar10_rtx4060_best/samples/epoch_0270_ddim50.png` |
| Status | Training incomplete; resume after QLoRA releases GPU |

## Final Metrics

Final FID and Inception Score are intentionally not filled yet because the main
training run has not completed and full 50,000-sample evaluation has not been
run.

| Run | Weights | Sampler | Steps | Samples | FID | IS mean | IS std | Samples/sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| best cosine | EMA | DDIM | 50 | 50,000 | pending | pending | pending | pending |
| best cosine | EMA | DDPM | 1000 | 50,000 | pending | pending | pending | pending |
| best cosine | raw | DDIM | 50 | 50,000 | pending | pending | pending | pending |
| linear ablation | EMA | DDIM | 50 | 50,000 | pending | pending | pending | pending |
| small model | EMA | DDIM | 50 | 50,000 | pending | pending | pending | pending |

## Commands To Complete

Resume training after the active QLoRA process finishes:

```powershell
python train.py --config configs/cifar10_rtx4060_best.yaml --resume runs/cifar10_rtx4060_best/last.pt
```

Monitor:

```powershell
python monitor.py --run-dir runs/cifar10_rtx4060_best
```

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

## Resume Bullets After Final Evaluation

Replace only the pending metrics with measured values.

- Implemented DDPM from scratch in PyTorch, including closed-form forward
  noising, epsilon-prediction reverse denoising, cosine noise scheduling, EMA
  checkpoints, and DDIM/DDPM samplers.
- Built a 71.03M-parameter residual U-Net with GroupNorm/SiLU blocks,
  sinusoidal timestep conditioning, encoder-decoder skip connections, and
  multi-head attention at 16x16 and 8x8 resolutions.
- Trained the main CIFAR-10 run to 106,500 / 300,000 optimizer steps before GPU
  handoff; latest logged loss was 0.058331 and checkpoint/sample artifacts were
  generated every 5 epochs.
- After final evaluation, report FID/Inception Score using 50,000 generated
  samples, sampler type, denoising steps, EMA/raw weights, checkpoint, and seed.

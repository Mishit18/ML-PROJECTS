# Training Status

Last checked: 2026-08-17.

## DDPM CIFAR-10 Main Run

| Item | Value |
| --- | --- |
| Run directory | `runs/cifar10_rtx4060_best` |
| Config | `configs/cifar10_rtx4060_best.yaml` |
| Model size | 71.03M parameters |
| Target steps | 300,000 |
| Final logged step | 300,000 |
| Final logged epoch | 769 |
| Final logged loss | 0.040992 |
| Evaluated checkpoint | `checkpoint_0770.pt` |
| Evaluation | 50,000 EMA DDIM-50 samples |
| FID | 10.0958 |
| Inception Score | 8.7801 +/- 0.0958 |
| Verification | `python -m pytest -q` passed, 4 tests |

## Status

Training and the primary EMA DDIM-50 evaluation are complete. Raw-weight and alternative-sampler ablations remain optional extensions.

## Resume Command

```powershell
python train.py --config configs/cifar10_rtx4060_best.yaml --resume runs/cifar10_rtx4060_best/last.pt
```

## Remaining Optional Ablations

- Benchmark DDIM 25/50/100 steps and optionally DDPM 1000 steps.
- Compare final EMA and raw weights under the same evaluator.

# Training Status

Last checked: 2026-08-08 02:58 IST.

## DDPM CIFAR-10 Main Run

| Item | Value |
| --- | --- |
| Run directory | `runs/cifar10_rtx4060_best` |
| Config | `configs/cifar10_rtx4060_best.yaml` |
| Model size | 71.03M parameters |
| Target steps | 300,000 |
| Current logged step | 106,500 |
| Current logged epoch | 273 |
| Last logged loss | 0.058331 |
| Last logged LR | 0.00014795 |
| Latest checkpoint | `checkpoint_0270.pt` |
| Latest sample grid | `epoch_0270_ddim50.png` |
| Verification | `python -m pytest -q` passed, 4 tests |

## Why Training Is Paused

The GPU is currently occupied by the QLoRA run:

```text
python -u -m src.train --config configs/t4_mistral_alpaca_qlora.json --resume-from-checkpoint outputs/mistral7b-alpaca-cleaned-qlora-r8/checkpoint-600
```

Do not resume DDPM while QLoRA is using the RTX 4060, because both jobs are
GPU-memory intensive.

## Resume Command

```powershell
python train.py --config configs/cifar10_rtx4060_best.yaml --resume runs/cifar10_rtx4060_best/last.pt
```

## Completion Criteria

- Reach 300,000 optimizer steps or intentionally stop with a documented compute
  budget decision.
- Generate at least one 64-image DDIM sample grid from the final EMA checkpoint.
- Run FID/Inception Score evaluation with the chosen sample count.
- Benchmark DDIM 25/50/100 steps and optionally DDPM 1000 steps.
- Replace pending values in `docs/EXPERIMENT_REPORT.md` with measured metrics.

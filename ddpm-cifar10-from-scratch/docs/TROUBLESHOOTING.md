# Troubleshooting

## CIFAR-10 Download Hangs

If `torchvision.datasets.CIFAR10(download=True)` is slow, manually download and extract:

```powershell
New-Item -ItemType Directory -Force -Path downloads,data
curl.exe -L -o downloads\cifar-10-python.tar.gz https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz
tar -xzf downloads\cifar-10-python.tar.gz -C data
```

The loader checks for `data/cifar-10-batches-py` before asking torchvision to download.

## Resume Training

```powershell
python train.py --config configs/cifar10_rtx4060_best.yaml --resume runs/cifar10_rtx4060_best/last.pt
```

Checkpoints store model weights, EMA weights, optimizer state, AMP scaler state, scheduler state, epoch, step, and config.

## Out of Memory

Use one of these changes:

- Reduce `batch_size` from 64 to 48 or 32.
- Increase `grad_accum_steps` to preserve effective batch size.
- Use `configs/abl_cifar10_small_cosine.yaml`.
- Reduce attention to only `[16]`.

## FID Takes Too Long

For iteration, run 10k samples:

```powershell
python evaluate.py --run-dir runs/cifar10_rtx4060_best --sampler ddim --ddim-steps 50 --num-samples 10000
```

For final reporting, always use 50k samples.

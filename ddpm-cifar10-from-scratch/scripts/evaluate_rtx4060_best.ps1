$ErrorActionPreference = "Stop"
python sample.py --run-dir runs/cifar10_rtx4060_best --sampler ddim --ddim-steps 50 --num-samples 64
python evaluate.py --run-dir runs/cifar10_rtx4060_best --sampler ddim --ddim-steps 50 --weights ema --num-samples 50000 --batch-size 128
python evaluate.py --run-dir runs/cifar10_rtx4060_best --sampler ddim --ddim-steps 50 --weights raw --num-samples 50000 --batch-size 128

$ErrorActionPreference = "Stop"
python benchmark_sampler.py --run-dir runs/cifar10_rtx4060_best --ddim-steps 25 50 100 --include-ddpm

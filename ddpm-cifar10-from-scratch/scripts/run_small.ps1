$ErrorActionPreference = "Stop"
python -m pip install -r requirements.txt
python train.py --config configs/cifar10_small.yaml

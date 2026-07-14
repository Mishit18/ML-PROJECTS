import argparse
import time

import torch
from torch.amp import GradScaler, autocast
from tqdm import trange

from train import build_model_and_diffusion, configure_torch
from src.ddpm.config import validate_config
from src.ddpm.data import make_cifar10_loader
from src.ddpm.utils import load_config, seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cifar10_small.yaml")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    config = load_config(args.config)
    validate_config(config)
    seed_everything(config["seed"])
    configure_torch(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, diffusion = build_model_and_diffusion(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scaler = GradScaler("cuda", enabled=config["train"]["amp"] and device.type == "cuda")
    loader = make_cifar10_loader(config["data_dir"], config["batch_size"], config["num_workers"], train=True)
    images, _ = next(iter(loader))
    images = images.to(device, non_blocking=True)

    start = time.time()
    first_loss = None
    last_loss = None
    for _ in trange(args.steps, desc="overfit one batch"):
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=config["train"]["amp"] and device.type == "cuda"):
            loss = diffusion.training_losses(model, images)
        if first_loss is None:
            first_loss = loss.item()
        last_loss = loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    elapsed = time.time() - start
    ratio = last_loss / first_loss
    print(f"first_loss={first_loss:.6f} last_loss={last_loss:.6f} ratio={ratio:.4f} seconds={elapsed:.2f}")
    if ratio > 0.8:
        raise SystemExit("loss did not decrease enough on one-batch overfit check")


if __name__ == "__main__":
    main()

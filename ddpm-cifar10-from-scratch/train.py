import argparse
import time
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.ddpm import GaussianDiffusion, UNet, count_parameters
from src.ddpm.data import make_cifar10_loader
from src.ddpm.ema import EMA
from src.ddpm.utils import ensure_dir, load_config, save_config, save_sample_grid, seed_everything


def build_model_and_diffusion(config: dict, device: torch.device):
    model_cfg = config["model"]
    model = UNet(
        in_channels=config["channels"],
        image_size=config["image_size"],
        base_channels=model_cfg["base_channels"],
        channel_mults=model_cfg["channel_mults"],
        num_res_blocks=model_cfg["num_res_blocks"],
        attention_resolutions=model_cfg["attention_resolutions"],
        num_heads=model_cfg["num_heads"],
        dropout=model_cfg["dropout"],
        time_emb_mult=model_cfg["time_emb_mult"],
    ).to(device)
    diffusion = GaussianDiffusion(**config["diffusion"]).to(device)
    return model, diffusion


def save_checkpoint(path: Path, model, ema, optimizer, scaler, epoch: int, step: int, config: dict) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "step": step,
            "config": config,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cifar10_small.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config["seed"])
    out_dir = ensure_dir(config["out_dir"])
    samples_dir = ensure_dir(out_dir / "samples")
    save_config(config, out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, diffusion = build_model_and_diffusion(config, device)
    ema = EMA(model, decay=config["train"]["ema_decay"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )
    scaler = GradScaler(enabled=config["train"]["amp"] and device.type == "cuda")
    loader = make_cifar10_loader(config["data_dir"], config["batch_size"], config["num_workers"], train=True)

    start_epoch = 0
    step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        ema.ema_model.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        step = ckpt["step"]

    print(f"device={device} parameters={count_parameters(model) / 1e6:.2f}M out_dir={out_dir}")
    log_path = out_dir / "train_log.csv"
    if not log_path.exists():
        log_path.write_text("step,epoch,loss,lr,seconds\n", encoding="utf-8")

    for epoch in range(start_epoch, config["train"]["epochs"]):
        model.train()
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        epoch_start = time.time()
        for images, _ in pbar:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=config["train"]["amp"] and device.type == "cuda"):
                loss = diffusion.training_losses(model, images)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            if step % config["train"]["log_every"] == 0:
                elapsed = time.time() - epoch_start
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"{step},{epoch},{loss.item():.6f},{optimizer.param_groups[0]['lr']:.8f},{elapsed:.2f}\n")

        if (epoch + 1) % config["train"]["sample_every"] == 0:
            n = config["train"]["sample_count"]
            samples = diffusion.sample_ddim(
                ema.ema_model,
                (n, config["channels"], config["image_size"], config["image_size"]),
                device,
                steps=50,
            )
            save_sample_grid(samples, samples_dir / f"epoch_{epoch + 1:04d}_ddim50.png")

        if (epoch + 1) % config["train"]["save_every"] == 0 or epoch == config["train"]["epochs"] - 1:
            save_checkpoint(out_dir / f"checkpoint_{epoch + 1:04d}.pt", model, ema, optimizer, scaler, epoch, step, config)


if __name__ == "__main__":
    main()

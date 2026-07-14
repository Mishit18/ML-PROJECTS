import argparse
import math
import time
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
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


def configure_torch(config: dict) -> None:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = config["train"].get("tf32", True)
    torch.backends.cudnn.allow_tf32 = config["train"].get("tf32", True)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def make_scheduler(optimizer, config: dict, steps_per_epoch: int):
    train_cfg = config["train"]
    total_steps = train_cfg.get("max_steps") or (train_cfg["epochs"] * steps_per_epoch)
    warmup_steps = train_cfg.get("warmup_steps", 500)
    min_lr_ratio = train_cfg.get("min_lr_ratio", 0.05)

    def lr_lambda(step: int):
        if step < warmup_steps:
            return max(1, step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda), total_steps


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
    configure_torch(config)
    out_dir = ensure_dir(config["out_dir"])
    samples_dir = ensure_dir(out_dir / "samples")
    save_config(config, out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, diffusion = build_model_and_diffusion(config, device)
    if config["train"].get("channels_last", False):
        model = model.to(memory_format=torch.channels_last)
    ema = EMA(model, decay=config["train"]["ema_decay"]).to(device)
    if config["train"].get("channels_last", False):
        ema.ema_model = ema.ema_model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )
    scaler = GradScaler("cuda", enabled=config["train"]["amp"] and device.type == "cuda")
    loader = make_cifar10_loader(config["data_dir"], config["batch_size"], config["num_workers"], train=True)
    grad_accum_steps = config["train"].get("grad_accum_steps", 1)
    scheduler, total_steps = make_scheduler(optimizer, config, len(loader) // grad_accum_steps)

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
        for _ in range(step):
            scheduler.step()

    print(
        f"device={device} parameters={count_parameters(model) / 1e6:.2f}M "
        f"batch={config['batch_size']} accum={grad_accum_steps} total_steps={total_steps} out_dir={out_dir}"
    )
    log_path = out_dir / "train_log.csv"
    if not log_path.exists():
        log_path.write_text("step,epoch,loss,lr,seconds\n", encoding="utf-8")

    for epoch in range(start_epoch, config["train"]["epochs"]):
        model.train()
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        epoch_start = time.time()
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            if config["train"].get("channels_last", False):
                images = images.to(memory_format=torch.channels_last)
            with autocast("cuda", enabled=config["train"]["amp"] and device.type == "cuda"):
                loss = diffusion.training_losses(model, images)
                scaled_loss = loss / grad_accum_steps
            scaler.scale(scaled_loss).backward()

            pbar.set_postfix(loss=f"{loss.item():.4f}")
            if (batch_idx + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                ema.update(model)

                step += 1
                if step % config["train"]["log_every"] == 0:
                    elapsed = time.time() - epoch_start
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"{step},{epoch},{loss.item():.6f},{optimizer.param_groups[0]['lr']:.8f},{elapsed:.2f}\n")
                if step >= total_steps:
                    break

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
            save_checkpoint(out_dir / "last.pt", model, ema, optimizer, scaler, epoch, step, config)

        if step >= total_steps:
            break


if __name__ == "__main__":
    main()

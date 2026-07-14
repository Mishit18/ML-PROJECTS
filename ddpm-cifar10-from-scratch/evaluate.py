import argparse
from pathlib import Path

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from tqdm import tqdm

from train import build_model_and_diffusion
from src.ddpm.data import make_cifar10_loader
from src.ddpm.utils import denorm, latest_checkpoint


def to_uint8(x: torch.Tensor) -> torch.Tensor:
    return (denorm(x) * 255).round().to(torch.uint8)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddim")
    parser.add_argument("--ddim-steps", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(args.run_dir)
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]
    model, diffusion = build_model_and_diffusion(config, device)
    model.load_state_dict(ckpt["ema"])
    model.eval()

    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    inception = InceptionScore(normalize=False).to(device)

    real_loader = make_cifar10_loader(config["data_dir"], args.batch_size, config["num_workers"], train=False)
    seen_real = 0
    for real, _ in tqdm(real_loader, desc="real stats"):
        real = real.to(device)
        fid.update(to_uint8(real), real=True)
        seen_real += real.shape[0]
        if seen_real >= args.num_samples:
            break

    generated = 0
    for _ in tqdm(range(0, args.num_samples, args.batch_size), desc="generated stats"):
        bsz = min(args.batch_size, args.num_samples - generated)
        shape = (bsz, config["channels"], config["image_size"], config["image_size"])
        if args.sampler == "ddpm":
            samples = diffusion.sample_ddpm(model, shape, device)
        else:
            samples = diffusion.sample_ddim(model, shape, device, steps=args.ddim_steps)
        images = to_uint8(samples)
        fid.update(images, real=False)
        inception.update(images)
        generated += bsz

    is_mean, is_std = inception.compute()
    print(f"FID: {fid.compute().item():.4f}")
    print(f"IS: {is_mean.item():.4f} +/- {is_std.item():.4f}")


if __name__ == "__main__":
    main()

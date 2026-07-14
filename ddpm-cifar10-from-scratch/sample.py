import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from train import build_model_and_diffusion
from src.ddpm.utils import denorm, ensure_dir, latest_checkpoint, save_sample_grid
from torchvision.utils import save_image


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddim")
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--weights", choices=["ema", "raw"], default="ema")
    parser.add_argument("--save-images", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(args.run_dir)
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]
    model, diffusion = build_model_and_diffusion(config, device)
    weight_key = "ema" if args.weights == "ema" and "ema" in ckpt else "model"
    model.load_state_dict(ckpt[weight_key])
    model.eval()

    out_dir = ensure_dir(Path(args.run_dir) / f"samples_{args.sampler}_{args.weights}")
    generated = 0
    first_batch = None
    start = time.time()
    for batch_start in tqdm(range(0, args.num_samples, args.batch_size), desc="sampling"):
        bsz = min(args.batch_size, args.num_samples - batch_start)
        shape = (bsz, config["channels"], config["image_size"], config["image_size"])
        if args.sampler == "ddpm":
            samples = diffusion.sample_ddpm(model, shape, device)
        else:
            samples = diffusion.sample_ddim(model, shape, device, steps=args.ddim_steps, eta=args.eta)
        if first_batch is None:
            first_batch = samples.detach().cpu()
        if args.save_images:
            for i, image in enumerate(denorm(samples).cpu()):
                save_image(image, out_dir / f"{generated + i:06d}.png")
        generated += bsz

    elapsed = time.time() - start
    save_sample_grid(first_batch[:64], out_dir / "grid.png")
    metadata = {
        "checkpoint": str(ckpt_path),
        "weights": weight_key,
        "sampler": args.sampler,
        "ddim_steps": args.ddim_steps if args.sampler == "ddim" else config["diffusion"]["timesteps"],
        "eta": args.eta,
        "num_samples": generated,
        "seconds": elapsed,
        "samples_per_second": generated / elapsed,
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"saved grid to {out_dir / 'grid.png'}")
    print(f"sampling speed: {metadata['samples_per_second']:.2f} samples/sec")
    if args.save_images:
        print(f"saved {generated} images to {out_dir}")


if __name__ == "__main__":
    main()

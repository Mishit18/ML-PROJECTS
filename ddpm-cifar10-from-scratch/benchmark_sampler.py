import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from train import build_model_and_diffusion
from src.ddpm.utils import latest_checkpoint


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--weights", choices=["ema", "raw"], default="ema")
    parser.add_argument("--ddim-steps", nargs="+", type=int, default=[25, 50, 100])
    parser.add_argument("--include-ddpm", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(args.run_dir)
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]
    model, diffusion = build_model_and_diffusion(config, device)
    weight_key = "ema" if args.weights == "ema" and "ema" in ckpt else "model"
    model.load_state_dict(ckpt[weight_key])
    model.eval()
    shape = (args.batch_size, config["channels"], config["image_size"], config["image_size"])

    rows = []
    for steps in args.ddim_steps:
        diffusion.sample_ddim(model, shape, device, steps=steps)
        throughputs = []
        for _ in range(args.repeats):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            diffusion.sample_ddim(model, shape, device, steps=steps)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            throughputs.append(args.batch_size / elapsed)
        rows.append({
            "sampler": "ddim",
            "weights": weight_key,
            "steps": steps,
            "batch_size": args.batch_size,
            "repeats": args.repeats,
            "samples_per_second_mean": float(np.mean(throughputs)),
            "samples_per_second_std": float(np.std(throughputs, ddof=1)) if len(throughputs) > 1 else 0.0,
            "samples_per_second_runs": throughputs,
        })

    if args.include_ddpm:
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        diffusion.sample_ddpm(model, shape, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start
        rows.append(
            {
                "sampler": "ddpm",
                "weights": weight_key,
                "steps": config["diffusion"]["timesteps"],
                "batch_size": args.batch_size,
                "repeats": 1,
                "samples_per_second_mean": args.batch_size / elapsed,
                "samples_per_second_std": 0.0,
                "samples_per_second_runs": [args.batch_size / elapsed],
            }
        )

    out_dir = Path(args.run_dir) / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sampler_benchmark_{weight_key}.json"
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for row in rows:
        print(
            f"{row['sampler']} {row['steps']} steps: "
            f"{row['samples_per_second_mean']:.2f} +/- {row['samples_per_second_std']:.2f} samples/sec"
        )
    print(f"saved benchmark to {out_path}")


if __name__ == "__main__":
    main()

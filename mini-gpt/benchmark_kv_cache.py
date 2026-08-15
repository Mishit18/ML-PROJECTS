"""Benchmark autoregressive generation with and without KV cache."""

import argparse
import json
import math
import os
import sys
import time
from statistics import median

import torch
import yaml

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.gpt import GPT, GPTConfig
from model.utils import get_device, set_seed
from tokenizer.tokenizer import create_tokenizer


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def load_model(config_path, checkpoint_path, device):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model = GPT(GPTConfig(**config["model"])).to(device)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


@torch.no_grad()
def time_generation(model, input_ids, generated_tokens, use_cache, repeats, warmup, device):
    times = []
    for i in range(warmup + repeats):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        synchronize(device)
        start = time.perf_counter()
        model.generate(
            input_ids.clone(),
            max_new_tokens=generated_tokens,
            temperature=1.0,
            top_k=1,
            top_p=None,
            use_cache=use_cache,
        )
        synchronize(device)
        elapsed = time.perf_counter() - start
        if i >= warmup:
            times.append(elapsed)
    p50 = median(times)
    p95 = sorted(times)[max(0, math.ceil(0.95 * len(times)) - 1)]
    return {
        "seconds_p50": p50,
        "seconds_p95": p95,
        "tokens_per_sec_p50": generated_tokens / p50 if p50 > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark KV cache speedup")
    parser.add_argument("--config", default="configs/small.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--lengths", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", default="reports/kv_cache_benchmark.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    tokenizer = create_tokenizer()
    model, config = load_model(args.config, args.checkpoint, device)
    input_ids = torch.tensor([tokenizer.encode(args.prompt)], device=device)

    results = []
    for generated_tokens in args.lengths:
        no_cache = time_generation(model, input_ids, generated_tokens, False, args.repeats, args.warmup, device)
        cache = time_generation(model, input_ids, generated_tokens, True, args.repeats, args.warmup, device)
        results.append({
            "generated_tokens": generated_tokens,
            "no_cache": no_cache,
            "kv_cache": cache,
            "speedup_p50": no_cache["seconds_p50"] / cache["seconds_p50"] if cache["seconds_p50"] > 0 else 0.0,
        })

    report = {
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "checkpoint": args.checkpoint or "randomly initialized model; timing only, not quality",
        "config": config["model"],
        "results": results,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

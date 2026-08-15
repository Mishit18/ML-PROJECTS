"""Measure inference throughput, latency, memory, batch impact, and KV-cache growth."""

import argparse
import json
import math
import os
import sys
import time
from statistics import median

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from benchmark_kv_cache import load_model, synchronize
from model.utils import get_device, set_seed
from tokenizer.tokenizer import create_tokenizer


@torch.no_grad()
def measure(model, prompt_ids, batch_size, generated_tokens, use_cache, repeats, device):
    input_ids = prompt_ids.repeat(batch_size, 1)
    times = []
    mem = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        synchronize(device)
        start = time.perf_counter()
        model.generate(input_ids.clone(), max_new_tokens=generated_tokens, top_k=1, use_cache=use_cache)
        synchronize(device)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        mem.append(torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0)
    p50 = median(times)
    p95 = sorted(times)[max(0, math.ceil(0.95 * len(times)) - 1)]
    total_tokens = batch_size * generated_tokens
    cache_bytes = (
        2
        * model.config.num_layers
        * batch_size
        * model.config.num_kv_heads
        * (prompt_ids.size(1) + generated_tokens)
        * (model.config.d_model // model.config.num_heads)
        * torch.finfo(next(model.parameters()).dtype).bits
        // 8
    )
    return {
        "batch_size": batch_size,
        "use_cache": use_cache,
        "generated_tokens": generated_tokens,
        "tokens_per_sec_p50": total_tokens / p50 if p50 > 0 else 0.0,
        "latency_seconds_p50": p50,
        "latency_seconds_p95": p95,
        "peak_gpu_memory_mb": max(mem),
        "estimated_kv_cache_mb": cache_bytes / (1024**2),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Mini-GPT inference")
    parser.add_argument("--config", default="configs/small.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--generated-tokens", type=int, default=128)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", default="reports/inference_optimization.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    tokenizer = create_tokenizer()
    model, config = load_model(args.config, args.checkpoint, device)
    prompt_ids = torch.tensor([tokenizer.encode(args.prompt)], device=device)

    results = []
    for batch_size in args.batch_sizes:
        for use_cache in [False, True]:
            results.append(measure(model, prompt_ids, batch_size, args.generated_tokens, use_cache, args.repeats, device))

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

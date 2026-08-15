"""Matrix benchmark for KV cache latency, memory, batch size, and prompt length."""

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


def make_prompt_ids(tokenizer, prompt_len, device):
    seed_text = (
        "Once upon a time, a small child walked through a bright garden and found a kind friend. "
        "They shared a toy, solved a problem, and went home happy. "
    )
    ids = tokenizer.encode(seed_text, add_special_tokens=False)
    repeated = (ids * ((prompt_len // len(ids)) + 1))[:prompt_len]
    return torch.tensor([repeated], dtype=torch.long, device=device)


@torch.no_grad()
def measure_case(model, prompt_ids, batch_size, generated_tokens, use_cache, repeats, warmup, device):
    input_ids = prompt_ids.repeat(batch_size, 1)
    times = []
    memory = []
    for i in range(warmup + repeats):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        synchronize(device)
        start = time.perf_counter()
        model.generate(input_ids.clone(), max_new_tokens=generated_tokens, top_k=1, use_cache=use_cache)
        synchronize(device)
        elapsed = time.perf_counter() - start
        if i >= warmup:
            times.append(elapsed)
            memory.append(torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0)
    p50 = median(times)
    p95 = sorted(times)[max(0, math.ceil(0.95 * len(times)) - 1)]
    return {
        "latency_seconds_p50": p50,
        "latency_seconds_p95": p95,
        "tokens_per_sec_p50": batch_size * generated_tokens / p50 if p50 else 0.0,
        "peak_gpu_memory_mb": max(memory),
    }


def main():
    parser = argparse.ArgumentParser(description="KV cache matrix benchmark")
    parser.add_argument("--config", default="configs/small_tinystories_context1024.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt-lengths", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--generated-tokens", nargs="+", type=int, default=[128, 256])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", default="reports/kv_cache_matrix.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    tokenizer = create_tokenizer()
    model, config = load_model(args.config, args.checkpoint, device)

    rows = []
    helpful = []
    for prompt_len in args.prompt_lengths:
        prompt_ids = make_prompt_ids(tokenizer, prompt_len, device)
        for gen_tokens in args.generated_tokens:
            if prompt_len + gen_tokens > model.config.max_seq_len:
                rows.append({
                    "prompt_length": prompt_len,
                    "generated_tokens": gen_tokens,
                    "status": f"skipped: prompt+generation exceeds context {model.config.max_seq_len}",
                })
                continue
            for batch_size in args.batch_sizes:
                no_cache = measure_case(model, prompt_ids, batch_size, gen_tokens, False, args.repeats, args.warmup, device)
                cache = measure_case(model, prompt_ids, batch_size, gen_tokens, True, args.repeats, args.warmup, device)
                row = {
                    "prompt_length": prompt_len,
                    "generated_tokens": gen_tokens,
                    "batch_size": batch_size,
                    "no_cache": no_cache,
                    "kv_cache": cache,
                    "p95_latency_speedup": no_cache["latency_seconds_p95"] / cache["latency_seconds_p95"] if cache["latency_seconds_p95"] else 0.0,
                    "peak_memory_reduction_mb": no_cache["peak_gpu_memory_mb"] - cache["peak_gpu_memory_mb"],
                    "status": "measured",
                }
                rows.append(row)
                if row["p95_latency_speedup"] > 1.0 or row["peak_memory_reduction_mb"] > 0:
                    helpful.append(row)

    report = {
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "checkpoint": args.checkpoint,
        "config": config["model"],
        "rows": rows,
        "helpful_cases": helpful,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json

import torch

from src.config import load_config
from src.model import count_parameters, load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check CUDA and model loading before training.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    print("cuda_available:", torch.cuda.is_available(), flush=True)
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0), flush=True)
        free, total = torch.cuda.mem_get_info(0)
        print(
            "memory_gb:",
            {"free": round(free / 1024**3, 2), "total": round(total / 1024**3, 2)},
            flush=True,
        )

    print("loading_model:", config.model_name, flush=True)
    model, tokenizer = load_model_and_tokenizer(config)
    print("loaded_tokenizer_vocab:", len(tokenizer), flush=True)
    print("parameter_counts:", json.dumps(count_parameters(model), indent=2), flush=True)
    if torch.cuda.is_available():
        print(
            "max_allocated_gb:",
            round(torch.cuda.max_memory_allocated(0) / 1024**3, 2),
            flush=True,
        )


if __name__ == "__main__":
    main()

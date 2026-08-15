"""Generate qualitative samples from a trained Mini-GPT checkpoint."""

import argparse
import json
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference.text_generation import load_model
from model.utils import get_device, set_seed
from tokenizer.tokenizer import create_tokenizer


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Generate sample completions")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts", nargs="+", default=["Once upon a time", "Lily found a small"])
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--output", default="reports/generation_samples.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    tokenizer = create_tokenizer()
    model = load_model(args.checkpoint, device)

    samples = []
    for prompt in args.prompts:
        input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            use_cache=True,
        )
        samples.append({
            "prompt": prompt,
            "completion": tokenizer.decode(output_ids[0].tolist()),
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
        })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"checkpoint": args.checkpoint, "samples": samples}, f, indent=2)
    print(json.dumps({"checkpoint": args.checkpoint, "samples": samples}, indent=2))


if __name__ == "__main__":
    main()

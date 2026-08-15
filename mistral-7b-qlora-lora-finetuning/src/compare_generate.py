from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from src.evaluate import load_eval_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate side-by-side base and adapter outputs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", default="reports/generation_comparison.json")
    parser.add_argument("--max-new-tokens", type=int, default=220)
    return parser.parse_args()


@torch.inference_mode()
def generate_one(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text[len(prompt) :].strip()


def main() -> None:
    args = parse_args()
    with open(args.prompts, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    _, base_model, base_tokenizer = load_eval_model(args.config, None)
    base_outputs = [
        generate_one(base_model, base_tokenizer, prompt, args.max_new_tokens) for prompt in prompts
    ]
    del base_tokenizer
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    _, tuned_model, tuned_tokenizer = load_eval_model(args.config, args.adapter_dir)
    tuned_outputs = [
        generate_one(tuned_model, tuned_tokenizer, prompt, args.max_new_tokens) for prompt in prompts
    ]

    rows = [
        {"prompt": prompt, "base": base, "fine_tuned": tuned}
        for prompt, base, tuned in zip(prompts, base_outputs, tuned_outputs, strict=True)
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

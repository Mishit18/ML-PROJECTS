from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.evaluate import load_eval_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate outputs for a list of prompts.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    with open(args.prompts, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    _, model, tokenizer = load_eval_model(args.config, args.adapter_dir)
    rows: list[dict[str, str]] = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        rows.append({"prompt": prompt, "completion": text[len(prompt) :].strip()})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

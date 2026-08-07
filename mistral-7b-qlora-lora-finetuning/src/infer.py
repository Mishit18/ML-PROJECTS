from __future__ import annotations

import argparse

import torch

from src.evaluate import load_eval_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run adapter inference.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    _, model, tokenizer = load_eval_model(args.config, args.adapter_dir)
    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()

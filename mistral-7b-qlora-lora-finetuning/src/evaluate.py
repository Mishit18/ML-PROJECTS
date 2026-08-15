from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling

from src.config import load_config
from src.data import load_and_format_dataset, tokenize_for_lm
from src.model import build_bnb_config, resolve_torch_dtype


def load_eval_model(config_path: str, adapter_dir: str | None = None):
    config = load_config(config_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir or config.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=build_bnb_config(config),
        torch_dtype=resolve_torch_dtype(config.bnb_4bit_compute_dtype),
        device_map="auto",
    )
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return config, model, tokenizer


def compute_perplexity(config_path: str, adapter_dir: str | None, max_eval_samples: int | None) -> dict:
    config, model, tokenizer = load_eval_model(config_path, adapter_dir)
    dataset = load_and_format_dataset(config)
    tokenized = tokenize_for_lm(dataset, tokenizer, config.max_seq_length)["validation"]
    if max_eval_samples:
        tokenized = tokenized.select(range(min(max_eval_samples, len(tokenized))))

    dataloader = DataLoader(
        tokenized,
        batch_size=config.per_device_eval_batch_size,
        collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    losses: list[float] = []
    for batch in dataloader:
        batch = {key: value.to(model.device) for key, value in batch.items()}
        with torch.inference_mode():
            outputs = model(**batch)
        losses.append(float(outputs.loss.detach().cpu()))

    loss = sum(losses) / len(losses)
    return {
        "eval_loss": loss,
        "perplexity": math.exp(loss) if loss < 20 else float("inf"),
        "num_eval_samples": len(tokenized),
    }


@torch.inference_mode()
def generate_samples(
    config_path: str,
    adapter_dir: str | None,
    prompts: list[str],
    max_new_tokens: int = 256,
) -> list[dict[str, str]]:
    _, model, tokenizer = load_eval_model(config_path, adapter_dir)
    outputs: list[dict[str, str]] = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        outputs.append({"prompt": prompt, "completion": text[len(prompt) :]})
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate base or adapter model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--max-eval-samples", type=int, default=500)
    parser.add_argument("--sample-prompts", default=None, help="JSON file with a list of prompts.")
    parser.add_argument("--output", default="reports/eval_results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = {"perplexity": compute_perplexity(args.config, args.adapter_dir, args.max_eval_samples)}
    if args.sample_prompts:
        with open(args.sample_prompts, "r", encoding="utf-8") as f:
            prompts = json.load(f)
        results["samples"] = generate_samples(args.config, args.adapter_dir, prompts)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

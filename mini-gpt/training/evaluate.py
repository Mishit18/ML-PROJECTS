"""Evaluate Mini-GPT checkpoints on real text datasets."""

import argparse
import json
import math
import os
import sys
import time

import torch
import yaml
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import create_dataloaders, load_sample_data
from model.gpt import GPT, GPTConfig
from model.utils import get_device, set_seed
from tokenizer.tokenizer import create_tokenizer


@torch.no_grad()
def evaluate_model(model, val_loader, device, bootstrap_samples=1000, seed=42):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    start = time.perf_counter()
    batch_losses = []
    batch_tokens = []

    for batch in tqdm(val_loader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        shifted_labels = labels[:, 1:]
        valid_tokens = (shifted_labels != -100).sum().item()
        if valid_tokens == 0:
            continue
        total_loss += outputs["loss"].item() * valid_tokens
        total_tokens += valid_tokens
        batch_losses.append(outputs["loss"].item())
        batch_tokens.append(valid_tokens)

    elapsed = time.perf_counter() - start
    val_loss = total_loss / max(1, total_tokens)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(batch_losses))
    bootstrap_perplexities = []
    for _ in range(bootstrap_samples):
        sample = rng.choice(indices, size=len(indices), replace=True)
        sampled_tokens = sum(batch_tokens[index] for index in sample)
        sampled_loss = sum(batch_losses[index] * batch_tokens[index] for index in sample) / sampled_tokens
        bootstrap_perplexities.append(math.exp(min(20, sampled_loss)))
    ci_low, ci_high = np.percentile(bootstrap_perplexities, [2.5, 97.5])
    return {
        "val_loss": val_loss,
        "perplexity": math.exp(min(20, val_loss)),
        "tokens_evaluated": total_tokens,
        "tokens_per_sec": total_tokens / elapsed if elapsed > 0 else 0.0,
        "perplexity_bootstrap_ci_95": [float(ci_low), float(ci_high)],
        "bootstrap_samples": bootstrap_samples,
        "validation_batches": len(batch_losses),
    }


def load_checkpoint_model(checkpoint_path, config_path, device):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model_config = GPTConfig(**config["model"])
    model = GPT(model_config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, config


def main():
    parser = argparse.ArgumentParser(description="Evaluate Mini-GPT on validation data")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/small.yaml")
    parser.add_argument("--dataset", default="wikitext-2", choices=["wikitext-2", "tinystories", "openwebtext-small"])
    parser.add_argument("--num-val", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output", default="reports/evaluation.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--allow-synthetic-fallback", action="store_true",
                        help="Allow synthetic fallback only for smoke testing when dataset download fails")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    tokenizer = create_tokenizer()
    model, config = load_checkpoint_model(args.checkpoint, args.config, device)
    _, val_texts, data_metadata = load_sample_data(
        tokenizer,
        num_train=0,
        num_val=args.num_val,
        dataset_name=args.dataset,
        allow_synthetic_fallback=args.allow_synthetic_fallback,
        return_metadata=True,
    )
    _, val_loader = create_dataloaders(
        [],
        val_texts,
        tokenizer,
        batch_size=args.batch_size or config["training"]["batch_size"],
        max_length=config["model"]["max_seq_len"],
    )
    metrics = evaluate_model(
        model, val_loader, device, bootstrap_samples=args.bootstrap_samples, seed=args.seed
    )
    metrics.update({"dataset": args.dataset, "checkpoint": args.checkpoint, "synthetic": data_metadata["synthetic"]})

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

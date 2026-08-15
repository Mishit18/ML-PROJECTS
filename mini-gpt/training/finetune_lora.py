"""Toy instruction fine-tuning with LoRA adapters."""

import argparse
import json
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import TextDataset, collate_fn
from model.gpt import GPT, GPTConfig
from model.lora import add_lora_adapters, trainable_parameter_summary
from model.utils import get_device, set_seed
from tokenizer.tokenizer import create_tokenizer
from training.optimizer import configure_optimizers


TOY_INSTRUCTIONS = [
    ("Summarize: Transformers use attention to route information.", "Transformers use attention to move useful context between tokens."),
    ("Answer: What does KV cache store?", "It stores key and value tensors from earlier tokens for faster decoding."),
    ("Classify sentiment: I loved the clear documentation.", "positive"),
    ("Rewrite formally: this model is tiny but useful.", "This model is compact, but it remains useful."),
    ("Explain: Why use causal masking?", "Causal masking prevents a token from attending to future tokens during next-token prediction."),
    ("Extract keyword: grouped query attention reduces KV heads.", "grouped query attention"),
]


def format_example(instruction, response):
    return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"


def load_model(config_path, checkpoint, device):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model = GPT(GPTConfig(**config["model"])).to(device)
    if checkpoint:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
    return model, config


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune Mini-GPT on a toy instruction dataset")
    parser.add_argument("--config", default="configs/small.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", default="checkpoints/lora_toy.pt")
    parser.add_argument("--report", default="reports/lora_finetune.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    tokenizer = create_tokenizer()
    model, config = load_model(args.config, args.checkpoint, device)
    add_lora_adapters(model, rank=args.rank, alpha=args.alpha, dropout=0.05)
    model = model.to(device)
    trainable, total, pct = trainable_parameter_summary(model)

    texts = [format_example(i, r) for i, r in TOY_INSTRUCTIONS]
    dataset = TextDataset(texts, tokenizer, max_length=min(config["model"]["max_seq_len"], 256))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
    )
    optimizer = configure_optimizers(model, args.lr, weight_decay=0.0, device_type=device.type)

    losses = []
    model.train()
    for _ in range(args.epochs):
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            outputs["loss"].backward()
            optimizer.step()
            losses.append(outputs["loss"].item())

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": config, "lora_rank": args.rank, "lora_alpha": args.alpha}, args.output)
    report = {
        "dataset": "synthetic toy instruction dataset",
        "synthetic": True,
        "base_checkpoint": args.checkpoint,
        "trainable_params": trainable,
        "total_params_with_lora": total,
        "trainable_percent": pct,
        "final_train_loss": losses[-1] if losses else None,
        "losses": losses,
    }
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

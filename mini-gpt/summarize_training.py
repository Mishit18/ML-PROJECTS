"""Extract a compact training summary from a Mini-GPT checkpoint."""

import argparse
import json
import math
import os
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="Summarize Mini-GPT checkpoint training metadata")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="reports/training_summary.json")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_losses = checkpoint.get("train_losses", [])
    val_metrics = checkpoint.get("val_metrics", [])
    final_val = val_metrics[-1] if val_metrics else {}
    best_val = min(val_metrics, key=lambda row: row.get("val_loss", math.inf)) if val_metrics else {}

    summary = {
        "checkpoint": args.checkpoint,
        "step": checkpoint.get("step"),
        "epoch": checkpoint.get("epoch") + 1 if checkpoint.get("epoch") is not None else None,
        "tokens_trained": checkpoint.get("tokens_trained"),
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_val_loss": final_val.get("val_loss"),
        "final_perplexity": final_val.get("perplexity"),
        "best_val_loss": best_val.get("val_loss"),
        "best_perplexity": best_val.get("perplexity"),
        "num_train_loss_points": len(train_losses),
        "num_val_points": len(val_metrics),
        "synthetic": checkpoint.get("full_config", {}).get("data", {}).get("synthetic", None),
        "dataset": checkpoint.get("full_config", {}).get("data", {}).get("dataset_name", None),
        "run_name": checkpoint.get("full_config", {}).get("training", {}).get("run_name", None),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

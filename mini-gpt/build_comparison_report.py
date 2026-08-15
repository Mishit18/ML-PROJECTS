"""Build baseline and ablation comparison tables from training summaries."""

import argparse
import json
from pathlib import Path


RUNS = [
    ("Baseline GPT block", "training_summary_baseline.json", "Learned positions + LayerNorm + GELU + full MHA"),
    ("Modern decoder", "training_summary_tinystories.json", "RoPE + RMSNorm + SwiGLU + GQA"),
    ("Modern 20M tokens", "training_summary_tinystories_20m.json", "Same modern decoder, longer TinyStories run"),
]

ABLATIONS = [
    ("RoPE vs learned positions", "training_summary_ablate_learned_pos.json", "learned absolute positions"),
    ("SwiGLU vs GELU", "training_summary_ablate_gelu.json", "GELU FFN"),
    ("GQA vs full MHA", "training_summary_ablate_full_mha.json", "full MHA"),
]


def load_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value, digits=4):
    if value is None:
        return "pending"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def row(label, summary, notes):
    return (
        f"| {label} | {fmt(summary.get('dataset') if summary else None)} | "
        f"{fmt(summary.get('tokens_trained') if summary else None)} | "
        f"{fmt(summary.get('final_train_loss') if summary else None)} | "
        f"{fmt(summary.get('final_val_loss') if summary else None)} | "
        f"{fmt(summary.get('final_perplexity') if summary else None, 2)} | {notes} |"
    )


def main():
    parser = argparse.ArgumentParser(description="Build model comparison report")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--output", default="reports/COMPARISON_REPORT.md")
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    lines = [
        "# Mini-GPT Baseline And Ablation Report",
        "",
        "All rows are generated from checkpoint summaries. `pending` means the run/config exists but has not been trained and summarized yet.",
        "",
        "## Baseline vs Modern Decoder",
        "",
        "| Run | Dataset | Tokens trained | Train loss | Val loss | Perplexity | Architecture |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for label, filename, notes in RUNS:
        lines.append(row(label, load_json(reports_dir / filename), notes))

    baseline = load_json(reports_dir / "training_summary_baseline.json")
    modern = load_json(reports_dir / "training_summary_tinystories.json")
    if baseline and modern:
        delta = baseline["final_perplexity"] - modern["final_perplexity"]
        pct = 100.0 * delta / baseline["final_perplexity"]
        lines.extend([
            "",
            f"Modern decoder reduced validation perplexity from {baseline['final_perplexity']:.2f} to {modern['final_perplexity']:.2f} "
            f"at the same small-model parameter budget ({pct:.1f}% relative reduction).",
        ])

    lines.extend([
        "",
        "## Focused Ablations",
        "",
        "| Ablation | Dataset | Tokens trained | Train loss | Val loss | Perplexity | Changed feature |",
        "|---|---|---:|---:|---:|---:|---|",
    ])
    for label, filename, notes in ABLATIONS:
        lines.append(row(label, load_json(reports_dir / filename), notes))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

"""Build a Markdown evidence report from generated Mini-GPT artifacts."""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "pending"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def latest_train_row(csv_path: Path) -> Optional[Dict[str, str]]:
    if not csv_path.exists():
        return None
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    train_rows = [row for row in rows if row.get("split") == "train"]
    return train_rows[-1] if train_rows else None


def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def build_report(reports_dir: Path, experiments_dir: Path, output: Path) -> None:
    evaluation_path = first_existing([reports_dir / "evaluation_tinystories_continue_low_lr.json", reports_dir / "evaluation_tinystories_20m.json", reports_dir / "evaluation_tinystories.json", reports_dir / "evaluation.json", reports_dir / "evaluation_tiny_smoke.json"])
    kv_path = first_existing([reports_dir / "kv_cache_benchmark_tinystories.json", reports_dir / "kv_cache_benchmark.json", reports_dir / "kv_cache_benchmark_tiny_smoke.json"])
    inference_path = first_existing([reports_dir / "inference_optimization_tinystories.json", reports_dir / "inference_optimization.json", reports_dir / "inference_optimization_tiny_smoke.json"])
    train_path = first_existing([experiments_dir / "small_tinystories.csv", experiments_dir / "small_wikitext2.csv", experiments_dir / "tiny_smoke.csv"])

    evaluation = load_json(evaluation_path)
    kv = load_json(kv_path)
    inference = load_json(inference_path)
    lora = load_json(reports_dir / "lora_tiny_smoke.json")
    samples = load_json(first_existing([reports_dir / "generation_samples_tinystories.json", reports_dir / "generation_samples.json"]))
    qualitative = load_json(reports_dir / "qualitative_scores_tinystories_20m.json")
    kv_matrix = load_json(reports_dir / "kv_cache_matrix_tinystories_20m.json")
    training_summary = load_json(first_existing([reports_dir / "training_summary_tinystories_continue_low_lr.json", reports_dir / "training_summary_tinystories_20m.json", reports_dir / "training_summary_tinystories.json", reports_dir / "training_summary.json"]))
    tinystories_20m_summary = load_json(reports_dir / "training_summary_tinystories_20m.json")
    tinystories_20m_eval = load_json(reports_dir / "evaluation_tinystories_20m.json")
    tinystories_5m_summary = load_json(reports_dir / "training_summary_tinystories.json")
    tinystories_5m_eval = load_json(reports_dir / "evaluation_tinystories.json")
    wikitext_summary = load_json(reports_dir / "training_summary.json")
    wikitext_eval = load_json(reports_dir / "evaluation.json")
    train = latest_train_row(train_path)
    run_name = training_summary.get("run_name") if training_summary else train_path.stem
    if training_summary and train_path.stem != run_name:
        train = None

    lines = [
        "# Mini-GPT Evidence Report",
        "",
        "This report is generated from local JSON/CSV artifacts. Final rows use trained-checkpoint artifacts when present; smoke rows are labeled separately.",
        "",
        "## Training And Evaluation",
        "",
        "| Run | Dataset | Synthetic | Train loss | Val loss | Perplexity | Eval tok/s | Tokens trained |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines.append(
        f"| {run_name} | "
        f"{fmt(evaluation.get('dataset') if evaluation else 'pending')} | "
        f"{fmt(evaluation.get('synthetic') if evaluation else 'pending')} | "
        f"{fmt(training_summary.get('final_train_loss') if training_summary else (float(train['train_loss']) if train and train.get('train_loss') else None))} | "
        f"{fmt(evaluation.get('val_loss') if evaluation else None)} | "
        f"{fmt(evaluation.get('perplexity') if evaluation else None, 2)} | "
        f"{fmt(evaluation.get('tokens_per_sec') if evaluation else None, 1)} | "
        f"{fmt(training_summary.get('tokens_trained') if training_summary else None)} |"
    )
    if wikitext_summary and wikitext_eval and run_name != wikitext_summary.get("run_name"):
        lines.append(
            f"| {wikitext_summary.get('run_name')} | "
            f"{fmt(wikitext_eval.get('dataset'))} | "
            f"{fmt(wikitext_eval.get('synthetic'))} | "
            f"{fmt(wikitext_summary.get('final_train_loss'))} | "
            f"{fmt(wikitext_eval.get('val_loss'))} | "
            f"{fmt(wikitext_eval.get('perplexity'), 2)} | "
            f"{fmt(wikitext_eval.get('tokens_per_sec'), 1)} | "
            f"{fmt(wikitext_summary.get('tokens_trained'))} |"
        )
    if tinystories_5m_summary and tinystories_5m_eval and run_name != tinystories_5m_summary.get("run_name"):
        lines.append(
            f"| {tinystories_5m_summary.get('run_name')} | "
            f"{fmt(tinystories_5m_eval.get('dataset'))} | "
            f"{fmt(tinystories_5m_eval.get('synthetic'))} | "
            f"{fmt(tinystories_5m_summary.get('final_train_loss'))} | "
            f"{fmt(tinystories_5m_eval.get('val_loss'))} | "
            f"{fmt(tinystories_5m_eval.get('perplexity'), 2)} | "
            f"{fmt(tinystories_5m_eval.get('tokens_per_sec'), 1)} | "
            f"{fmt(tinystories_5m_summary.get('tokens_trained'))} |"
        )
    if tinystories_20m_summary and tinystories_20m_eval and run_name != tinystories_20m_summary.get("run_name"):
        lines.append(
            f"| {tinystories_20m_summary.get('run_name')} | "
            f"{fmt(tinystories_20m_eval.get('dataset'))} | "
            f"{fmt(tinystories_20m_eval.get('synthetic'))} | "
            f"{fmt(tinystories_20m_summary.get('final_train_loss'))} | "
            f"{fmt(tinystories_20m_eval.get('val_loss'))} | "
            f"{fmt(tinystories_20m_eval.get('perplexity'), 2)} | "
            f"{fmt(tinystories_20m_eval.get('tokens_per_sec'), 1)} | "
            f"{fmt(tinystories_20m_summary.get('tokens_trained'))} |"
        )

    lines.extend(["", "## KV Cache Benchmark", ""])
    lines.extend([
        "| Generated tokens | No-cache tok/s | KV-cache tok/s | Speedup |",
        "|---:|---:|---:|---:|",
    ])
    for row in (kv or {}).get("results", []):
        lines.append(
            f"| {row['generated_tokens']} | "
            f"{fmt(row['no_cache']['tokens_per_sec_p50'], 1)} | "
            f"{fmt(row['kv_cache']['tokens_per_sec_p50'], 1)} | "
            f"{fmt(row['speedup_p50'], 2)}x |"
        )

    lines.extend(["", "## Inference Optimization", ""])
    lines.extend([
        "| Batch | Cache | Tok/s p50 | Latency p50 (s) | Latency p95 (s) | Peak GPU MB | KV cache MB |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in (inference or {}).get("results", []):
        lines.append(
            f"| {row['batch_size']} | {row['use_cache']} | "
            f"{fmt(row['tokens_per_sec_p50'], 1)} | "
            f"{fmt(row['latency_seconds_p50'], 4)} | "
            f"{fmt(row['latency_seconds_p95'], 4)} | "
            f"{fmt(row['peak_gpu_memory_mb'], 1)} | "
            f"{fmt(row['estimated_kv_cache_mb'], 4)} |"
        )

    if kv_matrix:
        lines.extend(["", "## Long-Prompt KV Cache Matrix", ""])
        lines.extend([
            "| Prompt tokens | Batch | Generated tokens | No-cache p95 (s) | KV-cache p95 (s) | Speedup | Memory saved MB |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in kv_matrix.get("rows", []):
            if row.get("status") != "measured":
                continue
            lines.append(
                f"| {row['prompt_length']} | {row['batch_size']} | {row['generated_tokens']} | "
                f"{fmt(row['no_cache']['latency_seconds_p95'], 3)} | "
                f"{fmt(row['kv_cache']['latency_seconds_p95'], 3)} | "
                f"{fmt(row['p95_latency_speedup'], 2)}x | "
                f"{fmt(row['peak_memory_reduction_mb'], 1)} |"
            )

    lines.extend(["", "## LoRA Smoke Fine-Tune", ""])
    lines.extend([
        "| Dataset | Synthetic | Trainable params | Total params | Trainable % | Final train loss |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    if lora:
        lines.append(
            f"| {lora['dataset']} | {lora['synthetic']} | "
            f"{lora['trainable_params']:,} | {lora['total_params_with_lora']:,} | "
            f"{lora['trainable_percent']:.3f}% | {lora['final_train_loss']:.4f} |"
        )
    else:
        lines.append("| pending | pending | pending | pending | pending | pending |")

    if samples:
        lines.extend(["", "## Qualitative Samples", ""])
        for sample in samples.get("samples", [])[:3]:
            completion = sample["completion"].replace("\n", " ")
            lines.append(f"- Prompt `{sample['prompt']}` -> {completion[:240]}")

    if qualitative:
        scores = qualitative.get("mean_scores", {})
        lines.extend(["", "## Qualitative Rubric", ""])
        lines.append(
            f"Automated 50-prompt proxy rubric average: {fmt(scores.get('average'), 2)} / 5 "
            f"(coherence {fmt(scores.get('coherence'), 2)}, repetition {fmt(scores.get('repetition'), 2)}, "
            f"entity consistency {fmt(scores.get('entity_consistency'), 2)}, ending completeness {fmt(scores.get('ending_completeness'), 2)}, "
            f"grammar {fmt(scores.get('grammar'), 2)})."
        )

    lines.extend([
        "",
        "## Honesty Notes",
        "",
        "- Tiny smoke metrics are intentionally labeled as smoke-test results.",
        "- Final resume claims should use generated JSON artifacts and should distinguish best-checkpoint metrics from final-epoch metrics.",
        "- Synthetic data is only used in the LoRA toy adapter smoke run unless explicitly enabled with `--allow-synthetic-fallback`.",
        "- The qualitative rubric is an automated proxy, not a human preference evaluation.",
    ])

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build Mini-GPT evidence report")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--experiments-dir", default="experiments")
    parser.add_argument("--output", default="reports/PROJECT_EVIDENCE.md")
    args = parser.parse_args()
    build_report(Path(args.reports_dir), Path(args.experiments_dir), Path(args.output))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_lifecycle.governance import champion_challenger_report, lifecycle_survival_segments, reject_inference_proxy


def main() -> None:
    outputs = ROOT / "outputs"
    reports = ROOT / "reports"
    scored_path = outputs / "scored_customers.csv"
    metrics_path = outputs / "model_metrics.csv"
    if not scored_path.exists():
        raise FileNotFoundError(f"missing {scored_path}; run scripts/run_pipeline.py first")

    scored = pd.read_csv(scored_path)
    reject = reject_inference_proxy(scored)
    lifecycle = lifecycle_survival_segments(scored)
    reject.to_csv(outputs / "reject_inference_proxy.csv", index=False)
    lifecycle.to_csv(outputs / "lifecycle_survival_segments.csv", index=False)

    summary_lines = [
        "# Credit Governance Evidence Pack",
        "",
        f"- Reject-inference proxy buckets: {len(reject)}",
        f"- Lifecycle segments ranked: {len(lifecycle)}",
        f"- Highest value segment: {lifecycle.iloc[0]['customer_segment']}",
    ]
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        challenger = champion_challenger_report(metrics)
        challenger.to_csv(outputs / "champion_challenger_verdicts.csv", index=False)
        summary_lines.append(f"- Champion model: {challenger.iloc[0]['model']}")

    reports.mkdir(exist_ok=True)
    (reports / "governance_evidence_pack.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


REQUIRED_FILES = [
    "README.md",
    "docs/methodology.md",
    "docs/ats_keywords.md",
    "reports/executive_brief.md",
    "reports/model_card.md",
    "reports/resume_bullets.md",
    "reports/INTERVIEW_CASEBOOK.md",
    "outputs/model_metrics.csv",
    "outputs/scored_customers.csv",
    "outputs/risk_band_policy.csv",
    "outputs/channel_lifecycle.csv",
    "outputs/segment_profitability.csv",
    "outputs/drift_report.csv",
    "outputs/policy_threshold_frontier.csv",
    "outputs/fairness_group_metrics.csv",
    "outputs/adverse_action_reason_codes.csv",
    "outputs/customer_decision_explanations.csv",
    "outputs/real_openml_german_credit_benchmark.csv",
    "outputs/business_policy_simulator.csv",
    "outputs/acquisition_efficiency.csv",
    "outputs/cohort_retention_quality.csv",
    "outputs/ltv_default_segments.csv",
    "outputs/retention_risk_watchlist.csv",
    "outputs/pricing_risk_grid.csv",
    "outputs/collections_prioritization.csv",
    "outputs/risk_band_default_rate.png",
    "outputs/feature_importance.png",
    "outputs/monthly_risk_lifecycle_trend.png",
    "outputs/policy_threshold_frontier.png",
    "outputs/fairness_approval_ratio_watchlist.png",
    "queries/risk_band_policy.sql",
    "queries/channel_lifecycle.sql",
    "queries/acquisition_efficiency.sql",
    "queries/cohort_retention_quality.sql",
    "queries/ltv_default_segments.sql",
    "queries/retention_risk_watchlist.sql",
    "queries/pricing_risk_grid.sql",
    "queries/collections_prioritization.sql",
]


def main() -> int:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).exists()]
    if missing:
        print("Missing required outputs:")
        for path in missing:
            print(f"- {path}")
        return 1

    metrics = pd.read_csv(ROOT / "outputs" / "model_metrics.csv")
    best = metrics.iloc[0]
    checks = {
        "roc_auc_at_least_0_72": best["roc_auc"] >= 0.72,
        "ks_at_least_0_30": best["ks"] >= 0.30,
        "top_decile_recall_at_least_0_25": best["recall_top10"] >= 0.25,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        print("Metric validation failed:")
        for name in failed:
            print(f"- {name}")
        return 1

    scored = pd.read_csv(ROOT / "outputs" / "scored_customers.csv")
    required_cols = {"pd_score", "risk_band", "approve_policy", "risk_adjusted_margin", "default_12m", "churn_6m"}
    if not required_cols.issubset(scored.columns):
        print("Scored output missing required columns")
        return 1
    real = pd.read_csv(ROOT / "outputs" / "real_openml_german_credit_benchmark.csv")
    if real.iloc[0]["records"] < 1000 or real.iloc[0]["roc_auc"] < 0.60:
        print("Real-data benchmark validation failed")
        return 1

    frontier = pd.read_csv(ROOT / "outputs" / "policy_threshold_frontier.csv")
    if frontier.empty or frontier["approval_rate"].max() <= frontier["approval_rate"].min():
        print("Policy frontier validation failed")
        return 1

    fairness = pd.read_csv(ROOT / "outputs" / "fairness_group_metrics.csv")
    if fairness["dimension"].nunique() < 4:
        print("Fairness diagnostics validation failed")
        return 1

    simulator = pd.read_csv(ROOT / "outputs" / "business_policy_simulator.csv")
    if simulator.empty or simulator["risk_adjusted_margin_approved"].max() <= 0:
        print("Business policy simulator validation failed")
        return 1

    print("Project validation passed")
    print(f"Best model: {best['model']} | ROC-AUC {best['roc_auc']:.3f} | KS {best['ks']:.3f} | top-decile recall {best['recall_top10']:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

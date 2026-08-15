from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_lifecycle.data import DataConfig, write_dataset
from credit_lifecycle.explainability import approval_decision_explainer, build_reason_codes
from credit_lifecycle.fairness import fairness_by_group, fairness_summary
from credit_lifecycle.features import infer_feature_spec, temporal_split
from credit_lifecycle.modeling import permutation_explain, train_and_select
from credit_lifecycle.monitoring import drift_report
from credit_lifecycle.policy import select_policy, threshold_frontier
from credit_lifecycle.real_data import run_openml_german_credit
from credit_lifecycle.reporting import write_plots, write_reports
from credit_lifecycle.sql_analytics import run_sql_analytics
from build_casebook import main as build_casebook


def main() -> None:
    data_path = ROOT / "data" / "credit_customer_portfolio.csv"
    df = write_dataset(DataConfig(output_path=data_path))
    train_df, test_df = temporal_split(df, test_months=5)
    spec = infer_feature_spec(df)

    model, metrics, scored = train_and_select(train_df, test_df, spec, ROOT / "artifacts")
    scored.to_csv(ROOT / "outputs" / "scored_customers.csv", index=False)
    metrics.to_csv(ROOT / "outputs" / "model_metrics.csv", index=False)

    importance = permutation_explain(model, test_df, spec, ROOT / "outputs" / "permutation_importance.csv")
    latest_month = scored["vintage_month"].max()
    current_df = df.loc[df["vintage_month"] == latest_month].copy()
    drift = drift_report(train_df, current_df, spec.numeric_features)
    drift.to_csv(ROOT / "outputs" / "drift_report.csv", index=False)
    frontier = threshold_frontier(scored)
    frontier.to_csv(ROOT / "outputs" / "policy_threshold_frontier.csv", index=False)
    policy = select_policy(frontier)
    fairness = fairness_by_group(scored)
    fairness.to_csv(ROOT / "outputs" / "fairness_group_metrics.csv", index=False)
    reasons = build_reason_codes(scored, importance)
    reasons.to_csv(ROOT / "outputs" / "adverse_action_reason_codes.csv", index=False)
    approval_decision_explainer(scored).to_csv(ROOT / "outputs" / "customer_decision_explanations.csv", index=False)
    real_benchmark = run_openml_german_credit(ROOT / "outputs")

    run_sql_analytics(scored, ROOT / "outputs", ROOT / "queries")
    write_plots(scored, importance, ROOT / "outputs", frontier=frontier, fairness=fairness)
    write_reports(
        metrics,
        scored,
        importance,
        drift,
        ROOT / "reports",
        frontier=frontier,
        policy=policy,
        fairness_summary=fairness_summary(fairness),
        real_benchmark=real_benchmark,
    )
    build_casebook()

    print("Credit risk lifecycle pipeline complete")
    print(metrics.head(3).to_string(index=False))
    print(real_benchmark.head(2).to_string(index=False))
    print(f"Selected policy threshold: {policy['pd_threshold']:.2%} | approval {policy['approval_rate']:.2%} | default {policy['approved_default_rate']:.2%}")
    print(f"Scored customers: {len(scored):,}")
    print(f"Reports written to: {ROOT / 'reports'}")


if __name__ == "__main__":
    main()

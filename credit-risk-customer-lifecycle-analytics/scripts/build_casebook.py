from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "reports"


def money(x: float) -> str:
    return f"Rs {x:,.0f}"


def pct(x: float) -> str:
    return f"{x:.2%}"


def build_policy_simulator(scored: pd.DataFrame, frontier: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in frontier.iterrows():
        approved = scored.loc[scored["pd_score"] <= row["pd_threshold"]]
        declined = scored.loc[scored["pd_score"] > row["pd_threshold"]]
        rows.append(
            {
                "pd_threshold": row["pd_threshold"],
                "approval_rate": row["approval_rate"],
                "approved_customers": int(row["approved_customers"]),
                "declined_customers": int(row["declined_customers"]),
                "approved_default_rate": row["approved_default_rate"],
                "declined_default_rate": row["declined_default_rate"],
                "expected_margin_approved": approved["expected_margin_12m"].sum(),
                "risk_adjusted_margin_approved": approved["risk_adjusted_margin"].sum(),
                "expected_defaults_approved": approved["pd_score"].sum(),
                "high_value_declined_customers": int((declined["expected_margin_12m"] >= 3000).sum()),
                "watchlist_share_declined": float((declined["risk_band"].isin(["D: high-risk", "E: severe-risk"])).mean())
                if len(declined)
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    scored = pd.read_csv(OUTPUTS / "scored_customers.csv")
    frontier = pd.read_csv(OUTPUTS / "policy_threshold_frontier.csv")
    metrics = pd.read_csv(OUTPUTS / "model_metrics.csv")
    risk_band = pd.read_csv(OUTPUTS / "risk_band_policy.csv")
    channel = pd.read_csv(OUTPUTS / "acquisition_efficiency.csv")
    retention = pd.read_csv(OUTPUTS / "retention_risk_watchlist.csv")
    fairness = pd.read_csv(OUTPUTS / "fairness_group_metrics.csv")

    simulator = build_policy_simulator(scored, frontier)
    simulator.to_csv(OUTPUTS / "business_policy_simulator.csv", index=False)

    selected = simulator.sort_values("risk_adjusted_margin_approved", ascending=False).iloc[0]
    best_model = metrics.iloc[0]
    best_channel = channel.sort_values("avg_risk_adjusted_margin", ascending=False).iloc[0]
    worst_retention = retention.sort_values("churn_rate", ascending=False).iloc[0]
    lowest_fairness = fairness.sort_values("approval_rate_ratio_vs_max").iloc[0]
    safest_band = risk_band.sort_values("realized_default_rate").iloc[0]
    riskiest_band = risk_band.sort_values("realized_default_rate", ascending=False).iloc[0]

    casebook = f"""# Interview Casebook: Credit Risk and Customer Lifecycle Analytics

## 30-Second Pitch

I built an end-to-end fintech analytics system that scores customer default risk, maps customers into risk bands, simulates approval policy, monitors drift/fairness, and exports SQL-ready business views. The strongest decision artifact is the policy simulator: it shows how approval rate, default risk, expected margin, and declined-customer opportunity cost change as the PD threshold moves.

## Screening-Grade Evidence

| Area | Evidence |
|---|---|
| Model quality | {best_model['model']} ROC-AUC {best_model['roc_auc']:.3f}, KS {best_model['ks']:.3f}, top-decile recall {best_model['recall_top10']:.2%} |
| Public benchmark | OpenML German Credit benchmark included in `outputs/real_openml_german_credit_benchmark.csv` |
| Policy simulation | `outputs/business_policy_simulator.csv` and `outputs/policy_threshold_frontier.csv` |
| SQL analytics | Risk, cohort, acquisition, retention, pricing, and collections queries in `queries/` |
| Governance | PSI drift, fairness watchlist, reason codes, and model card |

## Business Decision

The best margin-maximizing threshold in this simulation is PD <= {pct(selected['pd_threshold'])}. It approves {pct(selected['approval_rate'])} of customers, expects {selected['expected_defaults_approved']:.1f} defaults among approved customers, and retains {money(selected['risk_adjusted_margin_approved'])} of risk-adjusted margin.

## Segment Actions

- Safest observed band: {safest_band['risk_band']} with realized default rate {pct(safest_band['realized_default_rate'])}.
- Riskiest observed band: {riskiest_band['risk_band']} with realized default rate {pct(riskiest_band['realized_default_rate'])}.
- Best acquisition pocket: {best_channel['acquisition_channel']} / {best_channel['region']} with {pct(best_channel['approval_rate'])} approval and {money(best_channel['avg_risk_adjusted_margin'])} average risk-adjusted margin.
- Highest retention-risk pocket: {worst_retention['customer_segment']} / {worst_retention['acquisition_channel']} / {worst_retention['risk_band']} with {pct(worst_retention['churn_rate'])} churn.
- Fairness watchlist: {lowest_fairness['dimension']}={lowest_fairness['group']} has approval ratio {lowest_fairness['approval_rate_ratio_vs_max']:.2f} versus the highest-approved peer group.

## SQL Case Questions Covered

1. Which risk bands should we approve, review, or decline?
2. Which acquisition channels produce the best risk-adjusted customers?
3. Which cohorts show deterioration in retention, default, or product depth?
4. Which high-value customers are declined and should be manually reviewed?
5. Which customer pockets need retention campaigns?
6. Which fairness groups require policy investigation?
7. Where does expected margin stop compensating for default risk?

## Interview Defense

If asked whether this is production-ready, the correct answer is no: it is a reproducible analytics and governance prototype. A real lender would require bureau contracts, reject inference, legal review, protected-class testing, challenger monitoring, and live A/B policy evaluation. The project is strong because it shows the full workflow from model score to business decision, not because it pretends synthetic data is a bank's real portfolio.
"""
    (REPORTS / "INTERVIEW_CASEBOOK.md").write_text(casebook, encoding="utf-8")

    print("Wrote outputs/business_policy_simulator.csv")
    print("Wrote reports/INTERVIEW_CASEBOOK.md")


if __name__ == "__main__":
    main()

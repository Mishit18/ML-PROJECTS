from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _benchmark_summary(real_benchmark: pd.DataFrame | None) -> str:
    if real_benchmark is None or real_benchmark.empty:
        return ""
    real_best = real_benchmark.iloc[0]
    dataset_name = real_best.get("dataset", "public credit-risk dataset")
    return (
        f"- Real {dataset_name} benchmark: {real_best['model']} ROC-AUC "
        f"{real_best['roc_auc']:.3f}, KS {real_best['ks']:.3f} on "
        f"{int(real_best['records']):,} public records\n"
    )


def write_plots(scored: pd.DataFrame, importance: pd.DataFrame, output_dir: Path, frontier: pd.DataFrame | None = None, fairness: pd.DataFrame | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 4.8))
    band = scored.groupby("risk_band", observed=False)["default_12m"].mean().reset_index()
    sns.barplot(data=band, x="risk_band", y="default_12m", color="#3b82f6")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Realized default rate")
    plt.xlabel("Risk band")
    plt.tight_layout()
    plt.savefig(output_dir / "risk_band_default_rate.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    top = importance.head(12).iloc[::-1]
    sns.barplot(data=top, x="importance_mean", y="feature", color="#14b8a6")
    plt.xlabel("Permutation importance: ROC-AUC drop")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    monthly = scored.groupby("vintage_month", observed=False)[["pd_score", "default_12m", "churn_6m"]].mean().reset_index()
    plt.plot(monthly["vintage_month"], monthly["pd_score"], marker="o", label="Predicted default")
    plt.plot(monthly["vintage_month"], monthly["default_12m"], marker="o", label="Realized default")
    plt.plot(monthly["vintage_month"], monthly["churn_6m"], marker="o", label="Churn")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "monthly_risk_lifecycle_trend.png", dpi=160)
    plt.close()

    if frontier is not None and not frontier.empty:
        plt.figure(figsize=(8, 4.8))
        plt.plot(frontier["approval_rate"], frontier["approved_default_rate"], marker="o", label="Default rate")
        plt.xlabel("Approval rate")
        plt.ylabel("Approved default rate")
        plt.title("Policy Frontier")
        plt.tight_layout()
        plt.savefig(output_dir / "policy_threshold_frontier.png", dpi=160)
        plt.close()

    if fairness is not None and not fairness.empty:
        watched = fairness.sort_values("approval_rate_ratio_vs_max").head(12)
        plt.figure(figsize=(8, 4.8))
        sns.barplot(data=watched, x="approval_rate_ratio_vs_max", y="group", hue="dimension", dodge=False)
        plt.axvline(0.80, color="red", linestyle="--", linewidth=1)
        plt.xlabel("Approval-rate ratio vs highest group")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(output_dir / "fairness_approval_ratio_watchlist.png", dpi=160)
        plt.close()


def write_reports(
    metrics: pd.DataFrame,
    scored: pd.DataFrame,
    importance: pd.DataFrame,
    drift: pd.DataFrame,
    report_dir: Path,
    frontier: pd.DataFrame | None = None,
    policy: dict[str, float] | None = None,
    fairness_summary: dict[str, object] | None = None,
    real_benchmark: pd.DataFrame | None = None,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    best = metrics.iloc[0]
    approval_rate = scored["approve_policy"].mean()
    approved_default = scored.loc[scored["approve_policy"] == 1, "default_12m"].mean()
    declined_default = scored.loc[scored["approve_policy"] == 0, "default_12m"].mean()
    top_features = ", ".join(importance.head(6)["feature"].tolist())
    severe_count = int((drift["status"] == "severe").sum())
    policy = policy or {}
    fairness_summary = fairness_summary or {}
    real_line = _benchmark_summary(real_benchmark)

    executive = f"""# Executive Brief: Credit Risk and Customer Lifecycle Analytics

## Portfolio Snapshot

- Test customers scored: {len(scored):,}
- Selected model: {best['model']}
- ROC-AUC: {best['roc_auc']:.3f}
- PR-AUC: {best['pr_auc']:.3f}
- KS statistic: {best['ks']:.3f}
- Recall in riskiest decile: {best['recall_top10']:.2%}
- Policy approval rate: {approval_rate:.2%}
- Approved default rate: {approved_default:.2%}
- Declined default rate: {declined_default:.2%}
{real_line}

## Business Interpretation

The project converts raw applicant and customer behavior data into a probability-of-default score, risk bands, approval policy simulation, customer churn readout, and risk-adjusted margin view. This mirrors the analytics charter in banking and fintech roles: acquisition, engagement, retention, risk scorecards, behavioral models, and data-driven portfolio strategy.

## Main Drivers

Top model drivers by permutation importance: {top_features}.

## Monitoring

PSI drift monitoring flagged {severe_count} severe numeric feature shifts between the development window and the latest scoring window. The report is designed for a model governance workflow: monitor, investigate, recalibrate, and document policy changes.

## Policy Optimization

Best constrained threshold from the approval/default frontier:

- PD threshold: {policy.get('pd_threshold', float('nan')):.2%}
- Approval rate: {policy.get('approval_rate', float('nan')):.2%}
- Approved default rate: {policy.get('approved_default_rate', float('nan')):.2%}
- Total risk-adjusted margin: {policy.get('total_risk_adjusted_margin', float('nan')):,.0f}

## Fairness and Compliance Readiness

- Group dimensions checked: {fairness_summary.get('dimensions_checked', 0)}
- Groups checked: {fairness_summary.get('groups_checked', 0)}
- Adverse-impact watch groups: {fairness_summary.get('watch_groups', 0)}
- Lowest approval-rate ratio: {fairness_summary.get('lowest_approval_ratio', float('nan')):.2f}
"""
    (report_dir / "executive_brief.md").write_text(executive, encoding="utf-8")

    resume = f"""# Resume Bullets

## Data Science / Analytics

- Built end-to-end credit risk and customer lifecycle analytics pipeline on {len(scored):,}+ synthetic banking customers, combining bureau, repayment, utilization, acquisition, churn, and margin signals.
- Trained and compared scorecard-style logistic regression, gradient boosting, and random forest models; selected {best['model']} with ROC-AUC {best['roc_auc']:.3f}, PR-AUC {best['pr_auc']:.3f}, KS {best['ks']:.3f}, and {best['recall_top10']:.1%} recall in the riskiest decile.
- Designed SQL analytics layer in DuckDB for risk-band policy, approval rate, realized default, churn, segment profitability, and acquisition-channel lifecycle KPIs.
- Added permutation explainability, reason-code generation, PSI drift monitoring, fairness diagnostics, score-band governance, and risk-adjusted margin simulation to translate model outputs into underwriting and retention actions.

## Interview Defense

- The dataset is synthetic for reproducibility, but relationships are intentionally modeled after unsecured credit portfolios.
- The split is temporal by vintage month, avoiding random-split leakage.
- Metrics include ROC-AUC, PR-AUC, KS, top-decile recall, approval policy, and realized default by risk band.
- The project avoids claiming production profit uplift; it reports a policy simulation, threshold frontier, reason codes, fairness diagnostics, real public-data benchmark, and documented governance checks.
"""
    (report_dir / "resume_bullets.md").write_text(resume, encoding="utf-8")

    model_card = f"""# Model Card

## Intended Use

Rank-order applicants/customers by 12-month default risk and support portfolio-level decisions across approval policy, acquisition channel quality, retention, and risk-adjusted margin.

## Data

Synthetic portfolio with bureau behavior, income, utilization, repayment behavior, card tenure, digital engagement, acquisition channel, region, churn, expected margin, and default labels.

## Metrics

{metrics.to_markdown(index=False)}

## Real Public Dataset Benchmark

{real_benchmark.to_markdown(index=False) if real_benchmark is not None and not real_benchmark.empty else "Real-data benchmark not generated."}

## Policy Frontier

{frontier.head(12).to_markdown(index=False) if frontier is not None and not frontier.empty else "Policy frontier not generated."}

## Limitations

- Synthetic data is useful for reproducibility and interview demonstration, not direct production deployment.
- A real lender would require protected-class fairness review, reject inference, bureau governance, macroeconomic validation, and legal/compliance approval.
- The policy simulation is directional; it does not claim actual business uplift without live experimentation.
"""
    (report_dir / "model_card.md").write_text(model_card, encoding="utf-8")

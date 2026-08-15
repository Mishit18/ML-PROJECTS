from __future__ import annotations

import numpy as np
import pandas as pd


def reject_inference_proxy(
    scored: pd.DataFrame,
    score_col: str = "pd_score",
    approved_col: str = "approve_policy",
    default_col: str = "default_12m",
    bins: int = 6,
) -> pd.DataFrame:
    """Estimate hidden declined-book risk by score band.

    This is a conservative interview-safe proxy, not a regulatory reject
    inference claim. It compares observed default in approved customers with
    model-implied PD for declined customers in the same score buckets.
    """
    required = {score_col, approved_col, default_col}
    missing = required - set(scored.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    work = scored.copy()
    work["score_bucket"] = pd.qcut(work[score_col], q=bins, duplicates="drop")
    rows = []
    for bucket, group in work.groupby("score_bucket", observed=True):
        approved = group[group[approved_col] == 1]
        declined = group[group[approved_col] == 0]
        observed_bad = float(approved[default_col].mean()) if not approved.empty else np.nan
        inferred_declined_bad = float(declined[score_col].mean()) if not declined.empty else np.nan
        uplift_vs_observed = (
            inferred_declined_bad - observed_bad
            if np.isfinite(observed_bad) and np.isfinite(inferred_declined_bad)
            else np.nan
        )
        rows.append(
            {
                "score_bucket": str(bucket),
                "customers": int(len(group)),
                "approved_customers": int(len(approved)),
                "declined_customers": int(len(declined)),
                "observed_approved_default_rate": observed_bad,
                "inferred_declined_default_rate": inferred_declined_bad,
                "declined_risk_uplift": uplift_vs_observed,
            }
        )
    return pd.DataFrame(rows)


def champion_challenger_report(
    metrics: pd.DataFrame,
    champion_col: str = "model",
    primary_metric: str = "roc_auc",
    guardrail_metric: str = "ks",
    min_primary_lift: float = 0.005,
) -> pd.DataFrame:
    """Rank models with a simple champion/challenger deployment verdict."""
    required = {champion_col, primary_metric, guardrail_metric}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    ranked = metrics.sort_values([primary_metric, guardrail_metric], ascending=False).reset_index(drop=True).copy()
    champion_primary = float(ranked.loc[0, primary_metric])
    champion_guardrail = float(ranked.loc[0, guardrail_metric])
    rows = []
    for idx, row in ranked.iterrows():
        lift = float(row[primary_metric]) - champion_primary
        guardrail_delta = float(row[guardrail_metric]) - champion_guardrail
        if idx == 0:
            verdict = "champion"
        elif lift >= min_primary_lift and guardrail_delta >= -0.01:
            verdict = "promote_candidate"
        elif guardrail_delta < -0.03:
            verdict = "reject_guardrail"
        else:
            verdict = "shadow_monitor"
        rows.append(
            {
                "model": row[champion_col],
                "rank": idx + 1,
                primary_metric: float(row[primary_metric]),
                guardrail_metric: float(row[guardrail_metric]),
                "primary_lift_vs_champion": lift,
                "guardrail_delta_vs_champion": guardrail_delta,
                "deployment_verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def lifecycle_survival_segments(
    scored: pd.DataFrame,
    segment_col: str = "customer_segment",
    churn_col: str = "churn_6m",
    margin_col: str = "expected_margin_12m",
    risk_col: str = "pd_score",
) -> pd.DataFrame:
    """Summarize retention, risk, and value by lifecycle segment."""
    required = {segment_col, churn_col, margin_col, risk_col}
    missing = required - set(scored.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    out = (
        scored.groupby(segment_col, observed=True)
        .agg(
            customers=(segment_col, "size"),
            churn_rate_6m=(churn_col, "mean"),
            survival_rate_6m=(churn_col, lambda s: 1 - float(s.mean())),
            avg_pd=(risk_col, "mean"),
            avg_margin_12m=(margin_col, "mean"),
            total_margin_12m=(margin_col, "sum"),
        )
        .reset_index()
        .sort_values(["survival_rate_6m", "avg_margin_12m"], ascending=False)
    )
    out["risk_adjusted_value_index"] = (
        out["survival_rate_6m"] * out["avg_margin_12m"] * (1 - out["avg_pd"])
    )
    return out


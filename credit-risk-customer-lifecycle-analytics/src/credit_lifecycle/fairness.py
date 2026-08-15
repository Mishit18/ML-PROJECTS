from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_ratio(a: float, b: float) -> float:
    if b == 0 or np.isnan(b):
        return float("nan")
    return float(a / b)


def fairness_by_group(scored: pd.DataFrame) -> pd.DataFrame:
    """Compute lightweight fairness and portfolio-composition diagnostics.

    This is not a substitute for legal/compliance review. It creates the first
    governance layer interviewers expect: approval rates, score averages,
    realized default rates, and adverse-impact-style ratios by available groups.
    """
    df = scored.copy()
    df["age_band"] = pd.cut(df["age"], bins=[0, 25, 35, 50, 120], labels=["<25", "25-34", "35-49", "50+"]).astype(str)
    income_bins = pd.qcut(df["annual_income"], q=4, duplicates="drop")
    df["income_band"] = income_bins.astype(str)
    groups = ["age_band", "income_band", "region", "customer_segment", "acquisition_channel"]

    rows: list[dict[str, object]] = []
    for column in groups:
        summary = (
            df.groupby(column, observed=False)
            .agg(
                customers=("customer_id", "count"),
                approval_rate=("approve_policy", "mean"),
                default_rate=("default_12m", "mean"),
                churn_rate=("churn_6m", "mean"),
                avg_pd=("pd_score", "mean"),
                avg_margin=("risk_adjusted_margin", "mean"),
            )
            .reset_index()
            .rename(columns={column: "group"})
        )
        max_approval = summary["approval_rate"].max()
        for record in summary.to_dict(orient="records"):
            ratio = _safe_ratio(record["approval_rate"], max_approval)
            rows.append(
                {
                    "dimension": column,
                    "group": record["group"],
                    "customers": int(record["customers"]),
                    "approval_rate": round(record["approval_rate"], 5),
                    "default_rate": round(record["default_rate"], 5),
                    "churn_rate": round(record["churn_rate"], 5),
                    "avg_pd": round(record["avg_pd"], 5),
                    "avg_margin": round(record["avg_margin"], 2),
                    "approval_rate_ratio_vs_max": round(ratio, 5) if not np.isnan(ratio) else np.nan,
                    "adverse_impact_watch": bool(ratio < 0.80) if not np.isnan(ratio) else False,
                }
            )
    return pd.DataFrame(rows)


def fairness_summary(fairness: pd.DataFrame) -> dict[str, object]:
    watched = fairness.loc[fairness["adverse_impact_watch"]].copy()
    return {
        "dimensions_checked": int(fairness["dimension"].nunique()),
        "groups_checked": int(len(fairness)),
        "watch_groups": int(len(watched)),
        "lowest_approval_ratio": float(fairness["approval_rate_ratio_vs_max"].min()),
        "watch_list": watched[["dimension", "group", "approval_rate", "approval_rate_ratio_vs_max"]].to_dict(orient="records"),
    }

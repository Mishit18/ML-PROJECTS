from __future__ import annotations

import numpy as np
import pandas as pd


def threshold_frontier(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    thresholds = np.round(np.arange(0.02, 0.301, 0.01), 3)
    for threshold in thresholds:
        approved = scored.loc[scored["pd_score"] <= threshold].copy()
        declined = scored.loc[scored["pd_score"] > threshold].copy()
        if approved.empty:
            continue
        rows.append(
            {
                "pd_threshold": threshold,
                "approval_rate": len(approved) / len(scored),
                "approved_customers": len(approved),
                "declined_customers": len(declined),
                "approved_default_rate": approved["default_12m"].mean(),
                "declined_default_rate": declined["default_12m"].mean() if not declined.empty else 0.0,
                "avg_risk_adjusted_margin": approved["risk_adjusted_margin"].mean(),
                "total_risk_adjusted_margin": approved["risk_adjusted_margin"].sum(),
                "avg_expected_margin": approved["expected_margin_12m"].mean(),
            }
        )
    return pd.DataFrame(rows)


def select_policy(frontier: pd.DataFrame, max_default_rate: float = 0.06, min_approval_rate: float = 0.50) -> dict[str, float]:
    feasible = frontier.loc[
        (frontier["approved_default_rate"] <= max_default_rate)
        & (frontier["approval_rate"] >= min_approval_rate)
    ].copy()
    if feasible.empty:
        feasible = frontier.copy()
    best = feasible.sort_values("total_risk_adjusted_margin", ascending=False).iloc[0]
    return {
        "pd_threshold": float(best["pd_threshold"]),
        "approval_rate": float(best["approval_rate"]),
        "approved_default_rate": float(best["approved_default_rate"]),
        "declined_default_rate": float(best["declined_default_rate"]),
        "total_risk_adjusted_margin": float(best["total_risk_adjusted_margin"]),
        "avg_risk_adjusted_margin": float(best["avg_risk_adjusted_margin"]),
    }

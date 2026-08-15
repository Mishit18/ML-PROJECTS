from __future__ import annotations

import numpy as np
import pandas as pd


def population_stability_index(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected = pd.Series(expected)
    actual = pd.Series(actual)
    expected = pd.to_numeric(expected, errors="coerce").dropna()
    actual = pd.to_numeric(actual, errors="coerce").dropna()
    if expected.empty or actual.empty:
        return float("nan")

    quantiles = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 3:
        quantiles = np.linspace(min(expected.min(), actual.min()), max(expected.max(), actual.max()), bins + 1)
    expected_counts, edges = np.histogram(expected, bins=quantiles)
    actual_counts, _ = np.histogram(actual, bins=edges)
    expected_pct = np.clip(expected_counts / max(expected_counts.sum(), 1), 1e-6, 1)
    actual_pct = np.clip(actual_counts / max(actual_counts.sum(), 1), 1e-6, 1)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def drift_report(train_df: pd.DataFrame, current_df: pd.DataFrame, numeric_features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in numeric_features:
        psi = population_stability_index(train_df[feature], current_df[feature])
        if psi < 0.10:
            status = "stable"
        elif psi < 0.25:
            status = "moderate"
        else:
            status = "severe"
        rows.append({"feature": feature, "psi": round(psi, 4), "status": status})
    return pd.DataFrame(rows).sort_values("psi", ascending=False)


def score_band(probability: pd.Series | np.ndarray) -> pd.Series:
    p = pd.Series(probability)
    return pd.cut(
        p,
        bins=[-0.001, 0.03, 0.06, 0.10, 0.16, 1.0],
        labels=["A: prime", "B: near-prime", "C: watchlist", "D: high-risk", "E: decline"],
    ).astype(str)

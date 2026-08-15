from __future__ import annotations

import numpy as np
import pandas as pd


REASON_CODE_MAP = {
    "credit_score": "Credit score is below the low-risk portfolio benchmark",
    "revolving_utilization": "Revolving utilization is elevated",
    "missed_payment_count_6m": "Recent missed-payment count is high",
    "delinquencies_12m": "Recent delinquency history increases risk",
    "debt_to_income": "Debt-to-income ratio is elevated",
    "bureau_inquiries_6m": "Recent bureau inquiries suggest credit-seeking behavior",
    "payment_to_min_ratio": "Payment-to-minimum ratio is weak",
    "employment_tenure_months": "Employment tenure is below the stable-income benchmark",
    "utilization_trend_3m": "Utilization trend has worsened recently",
}


HIGH_RISK_DIRECTION = {
    "credit_score": "low",
    "employment_tenure_months": "low",
    "payment_to_min_ratio": "low",
    "revolving_utilization": "high",
    "missed_payment_count_6m": "high",
    "delinquencies_12m": "high",
    "debt_to_income": "high",
    "bureau_inquiries_6m": "high",
    "utilization_trend_3m": "high",
}


def build_reason_codes(scored: pd.DataFrame, top_features: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    numeric_features = [f for f in top_features["feature"].tolist() if f in REASON_CODE_MAP]
    benchmark = scored[numeric_features].median(numeric_only=True)
    rows = []
    for _, customer in scored.sort_values("pd_score", ascending=False).head(250).iterrows():
        reasons = []
        for feature in numeric_features:
            direction = HIGH_RISK_DIRECTION[feature]
            value = customer[feature]
            median = benchmark[feature]
            if direction == "high" and value > median:
                reasons.append((feature, REASON_CODE_MAP[feature], value, median, abs(value - median)))
            elif direction == "low" and value < median:
                reasons.append((feature, REASON_CODE_MAP[feature], value, median, abs(value - median)))
        reasons = sorted(reasons, key=lambda x: x[-1], reverse=True)[:top_n]
        rows.append(
            {
                "customer_id": int(customer["customer_id"]),
                "pd_score": round(float(customer["pd_score"]), 5),
                "risk_band": customer["risk_band"],
                "reason_codes": " | ".join([r[1] for r in reasons]),
                "reason_features": ", ".join([r[0] for r in reasons]),
            }
        )
    return pd.DataFrame(rows)


def approval_decision_explainer(scored: pd.DataFrame) -> pd.DataFrame:
    frame = scored.copy()
    frame["decision"] = np.where(frame["approve_policy"] == 1, "approve", "decline/manual review")
    return frame[
        [
            "customer_id",
            "pd_score",
            "risk_band",
            "decision",
            "default_12m",
            "churn_6m",
            "expected_margin_12m",
            "risk_adjusted_margin",
        ]
    ].sort_values("pd_score", ascending=False)

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RetrainingRule:
    max_psi: float = 0.25
    moderate_feature_share: float = 0.20
    p95_latency_ms: float = 100.0
    min_requests_for_action: int = 25


def retraining_decision(
    drift_report: dict[str, object],
    latency_report: dict[str, float | int | None],
    rule: RetrainingRule = RetrainingRule(),
) -> dict[str, object]:
    features = list(drift_report.get("features", []))
    severe = [f["feature"] for f in features if f.get("status") == "severe"]
    moderate = [f["feature"] for f in features if f.get("status") == "moderate"]
    feature_share = (len(severe) + len(moderate)) / max(len(features), 1)
    p95_latency = latency_report.get("p95_ms")
    request_count = int(latency_report.get("request_count") or 0)

    reasons = []
    if float(drift_report.get("max_psi", 0.0)) >= rule.max_psi:
        reasons.append("max_psi_breach")
    if feature_share >= rule.moderate_feature_share:
        reasons.append("feature_drift_share_breach")
    if p95_latency is not None and float(p95_latency) > rule.p95_latency_ms:
        reasons.append("latency_slo_breach")

    action = "monitor"
    if reasons and request_count >= rule.min_requests_for_action:
        action = "retrain_candidate"
    elif reasons:
        action = "collect_more_traffic"

    return {
        "action": action,
        "reasons": reasons,
        "severe_features": severe,
        "moderate_features": moderate,
        "drifted_feature_share": round(float(feature_share), 4),
        "request_count": request_count,
    }


def shadow_model_agreement(
    champion_probabilities: list[float],
    challenger_probabilities: list[float],
    tolerance: float = 0.10,
) -> dict[str, float | int | str]:
    champion = np.asarray(champion_probabilities, dtype=float)
    challenger = np.asarray(challenger_probabilities, dtype=float)
    if champion.shape != challenger.shape:
        raise ValueError("champion and challenger probability arrays must have the same shape")
    delta = np.abs(champion - challenger)
    disagreement_rate = float(np.mean(delta > tolerance)) if delta.size else 0.0
    verdict = "safe_shadow" if disagreement_rate <= 0.05 else "investigate"
    return {
        "rows_compared": int(delta.size),
        "mean_abs_probability_delta": round(float(delta.mean()) if delta.size else 0.0, 6),
        "max_abs_probability_delta": round(float(delta.max()) if delta.size else 0.0, 6),
        "disagreement_rate": round(disagreement_rate, 6),
        "verdict": verdict,
    }


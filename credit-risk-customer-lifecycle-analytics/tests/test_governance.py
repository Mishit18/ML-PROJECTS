from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_lifecycle.data import DataConfig, generate_credit_portfolio
from credit_lifecycle.governance import (
    champion_challenger_report,
    lifecycle_survival_segments,
    reject_inference_proxy,
)
from credit_lifecycle.monitoring import score_band


def scored_sample() -> pd.DataFrame:
    df = generate_credit_portfolio(DataConfig(n_customers=1800, random_state=29))
    df["pd_score"] = df["default_probability_true"]
    df["risk_band"] = score_band(df["pd_score"])
    df["approve_policy"] = (df["pd_score"] <= 0.12).astype(int)
    return df


def test_reject_inference_proxy_has_declined_risk_bands():
    report = reject_inference_proxy(scored_sample(), bins=5)

    assert len(report) >= 4
    assert report["customers"].sum() == 1800
    assert report["declined_customers"].sum() > 0
    assert {"inferred_declined_default_rate", "declined_risk_uplift"}.issubset(report.columns)


def test_champion_challenger_report_assigns_verdicts():
    metrics = pd.DataFrame(
        [
            {"model": "scorecard", "roc_auc": 0.763, "ks": 0.405},
            {"model": "gbm", "roc_auc": 0.758, "ks": 0.392},
            {"model": "rf", "roc_auc": 0.744, "ks": 0.350},
        ]
    )
    report = champion_challenger_report(metrics)

    assert report.iloc[0]["deployment_verdict"] == "champion"
    assert set(report["deployment_verdict"]).issubset({"champion", "promote_candidate", "reject_guardrail", "shadow_monitor"})


def test_lifecycle_survival_segments_rank_value():
    segments = lifecycle_survival_segments(scored_sample())

    assert segments["customers"].sum() == 1800
    assert segments["survival_rate_6m"].between(0, 1).all()
    assert segments["risk_adjusted_value_index"].notna().all()


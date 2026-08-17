from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_lifecycle.data import DataConfig, generate_credit_portfolio
from credit_lifecycle.fairness import fairness_by_group
from credit_lifecycle.features import infer_feature_spec, temporal_split
from credit_lifecycle.monitoring import population_stability_index, score_band
from credit_lifecycle.policy import select_policy, threshold_frontier
from credit_lifecycle.real_data import expected_calibration_error


def test_data_generation_has_expected_targets_and_size():
    df = generate_credit_portfolio(DataConfig(n_customers=1500, random_state=7))
    assert len(df) == 1500
    assert {"default_12m", "churn_6m", "expected_margin_12m"}.issubset(df.columns)
    assert 0.02 < df["default_12m"].mean() < 0.55
    assert 0.01 < df["churn_6m"].mean() < 0.60


def test_temporal_split_and_features_avoid_target_leakage():
    df = generate_credit_portfolio(DataConfig(n_customers=1200, random_state=11))
    train, test = temporal_split(df, test_months=4)
    spec = infer_feature_spec(df)
    assert train["vintage_month"].max() < test["vintage_month"].min()
    assert "default_12m" not in spec.numeric_features
    assert "default_probability_true" not in spec.numeric_features
    assert "customer_segment" in spec.categorical_features


def test_psi_and_score_bands_are_well_formed():
    expected = np.linspace(0, 1, 1000)
    actual = np.linspace(0.2, 1.2, 1000)
    assert population_stability_index(expected, actual) > 0
    bands = score_band([0.01, 0.04, 0.08, 0.13, 0.30])
    assert list(bands) == ["A: prime", "B: near-prime", "C: watchlist", "D: high-risk", "E: decline"]


def test_policy_and_fairness_outputs_are_well_formed():
    df = generate_credit_portfolio(DataConfig(n_customers=1200, random_state=13))
    df["pd_score"] = df["default_probability_true"]
    df["risk_band"] = score_band(df["pd_score"])
    df["approve_policy"] = (df["pd_score"] <= 0.10).astype(int)
    df["risk_adjusted_margin"] = df["expected_margin_12m"] - df["pd_score"] * df["annual_income"] * df["debt_to_income"] * 0.18
    frontier = threshold_frontier(df)
    policy = select_policy(frontier)
    fairness = fairness_by_group(df)
    assert not frontier.empty
    assert 0 < policy["approval_rate"] <= 1
    assert fairness["dimension"].nunique() >= 4
    assert {"approval_rate_ratio_vs_max", "adverse_impact_watch"}.issubset(fairness.columns)


def test_expected_calibration_error_rewards_calibrated_probabilities():
    outcomes = np.array([0, 0, 1, 1])
    calibrated = np.array([0.05, 0.10, 0.90, 0.95])
    reversed_scores = 1 - calibrated
    assert expected_calibration_error(outcomes, calibrated, bins=4) < expected_calibration_error(
        outcomes, reversed_scores, bins=4
    )

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataConfig:
    n_customers: int = 25000
    random_state: int = 42
    output_path: Path = Path("data/credit_customer_portfolio.csv")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def generate_credit_portfolio(config: DataConfig = DataConfig()) -> pd.DataFrame:
    """Generate a realistic synthetic credit portfolio.

    The generator intentionally encodes business relationships seen in cards
    and unsecured lending: defaults rise with delinquencies, high utilization,
    debt burden, weak payment behavior, recent inquiries, and adverse drift.
    Churn and LTV are generated separately so the same dataset supports risk,
    marketing, and customer lifecycle analytics.
    """
    rng = np.random.default_rng(config.random_state)
    n = config.n_customers

    vintage = pd.period_range("2023-01", periods=24, freq="M")
    vintage_month = rng.choice(vintage.astype(str), size=n, p=np.linspace(1.4, 0.6, 24) / np.linspace(1.4, 0.6, 24).sum())
    age = np.clip(rng.normal(34, 9, n), 21, 64).round().astype(int)
    annual_income = np.clip(rng.lognormal(mean=11.25, sigma=0.48, size=n), 250000, 4500000)
    employment_tenure_months = np.clip(rng.gamma(4.0, 13.0, n), 0, 240)
    credit_score = np.clip(rng.normal(710, 62, n), 480, 850)
    bureau_inquiries_6m = rng.poisson(np.clip(1.4 + (680 - credit_score) / 180, 0.2, 5.5), n)
    delinquencies_12m = rng.poisson(np.clip(0.12 + (650 - credit_score) / 180 + bureau_inquiries_6m * 0.07, 0.02, 2.8), n)
    revolving_utilization = np.clip(rng.beta(2.2, 3.1, n) + (700 - credit_score) / 900 + delinquencies_12m * 0.035, 0.02, 1.25)
    debt_to_income = np.clip(rng.beta(2.0, 5.2, n) + revolving_utilization * 0.24, 0.03, 1.25)
    card_tenure_months = np.clip(rng.gamma(3.0, 10.0, n), 1, 180)
    avg_monthly_spend = np.clip(annual_income / 12 * rng.beta(2.0, 6.5, n), 500, 350000)
    payment_to_min_ratio = np.clip(rng.lognormal(0.55, 0.55, n) - delinquencies_12m * 0.18, 0.15, 7.5)
    utilization_trend_3m = rng.normal(0, 0.16, n) + delinquencies_12m * 0.035 + (revolving_utilization - 0.55) * 0.12
    missed_payment_count_6m = rng.poisson(np.clip(0.08 + delinquencies_12m * 0.45 + revolving_utilization * 0.18, 0.02, 4.5), n)
    product_count = np.clip(rng.poisson(1.4 + annual_income / 2800000 + card_tenure_months / 130), 1, 7)
    digital_logins_30d = rng.poisson(np.clip(7 + avg_monthly_spend / 22000 - missed_payment_count_6m * 0.8, 0.5, 35), n)

    customer_segment = pd.cut(
        annual_income,
        bins=[0, 600000, 1200000, 2500000, np.inf],
        labels=["mass", "emerging_affluent", "affluent", "premium"],
    ).astype(str)
    acquisition_channel = rng.choice(["organic", "partner", "paid_search", "branch", "referral"], n, p=[0.25, 0.21, 0.19, 0.18, 0.17])
    region = rng.choice(["north", "south", "west", "east", "metro"], n, p=[0.19, 0.21, 0.24, 0.16, 0.20])

    channel_risk = pd.Series(acquisition_channel).map({"organic": -0.20, "referral": -0.28, "branch": -0.04, "partner": 0.08, "paid_search": 0.24}).to_numpy()
    macro_month = pd.Series(vintage_month).str[-2:].astype(int).to_numpy()
    macro_stress = np.where(macro_month.isin([1, 2, 3]) if hasattr(macro_month, "isin") else np.isin(macro_month, [1, 2, 3]), 0.10, 0.0)
    default_logit = (
        -3.25
        + 0.95 * revolving_utilization
        + 1.18 * debt_to_income
        + 0.34 * delinquencies_12m
        + 0.22 * missed_payment_count_6m
        + 0.12 * bureau_inquiries_6m
        - 0.010 * (credit_score - 680)
        - 0.11 * np.log1p(payment_to_min_ratio)
        - 0.004 * np.minimum(employment_tenure_months, 120)
        + 0.55 * np.maximum(utilization_trend_3m, 0)
        + channel_risk
        + macro_stress
    )
    default_probability = np.clip(sigmoid(default_logit), 0.005, 0.75)
    default_12m = rng.binomial(1, default_probability)

    churn_logit = (
        -1.85
        - 0.08 * product_count
        - 0.015 * digital_logins_30d
        - 0.22 * np.log1p(avg_monthly_spend / 1000)
        + 0.28 * (customer_segment == "mass").astype(float)
        + 0.18 * (acquisition_channel == "paid_search").astype(float)
        + 0.55 * default_12m
        + rng.normal(0, 0.18, n)
    )
    churn_6m = rng.binomial(1, np.clip(sigmoid(churn_logit), 0.01, 0.70))

    expected_margin_12m = (
        avg_monthly_spend * 0.018 * 12
        + revolving_utilization * annual_income * 0.012
        + product_count * 850
        - default_probability * annual_income * debt_to_income * 0.18
        - churn_6m * 1200
    )

    df = pd.DataFrame(
        {
            "customer_id": np.arange(100000, 100000 + n),
            "vintage_month": vintage_month,
            "age": age,
            "annual_income": annual_income.round(0),
            "employment_tenure_months": employment_tenure_months.round(1),
            "credit_score": credit_score.round(0),
            "bureau_inquiries_6m": bureau_inquiries_6m,
            "delinquencies_12m": delinquencies_12m,
            "revolving_utilization": revolving_utilization.round(4),
            "debt_to_income": debt_to_income.round(4),
            "card_tenure_months": card_tenure_months.round(1),
            "avg_monthly_spend": avg_monthly_spend.round(0),
            "payment_to_min_ratio": payment_to_min_ratio.round(3),
            "utilization_trend_3m": utilization_trend_3m.round(4),
            "missed_payment_count_6m": missed_payment_count_6m,
            "product_count": product_count,
            "digital_logins_30d": digital_logins_30d,
            "customer_segment": customer_segment,
            "acquisition_channel": acquisition_channel,
            "region": region,
            "default_probability_true": default_probability.round(5),
            "default_12m": default_12m,
            "churn_6m": churn_6m,
            "expected_margin_12m": expected_margin_12m.round(2),
        }
    )
    return df.sample(frac=1, random_state=config.random_state).reset_index(drop=True)


def write_dataset(config: DataConfig = DataConfig()) -> pd.DataFrame:
    df = generate_credit_portfolio(config)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.output_path, index=False)
    return df

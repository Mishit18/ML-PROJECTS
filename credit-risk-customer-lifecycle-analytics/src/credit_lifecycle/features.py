from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "default_12m"
SECONDARY_TARGETS = ["churn_6m", "expected_margin_12m", "default_probability_true"]
ID_COLUMNS = ["customer_id", "vintage_month"]


@dataclass(frozen=True)
class FeatureSpec:
    numeric_features: list[str]
    categorical_features: list[str]


def infer_feature_spec(df: pd.DataFrame) -> FeatureSpec:
    excluded = set(ID_COLUMNS + [TARGET] + SECONDARY_TARGETS)
    feature_df = df.drop(columns=[c for c in excluded if c in df.columns])
    categorical = feature_df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    numeric = [c for c in feature_df.columns if c not in categorical]
    return FeatureSpec(numeric_features=numeric, categorical_features=categorical)


def build_preprocessor(spec: FeatureSpec) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), spec.numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), spec.categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def temporal_split(df: pd.DataFrame, test_months: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered_months = sorted(df["vintage_month"].unique())
    test_cutoff = set(ordered_months[-test_months:])
    train = df.loc[~df["vintage_month"].isin(test_cutoff)].copy()
    test = df.loc[df["vintage_month"].isin(test_cutoff)].copy()
    return train, test


def get_xy(df: pd.DataFrame, spec: FeatureSpec):
    feature_cols = spec.numeric_features + spec.categorical_features
    return df[feature_cols], df[TARGET]

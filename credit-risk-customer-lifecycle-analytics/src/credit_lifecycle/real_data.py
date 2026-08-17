from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .modeling import ks_statistic

import numpy as np


UCI_CREDIT_URL = (
    "https://archive.ics.uci.edu/static/public/350/"
    "default+of+credit+card+clients.zip"
)


def expected_calibration_error(y_true, probability, bins: int = 10) -> float:
    frame = pd.DataFrame({"actual": np.asarray(y_true), "probability": np.asarray(probability)})
    frame["bin"] = pd.cut(frame["probability"], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    grouped = frame.groupby("bin", observed=True).agg(
        observations=("actual", "size"),
        observed_rate=("actual", "mean"),
        predicted_rate=("probability", "mean"),
    )
    return float(
        ((grouped["observations"] / len(frame)) * (grouped["observed_rate"] - grouped["predicted_rate"]).abs()).sum()
    )


def bootstrap_auc_interval(y_true, probability, iterations: int = 300, seed: int = 42) -> tuple[float, float]:
    y_array = np.asarray(y_true)
    probability_array = np.asarray(probability)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(iterations):
        indices = rng.integers(0, len(y_array), len(y_array))
        if np.unique(y_array[indices]).size < 2:
            continue
        scores.append(roc_auc_score(y_array[indices], probability_array[indices]))
    return float(np.quantile(scores, 0.025)), float(np.quantile(scores, 0.975))


def _load_uci_credit_card_default(cache_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Download and load the UCI credit-card default dataset with local caching."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / "default_credit_card_clients.zip"
    if not archive.exists():
        urlretrieve(UCI_CREDIT_URL, archive)

    extract_dir = cache_dir / "default_credit_card_clients"
    if not extract_dir.exists():
        with ZipFile(archive) as zipped:
            zipped.extractall(extract_dir)

    workbooks = list(extract_dir.rglob("*.xls")) + list(extract_dir.rglob("*.xlsx"))
    if not workbooks:
        raise FileNotFoundError("UCI archive did not contain an Excel workbook")

    frame = pd.read_excel(workbooks[0], header=1)
    frame.columns = [str(column).strip() for column in frame.columns]
    target = "default payment next month"
    if target not in frame.columns:
        raise ValueError(f"Expected target column '{target}' was not found")

    y = frame.pop(target).astype(int)
    frame = frame.drop(columns=["ID"], errors="ignore")
    return frame, y


def run_uci_credit_card_default(output_dir: Path, cache_dir: Path) -> pd.DataFrame:
    """Benchmark scorecard and boosting models on 30,000 real UCI customers."""
    X, y = _load_uci_credit_card_default(cache_dir)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    gbm_parameters = dict(random_state=42, learning_rate=0.04, max_depth=2, n_estimators=180)
    candidates = {
        "logistic_scorecard": Pipeline(
            [("scale", StandardScaler()), ("model", LogisticRegression(max_iter=3000))]
        ),
        "gradient_boosting": GradientBoostingClassifier(**gbm_parameters),
        "gradient_boosting_sigmoid": CalibratedClassifierCV(
            GradientBoostingClassifier(**gbm_parameters), method="sigmoid", cv=5
        ),
        "gradient_boosting_isotonic": CalibratedClassifierCV(
            GradientBoostingClassifier(**gbm_parameters), method="isotonic", cv=5
        ),
    }
    rows: list[dict[str, float | int | str]] = []
    calibration_rows: list[dict[str, float | int | str]] = []
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        probability = model.predict_proba(X_test)[:, 1]
        auc_low, auc_high = bootstrap_auc_interval(y_test, probability)
        rows.append(
            {
                "dataset": "UCI Default of Credit Card Clients",
                "model": name,
                "records": len(X),
                "test_records": len(X_test),
                "default_rate": round(float(y.mean()), 5),
                "roc_auc": round(float(roc_auc_score(y_test, probability)), 5),
                "pr_auc": round(float(average_precision_score(y_test, probability)), 5),
                "ks": round(float(ks_statistic(y_test, probability)), 5),
                "brier_score": round(float(brier_score_loss(y_test, probability)), 5),
                "ece_10bin": round(expected_calibration_error(y_test, probability), 5),
                "roc_auc_ci_low": round(auc_low, 5),
                "roc_auc_ci_high": round(auc_high, 5),
                "validation": "stratified holdout; source has no observation timestamp",
            }
        )

        bins = pd.qcut(probability, q=10, duplicates="drop")
        calibration = pd.DataFrame(
            {"probability": probability, "actual": y_test.to_numpy(), "score_band": bins}
        )
        grouped = calibration.groupby("score_band", observed=True).agg(
            customers=("actual", "size"),
            predicted_default_rate=("probability", "mean"),
            observed_default_rate=("actual", "mean"),
        )
        for band_number, (_, values) in enumerate(grouped.iterrows(), start=1):
            calibration_rows.append(
                {
                    "model": name,
                    "score_band": band_number,
                    "customers": int(values["customers"]),
                    "predicted_default_rate": round(float(values["predicted_default_rate"]), 5),
                    "observed_default_rate": round(float(values["observed_default_rate"]), 5),
                }
            )

    results = pd.DataFrame(rows).sort_values(["roc_auc", "ks"], ascending=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "real_uci_credit_card_benchmark.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(
        output_dir / "real_uci_credit_card_calibration.csv", index=False
    )
    results[["model", "brier_score", "ece_10bin", "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high"]].to_csv(
        output_dir / "real_uci_calibration_comparison.csv", index=False
    )
    return results


def run_openml_german_credit(output_dir: Path) -> pd.DataFrame:
    """Run a real public-data benchmark on OpenML credit-g (data_id=31)."""
    dataset = fetch_openml(data_id=31, as_frame=True, parser="auto")
    frame = dataset.frame.copy()
    y = (frame["class"].astype(str) == "bad").astype(int)
    X = frame.drop(columns=["class"])
    categorical = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    prep = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        verbose_feature_names_out=False,
    )
    candidates = {
        "real_logistic_scorecard": Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=2000))]),
        "real_gradient_boosting": Pipeline(
            [
                ("prep", ColumnTransformer(
                    [
                        ("num", StandardScaler(), numeric),
                        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
                    ],
                    verbose_feature_names_out=False,
                )),
                ("model", GradientBoostingClassifier(random_state=42, learning_rate=0.04, max_depth=2, n_estimators=120)),
            ]
        ),
    }
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    rows = []
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        rows.append(
            {
                "dataset": "OpenML credit-g data_id=31",
                "model": name,
                "records": len(frame),
                "test_records": len(X_test),
                "bad_rate": round(float(y.mean()), 5),
                "roc_auc": round(float(roc_auc_score(y_test, proba)), 5),
                "pr_auc": round(float(average_precision_score(y_test, proba)), 5),
                "ks": round(float(ks_statistic(y_test, proba)), 5),
            }
        )
    results = pd.DataFrame(rows).sort_values(["roc_auc", "ks"], ascending=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "real_openml_german_credit_benchmark.csv", index=False)
    return results

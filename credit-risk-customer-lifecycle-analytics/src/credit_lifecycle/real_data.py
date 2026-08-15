from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .modeling import ks_statistic


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

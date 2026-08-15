from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from .features import FeatureSpec, TARGET, build_preprocessor, get_xy
from .monitoring import score_band


def ks_statistic(y_true: pd.Series, y_score: np.ndarray) -> float:
    frame = pd.DataFrame({"y": y_true.to_numpy(), "score": y_score}).sort_values("score", ascending=False)
    positives = max(frame["y"].sum(), 1)
    negatives = max((1 - frame["y"]).sum(), 1)
    frame["cum_bad"] = frame["y"].cumsum() / positives
    frame["cum_good"] = (1 - frame["y"]).cumsum() / negatives
    return float((frame["cum_bad"] - frame["cum_good"]).abs().max())


def recall_at_top_fraction(y_true: pd.Series, y_score: np.ndarray, fraction: float = 0.10) -> float:
    cutoff = max(int(len(y_true) * fraction), 1)
    order = np.argsort(-y_score)[:cutoff]
    return float(y_true.to_numpy()[order].sum() / max(y_true.sum(), 1))


def make_candidates(spec: FeatureSpec) -> dict[str, Pipeline]:
    preprocessor = build_preprocessor(spec)
    return {
        "logistic_scorecard": Pipeline(
            [
                ("prep", preprocessor),
                ("model", LogisticRegression(max_iter=2000, solver="lbfgs")),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("prep", build_preprocessor(spec)),
                ("model", GradientBoostingClassifier(random_state=42, learning_rate=0.045, n_estimators=180, max_depth=3)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("prep", build_preprocessor(spec)),
                ("model", RandomForestClassifier(n_estimators=240, min_samples_leaf=40, class_weight="balanced_subsample", random_state=42, n_jobs=-1)),
            ]
        ),
    }


def evaluate_model(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    proba = model.predict_proba(X)[:, 1]
    threshold = float(np.quantile(proba, 0.80))
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "ks": ks_statistic(y, proba),
        "f1_top20": f1_score(y, pred, zero_division=0),
        "precision_top20": precision_score(y, pred, zero_division=0),
        "recall_top20": recall_score(y, pred, zero_division=0),
        "recall_top10": recall_at_top_fraction(y, proba, 0.10),
        "threshold_top20": threshold,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def train_and_select(train_df: pd.DataFrame, test_df: pd.DataFrame, spec: FeatureSpec, artifact_dir: Path) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame]:
    X_train, y_train = get_xy(train_df, spec)
    X_test, y_test = get_xy(test_df, spec)
    rows = []
    fitted = {}
    for name, model in make_candidates(spec).items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        rows.append({"model": name, **{k: round(v, 5) if isinstance(v, float) else v for k, v in metrics.items()}})
        fitted[name] = model
    metrics_df = pd.DataFrame(rows).sort_values(["roc_auc", "ks"], ascending=False)
    best_name = metrics_df.iloc[0]["model"]
    best_model = fitted[best_name]

    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, artifact_dir / "credit_default_model.joblib")
    (artifact_dir / "model_metrics.json").write_text(json.dumps(metrics_df.to_dict(orient="records"), indent=2), encoding="utf-8")

    scored = test_df.copy()
    scored["pd_score"] = best_model.predict_proba(X_test)[:, 1]
    scored["risk_band"] = score_band(scored["pd_score"])
    scored["approve_policy"] = np.where(scored["risk_band"].isin(["A: prime", "B: near-prime", "C: watchlist"]), 1, 0)
    scored["risk_adjusted_margin"] = scored["expected_margin_12m"] - scored["pd_score"] * scored["annual_income"] * scored["debt_to_income"] * 0.18
    return best_model, metrics_df, scored


def permutation_explain(model: Pipeline, test_df: pd.DataFrame, spec: FeatureSpec, output_path: Path) -> pd.DataFrame:
    X_test, y_test = get_xy(test_df, spec)
    result = permutation_importance(model, X_test, y_test, scoring="roc_auc", n_repeats=7, random_state=42, n_jobs=-1)
    importance = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(output_path, index=False)
    return importance

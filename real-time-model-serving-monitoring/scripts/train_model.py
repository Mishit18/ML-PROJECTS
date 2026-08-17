from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from pandas.api.types import is_numeric_dtype
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MONOREPO_ROOT = PROJECT_ROOT.parent
CREDIT_PROJECT = MONOREPO_ROOT / "credit-risk-customer-lifecycle-analytics"
sys.path.insert(0, str(CREDIT_PROJECT))

from src.credit_lifecycle.home_credit import build_home_credit_features  # noqa: E402
from src.credit_lifecycle.modeling import ks_statistic  # noqa: E402
from src.credit_lifecycle.real_data import expected_calibration_error  # noqa: E402


ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports"
EXAMPLE_DIR = PROJECT_ROOT / "examples"


def metric_bundle(y_true, probability: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "ks": float(ks_statistic(y_true, probability)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "ece_10bin": float(expected_calibration_error(y_true, probability)),
    }


def write_model_card(metadata: dict[str, object]) -> None:
    metrics = metadata["metrics"]
    lines = [
        "# Model Card - Home Credit Default Risk",
        "",
        "## Intended Use",
        "",
        "Calibrated probability-of-default inference for a reproducible ML serving and monitoring demonstration. It is not a lending decision system.",
        "",
        "## Data and Model",
        "",
        f"- Source: {metadata['dataset']}",
        f"- Applications: {metadata['applications']:,}",
        f"- Behavioral records aggregated: {metadata['behavioral_records']:,}",
        f"- Training rows: {metadata['train_rows']:,}",
        f"- Validation rows: {metadata['validation_rows']:,}",
        f"- Test rows: {metadata['test_rows']:,}",
        f"- Numeric features: {metadata['feature_count']}",
        f"- Algorithm: {metadata['algorithm']} with isotonic calibration",
        "",
        "## Held-out Test Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in metrics.items():
        lines.append(f"| {key} | {value:.4f} |")
    lines.extend([
        "",
        "## Monitoring and Governance",
        "",
        "- Request validation and structured prediction logging.",
        "- p50/p95 latency telemetry and PSI feature-drift monitoring.",
        "- Shadow-model agreement and retraining-candidate decision endpoints.",
        "- Protected gender attribute excluded from model features.",
        "",
        "## Limitations",
        "",
        "- Public competition data may not represent a current lending population.",
        "- The API demonstrates engineering controls; it does not make autonomous credit decisions.",
        "- Expected loss and approval policies require institution-specific validation before use.",
    ])
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "model_card.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Home Credit deployment model.")
    parser.add_argument("--archive", type=Path, required=True, help="Official Kaggle Home Credit ZIP")
    args = parser.parse_args()

    frame, provenance = build_home_credit_features(args.archive)
    excluded = {"SK_ID_CURR", "TARGET", "CODE_GENDER"}
    feature_names = [
        column for column in frame.columns
        if column not in excluded and is_numeric_dtype(frame[column])
    ]
    features = frame[feature_names].replace([np.inf, -np.inf], np.nan)
    train_frame, holdout_frame, y_train, y_holdout = train_test_split(
        features, frame["TARGET"], test_size=0.30, stratify=frame["TARGET"], random_state=42
    )
    validation_frame, test_frame, y_validation, y_test = train_test_split(
        holdout_frame, y_holdout, test_size=0.50, stratify=y_holdout, random_state=43
    )
    medians = train_frame.median().fillna(0.0)
    train_frame = train_frame.fillna(medians)
    validation_frame = validation_frame.fillna(medians)
    test_frame = test_frame.fillna(medians)

    model = LGBMClassifier(
        n_estimators=500, learning_rate=0.035, num_leaves=31, subsample=0.85,
        colsample_bytree=0.80, reg_lambda=2.0, random_state=42, n_jobs=-1, verbosity=-1,
    )
    model.fit(train_frame, y_train)
    validation_probability = model.predict_proba(validation_frame)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(validation_probability, y_validation)
    test_probability = calibrator.predict(model.predict_proba(test_frame)[:, 1])
    metrics = metric_bundle(y_test, test_probability)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACT_DIR / "model.joblib")
    joblib.dump(calibrator, ARTIFACT_DIR / "calibrator.joblib")
    baseline = train_frame.sample(n=2000, random_state=42)
    (ARTIFACT_DIR / "baseline_stats.json").write_text(json.dumps({
        "feature_names": feature_names,
        "baseline_rows": np.round(baseline.to_numpy(), 6).tolist(),
        "medians": {key: round(float(value), 6) for key, value in medians.items()},
    }), encoding="utf-8")

    metadata = {
        "model_version": datetime.now(timezone.utc).strftime("lgbm-home-credit-%Y%m%d%H%M%S"),
        "algorithm": "LightGBMClassifier",
        "dataset": "Home Credit Default Risk public competition data",
        "applications": int(provenance["applications"]),
        "behavioral_records": int(provenance["bureau_records"] + provenance["previous_application_records"] + provenance["installment_records"]),
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "train_rows": int(len(train_frame)),
        "validation_rows": int(len(validation_frame)),
        "test_rows": int(len(test_frame)),
        "target_mapping": {"0": "non_default", "1": "default"},
        "decision_threshold": 0.5,
        "metrics": metrics,
        "calibration": "isotonic fitted only on validation split",
    }
    (ARTIFACT_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (EXAMPLE_DIR / "sample_request.json").write_text(
        json.dumps({"features": test_frame.iloc[0].astype(float).tolist()}, indent=2), encoding="utf-8"
    )
    write_model_card(metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

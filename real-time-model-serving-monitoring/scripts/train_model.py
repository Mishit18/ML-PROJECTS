from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports"


def rounded_dict(values: dict[str, float]) -> dict[str, float]:
    return {key: round(float(value), 6) for key, value in values.items()}


def write_model_card(metadata: dict[str, object], metrics: dict[str, float]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Model Card - Breast Cancer Risk Classifier",
        "",
        "## Intended Use",
        "",
        "Demonstration model for real-time serving, prediction logging, latency monitoring, and feature-drift checks. It is not a clinical product.",
        "",
        "## Model",
        "",
        f"- Algorithm: {metadata['algorithm']}",
        f"- Model version: {metadata['model_version']}",
        f"- Feature count: {metadata['feature_count']}",
        f"- Training rows: {metadata['train_rows']}",
        f"- Test rows: {metadata['test_rows']}",
        "",
        "## Test Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in metrics.items():
        lines.append(f"| {key} | {value:.4f} |")
    lines.extend(
        [
            "",
            "## Monitoring",
            "",
            "- Prediction requests are logged to `reports/prediction_log.jsonl`.",
            "- Latency is tracked in memory and exposed through `/monitor/latency`.",
            "- Feature drift is measured with population stability index through `/monitor/drift`.",
            "",
            "## Limitations",
            "",
            "- Public built-in dataset; no private production data.",
            "- Drift detection is monitoring only and does not automatically retrain the model.",
            "- This project demonstrates deployment discipline rather than state-of-the-art modeling.",
        ]
    )
    (REPORT_DIR / "model_card.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = [name.replace(" ", "_") for name in data.feature_names]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = model.predict(X_test)
    metrics = rounded_dict(
        {
            "accuracy": accuracy_score(y_test, predictions),
            "roc_auc": roc_auc_score(y_test, probabilities),
            "f1": f1_score(y_test, predictions),
            "precision": precision_score(y_test, predictions),
            "recall": recall_score(y_test, predictions),
        }
    )

    model_version = datetime.now(timezone.utc).strftime("rf-breast-cancer-%Y%m%d%H%M%S")
    metadata = {
        "model_version": model_version,
        "algorithm": "RandomForestClassifier",
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "target_mapping": {"0": "malignant", "1": "benign"},
        "metrics": metrics,
    }
    baseline_stats = {
        "feature_names": feature_names,
        "baseline_rows": np.round(X_train, 6).tolist(),
        "means": rounded_dict(dict(zip(feature_names, X_train.mean(axis=0)))),
        "stds": rounded_dict(dict(zip(feature_names, X_train.std(axis=0)))),
    }

    joblib.dump(model, ARTIFACT_DIR / "model.joblib")
    (ARTIFACT_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    (ARTIFACT_DIR / "baseline_stats.json").write_text(
        json.dumps(baseline_stats, indent=2),
        encoding="utf-8",
    )
    write_model_card(metadata, metrics)
    print(json.dumps({"model_version": model_version, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from .monitoring import append_prediction_log, now_ms, validate_feature_vector


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports"


@dataclass
class PredictionResult:
    prediction: int
    label: str
    probability_malignant: float
    probability_benign: float
    model_version: str
    latency_ms: float


class ModelService:
    def __init__(
        self,
        model_path: Path | None = None,
        metadata_path: Path | None = None,
        baseline_path: Path | None = None,
        log_path: Path | None = None,
    ) -> None:
        self.model_path = model_path or ARTIFACT_DIR / "model.joblib"
        self.metadata_path = metadata_path or ARTIFACT_DIR / "model_metadata.json"
        self.baseline_path = baseline_path or ARTIFACT_DIR / "baseline_stats.json"
        self.log_path = log_path or REPORT_DIR / "prediction_log.jsonl"
        self.model = None
        self.metadata: dict[str, object] = {}
        self.baseline: dict[str, object] = {}
        self.load()

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"missing model artifact at {self.model_path}. Run scripts/train_model.py first."
            )
        self.model = joblib.load(self.model_path)
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.baseline = json.loads(self.baseline_path.read_text(encoding="utf-8"))

    @property
    def model_version(self) -> str:
        return str(self.metadata.get("model_version", "unknown"))

    @property
    def feature_names(self) -> list[str]:
        return list(self.metadata.get("feature_names", []))

    @property
    def baseline_rows(self) -> list[list[float]]:
        return list(self.baseline.get("baseline_rows", []))

    def predict(self, features: list[float]) -> PredictionResult:
        validate_feature_vector(features, expected_count=len(self.feature_names))
        start = now_ms()
        row = np.asarray(features, dtype=float).reshape(1, -1)
        probabilities = self.model.predict_proba(row)[0]
        prediction = int(np.argmax(probabilities))
        latency_ms = round(now_ms() - start, 4)
        label = "malignant" if prediction == 0 else "benign"
        result = PredictionResult(
            prediction=prediction,
            label=label,
            probability_malignant=round(float(probabilities[0]), 6),
            probability_benign=round(float(probabilities[1]), 6),
            model_version=self.model_version,
            latency_ms=latency_ms,
        )
        append_prediction_log(
            self.log_path,
            {
                "model_version": result.model_version,
                "prediction": result.prediction,
                "label": result.label,
                "probability_malignant": result.probability_malignant,
                "probability_benign": result.probability_benign,
                "latency_ms": result.latency_ms,
            },
        )
        return result

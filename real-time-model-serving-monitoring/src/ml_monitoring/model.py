from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .monitoring import append_prediction_log, now_ms, validate_feature_vector


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports"


@dataclass
class PredictionResult:
    prediction: int
    label: str
    probability_default: float
    probability_non_default: float
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
        self.calibrator_path = ARTIFACT_DIR / "calibrator.joblib"
        self.log_path = log_path or REPORT_DIR / "prediction_log.jsonl"
        self.model = None
        self.calibrator = None
        self.metadata: dict[str, object] = {}
        self.baseline: dict[str, object] = {}
        self.load()

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"missing model artifact at {self.model_path}. Run scripts/train_model.py first."
            )
        self.model = joblib.load(self.model_path)
        if self.calibrator_path.exists():
            self.calibrator = joblib.load(self.calibrator_path)
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
        row = pd.DataFrame([np.asarray(features, dtype=float)], columns=self.feature_names)
        raw_default_probability = float(self.model.predict_proba(row)[0, 1])
        default_probability = (
            float(self.calibrator.predict([raw_default_probability])[0])
            if self.calibrator is not None
            else raw_default_probability
        )
        threshold = float(self.metadata.get("decision_threshold", 0.5))
        prediction = int(default_probability >= threshold)
        latency_ms = round(now_ms() - start, 4)
        label = "default" if prediction == 1 else "non_default"
        result = PredictionResult(
            prediction=prediction,
            label=label,
            probability_default=round(default_probability, 6),
            probability_non_default=round(1.0 - default_probability, 6),
            model_version=self.model_version,
            latency_ms=latency_ms,
        )
        append_prediction_log(
            self.log_path,
            {
                "model_version": result.model_version,
                "prediction": result.prediction,
                "label": result.label,
                "probability_default": result.probability_default,
                "probability_non_default": result.probability_non_default,
                "latency_ms": result.latency_ms,
            },
        )
        return result

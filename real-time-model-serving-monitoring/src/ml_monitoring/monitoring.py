from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


DRIFT_STABLE = 0.10
DRIFT_SEVERE = 0.25


@dataclass
class LatencyTracker:
    latencies_ms: list[float] = field(default_factory=list)

    def record(self, latency_ms: float) -> None:
        self.latencies_ms.append(float(latency_ms))

    def report(self) -> dict[str, float | int | None]:
        if not self.latencies_ms:
            return {
                "request_count": 0,
                "avg_ms": None,
                "p50_ms": None,
                "p95_ms": None,
                "max_ms": None,
            }
        values = np.array(self.latencies_ms, dtype=float)
        return {
            "request_count": int(values.size),
            "avg_ms": round(float(values.mean()), 4),
            "p50_ms": round(float(np.percentile(values, 50)), 4),
            "p95_ms": round(float(np.percentile(values, 95)), 4),
            "max_ms": round(float(values.max()), 4),
        }


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def validate_feature_vector(features: list[float], expected_count: int) -> None:
    if len(features) != expected_count:
        raise ValueError(f"expected {expected_count} features, received {len(features)}")
    for value in features:
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError("all features must be finite numeric values")


def population_stability_index(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
) -> float:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if expected.size == 0 or actual.size == 0:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = np.unique(np.quantile(expected, quantiles))
    if breakpoints.size < 3:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)
    expected_pct = np.clip(expected_counts / max(expected_counts.sum(), 1), 1e-6, None)
    actual_pct = np.clip(actual_counts / max(actual_counts.sum(), 1), 1e-6, None)
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return round(float(psi), 6)


def drift_status(psi: float) -> str:
    if psi >= DRIFT_SEVERE:
        return "severe"
    if psi >= DRIFT_STABLE:
        return "moderate"
    return "stable"


def compute_drift_report(
    baseline_rows: list[list[float]],
    current_rows: list[list[float]],
    feature_names: list[str],
) -> dict[str, object]:
    baseline = np.asarray(baseline_rows, dtype=float)
    current = np.asarray(current_rows, dtype=float)
    if current.ndim != 2:
        raise ValueError("current rows must be a 2D matrix")
    if baseline.shape[1] != current.shape[1]:
        raise ValueError("baseline and current rows must have the same feature count")

    feature_reports = []
    for idx, feature_name in enumerate(feature_names):
        psi = population_stability_index(baseline[:, idx], current[:, idx])
        feature_reports.append(
            {
                "feature": feature_name,
                "psi": psi,
                "status": drift_status(psi),
            }
        )

    max_psi = max((item["psi"] for item in feature_reports), default=0.0)
    drifted = sum(1 for item in feature_reports if item["status"] != "stable")
    overall = "severe" if any(item["status"] == "severe" for item in feature_reports) else (
        "moderate" if drifted else "stable"
    )
    return {
        "rows_checked": int(current.shape[0]),
        "drifted_features": int(drifted),
        "max_psi": round(float(max_psi), 6),
        "status": overall,
        "features": feature_reports,
    }


def append_prediction_log(log_path: Path, record: dict[str, object]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")

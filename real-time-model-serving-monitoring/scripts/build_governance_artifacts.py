from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ml_monitoring.governance import retraining_decision, shadow_model_agreement
from ml_monitoring.model import ModelService
from ml_monitoring.monitoring import LatencyTracker, compute_drift_report


def main() -> None:
    service = ModelService()
    baseline = np.asarray(service.baseline_rows, dtype=float)
    stable_drift = compute_drift_report(baseline, baseline.copy(), service.feature_names)
    shifted = baseline.copy()
    spread = np.nanstd(baseline, axis=0)
    shifted[:, :5] += 2.0 * np.where(spread[:5] > 0, spread[:5], 1.0)
    shifted_drift = compute_drift_report(baseline, shifted, service.feature_names)
    latency = LatencyTracker()
    for value in [8.2, 9.1, 7.8, 10.5, 11.2]:
        latency.record(value)
    decision = retraining_decision(shifted_drift, latency.report())
    agreement = shadow_model_agreement([0.10, 0.25, 0.81], [0.11, 0.22, 0.78])
    out = {
        "stable_drift": stable_drift,
        "shifted_fixture_drift": shifted_drift,
        "latency_fixture": latency.report(),
        "latency_fixture_note": "Deterministic governance fixture; use load_benchmark.json for measured HTTP latency.",
        "retraining_decision_on_shifted_fixture": decision,
        "shadow_agreement_fixture": agreement,
    }
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "governance_readiness.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

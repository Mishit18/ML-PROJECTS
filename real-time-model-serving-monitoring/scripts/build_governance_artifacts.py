from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ml_monitoring.governance import retraining_decision, shadow_model_agreement
from ml_monitoring.model import ModelService
from ml_monitoring.monitoring import LatencyTracker, compute_drift_report


def main() -> None:
    service = ModelService()
    baseline = service.baseline_rows[:50]
    drift = compute_drift_report(service.baseline_rows, baseline, service.feature_names)
    latency = LatencyTracker()
    for value in [8.2, 9.1, 7.8, 10.5, 11.2]:
        latency.record(value)
    decision = retraining_decision(drift, latency.report())
    agreement = shadow_model_agreement([0.10, 0.25, 0.81], [0.11, 0.22, 0.78])
    out = {"drift": drift, "latency": latency.report(), "retraining_decision": decision, "shadow_agreement": agreement}
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "governance_readiness.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

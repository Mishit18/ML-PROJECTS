import numpy as np

from src.ml_monitoring.monitoring import (
    LatencyTracker,
    compute_drift_report,
    population_stability_index,
    validate_feature_vector,
)


def test_latency_tracker_reports_percentiles():
    tracker = LatencyTracker()
    for value in [1, 2, 3, 4, 5]:
        tracker.record(value)
    report = tracker.report()
    assert report["request_count"] == 5
    assert report["p50_ms"] == 3.0
    assert report["max_ms"] == 5.0


def test_population_stability_index_zero_for_same_distribution():
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert population_stability_index(values, values) == 0.0


def test_compute_drift_report_detects_shift():
    baseline = [[1.0, 10.0], [2.0, 11.0], [3.0, 12.0], [4.0, 13.0]]
    current = [[100.0, 10.0], [110.0, 11.0], [120.0, 12.0], [130.0, 13.0]]
    report = compute_drift_report(baseline, current, ["a", "b"])
    assert report["rows_checked"] == 4
    assert report["max_psi"] > 0


def test_validate_feature_vector_rejects_wrong_count():
    try:
        validate_feature_vector([1.0, 2.0], expected_count=3)
    except ValueError as exc:
        assert "expected 3 features" in str(exc)
    else:
        raise AssertionError("expected ValueError")

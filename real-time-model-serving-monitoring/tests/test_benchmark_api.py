from scripts.benchmark_api import percentile, summarize


def test_percentile_uses_nearest_rank() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(values, 0.50) == 3.0
    assert percentile(values, 0.95) == 5.0


def test_summarize_reports_load_metrics() -> None:
    report = summarize(
        [10.0, 20.0, 30.0, 40.0],
        requests_count=5,
        failures=1,
        wall_seconds=2.0,
        concurrency=4,
    )
    assert report["successful_requests"] == 4
    assert report["error_rate_pct"] == 20.0
    assert report["throughput_rps"] == 2.0
    assert report["p95_ms"] == 40.0

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAYLOAD = PROJECT_ROOT / "examples" / "sample_request.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "load_benchmark.json"
_thread_local = threading.local()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)
    return ordered[max(0, index)]


def summarize(
    latencies_ms: list[float],
    *,
    requests_count: int,
    failures: int,
    wall_seconds: float,
    concurrency: int,
) -> dict[str, float | int]:
    if not latencies_ms:
        raise ValueError("benchmark completed without a successful request")
    return {
        "requests": requests_count,
        "successful_requests": len(latencies_ms),
        "failures": failures,
        "error_rate_pct": round(100 * failures / requests_count, 4),
        "concurrency": concurrency,
        "throughput_rps": round(len(latencies_ms) / wall_seconds, 4),
        "avg_ms": round(statistics.mean(latencies_ms), 4),
        "p50_ms": round(percentile(latencies_ms, 0.50), 4),
        "p95_ms": round(percentile(latencies_ms, 0.95), 4),
        "p99_ms": round(percentile(latencies_ms, 0.99), 4),
        "max_ms": round(max(latencies_ms), 4),
        "wall_seconds": round(wall_seconds, 4),
    }


def request_once(url: str, payload: dict) -> tuple[bool, float]:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        _thread_local.session = session
    start = time.perf_counter()
    try:
        response = session.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return True, (time.perf_counter() - start) * 1000
    except requests.RequestException:
        return False, (time.perf_counter() - start) * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark API latency, throughput, and error rate.")
    parser.add_argument("--url", default="http://localhost:8000/predict")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--payload", type=Path, default=DEFAULT_PAYLOAD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.requests <= 0 or args.concurrency <= 0 or args.warmup < 0:
        parser.error("requests and concurrency must be positive; warmup cannot be negative")

    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    for _ in range(args.warmup):
        ok, _ = request_once(args.url, payload)
        if not ok:
            raise RuntimeError("warmup request failed; confirm that the API is running")

    latencies: list[float] = []
    failures = 0
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(request_once, args.url, payload) for _ in range(args.requests)]
        for future in as_completed(futures):
            ok, latency_ms = future.result()
            if ok:
                latencies.append(latency_ms)
            else:
                failures += 1
    wall_seconds = time.perf_counter() - wall_start
    report = summarize(
        latencies,
        requests_count=args.requests,
        failures=failures,
        wall_seconds=wall_seconds,
        concurrency=args.concurrency,
    )
    report.update({"url": args.url, "warmup_requests": args.warmup})
    report["environment"] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "client": "requests.Session per worker thread",
        "server": "single local Uvicorn process",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

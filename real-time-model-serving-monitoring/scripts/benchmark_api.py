from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAYLOAD = PROJECT_ROOT / "examples" / "sample_request.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark prediction endpoint latency.")
    parser.add_argument("--url", default="http://localhost:8000/predict")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--payload", type=Path, default=DEFAULT_PAYLOAD)
    args = parser.parse_args()

    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    latencies = []
    for _ in range(args.requests):
        start = time.perf_counter()
        response = requests.post(args.url, json=payload, timeout=10)
        response.raise_for_status()
        latencies.append((time.perf_counter() - start) * 1000)

    latencies_sorted = sorted(latencies)
    p95_index = int(0.95 * (len(latencies_sorted) - 1))
    report = {
        "requests": args.requests,
        "avg_ms": round(statistics.mean(latencies), 4),
        "p50_ms": round(statistics.median(latencies), 4),
        "p95_ms": round(latencies_sorted[p95_index], 4),
        "max_ms": round(max(latencies), 4),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

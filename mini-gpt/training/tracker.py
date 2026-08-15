"""Lightweight JSON/CSV experiment tracking."""

import csv
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict


class ExperimentTracker:
    def __init__(self, output_dir: str = "experiments", run_name: str = "mini_gpt_run"):
        self.output_dir = output_dir
        self.run_name = run_name
        os.makedirs(output_dir, exist_ok=True)
        self.jsonl_path = os.path.join(output_dir, f"{run_name}.jsonl")
        self.csv_path = os.path.join(output_dir, f"{run_name}.csv")
        self._csv_fields = None

    def log(self, metrics: Dict[str, Any]) -> None:
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **metrics,
        }
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

        fieldnames = list(row.keys())
        write_header = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def cuda_memory_mb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024**2)
    except Exception:
        pass
    return 0.0

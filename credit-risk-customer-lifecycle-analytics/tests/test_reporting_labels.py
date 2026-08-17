import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from credit_lifecycle.reporting import _benchmark_summary


def test_benchmark_summary_uses_dataset_name() -> None:
    benchmark = pd.DataFrame(
        [
            {
                "dataset": "UCI Default of Credit Card Clients",
                "model": "gradient_boosting",
                "roc_auc": 0.775,
                "ks": 0.421,
                "records": 30_000,
            }
        ]
    )

    summary = _benchmark_summary(benchmark)
    assert "Real UCI Default of Credit Card Clients benchmark" in summary
    assert "OpenML German Credit" not in summary

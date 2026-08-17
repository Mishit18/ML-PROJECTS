from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_lifecycle.home_credit import run_home_credit_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the real Home Credit portfolio pipeline")
    parser.add_argument("--archive", type=Path, required=True, help="Official Kaggle competition ZIP")
    args = parser.parse_args()
    summary = run_home_credit_pipeline(
        archive_path=args.archive,
        output_dir=ROOT / "outputs",
        artifact_dir=ROOT / "artifacts",
        report_dir=ROOT / "reports",
    )
    print("Home Credit real-data pipeline complete")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

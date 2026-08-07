from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.config import load_config, save_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a JSON-defined LoRA ablation grid.")
    parser.add_argument("--grid", default="configs/ablation_grid.json")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.grid, "r", encoding="utf-8") as f:
        grid = json.load(f)

    base_config = load_config(grid["base_config"])
    shared = grid.get("shared_overrides", {})
    generated_dir = Path("configs/generated")
    generated_dir.mkdir(parents=True, exist_ok=True)

    for experiment in grid["experiments"]:
        name = experiment["name"]
        overrides = {**shared, **{k: v for k, v in experiment.items() if k != "name"}}
        config = load_config(grid["base_config"], overrides=overrides)
        config.run_name = f"{base_config.run_name}-{name}"
        config.output_dir = f"outputs/ablations/{name}"
        config_path = generated_dir / f"{name}.json"
        save_config(config, config_path)

        command = [sys.executable, "-m", "src.train", "--config", str(config_path)]
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

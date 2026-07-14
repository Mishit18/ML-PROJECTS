import argparse
import csv
import json
import subprocess
from pathlib import Path


def read_last_row(log_path: Path):
    if not log_path.exists():
        return None
    with open(log_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def get_gpu_summary() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,power.draw",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "nvidia-smi unavailable"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/cifar10_rtx4060_best")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    total_steps = config.get("train", {}).get("max_steps")
    row = read_last_row(run_dir / "train_log.csv")
    print(f"run_dir: {run_dir}")
    print(f"gpu: {get_gpu_summary()}")
    if row is None:
        print("no training rows logged yet")
        return
    step = int(row["step"])
    progress = f"{100 * step / total_steps:.2f}%" if total_steps else "unknown"
    loss = row.get("loss", "n/a")
    lr = row.get("lr", "n/a")
    epoch = row.get("epoch", "n/a")
    print(f"step: {step} / {total_steps or 'unknown'} ({progress})")
    print(f"epoch: {epoch}")
    print(f"loss: {loss}")
    print(f"lr: {lr}")


if __name__ == "__main__":
    main()

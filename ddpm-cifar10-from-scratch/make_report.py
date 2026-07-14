import argparse
import csv
import json
from pathlib import Path


def read_last_train_row(run_dir: Path):
    path = run_dir / "train_log.csv"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def read_metrics(run_dir: Path):
    path = run_dir / "metrics" / "metrics.csv"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/cifar10_rtx4060_best")
    parser.add_argument("--out", default="results/REPORT.md")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    train_row = read_last_train_row(run_dir)
    metrics = read_metrics(run_dir)

    lines = ["# DDPM CIFAR-10 Report", ""]
    if config:
        lines.extend(
            [
                "## Configuration",
                "",
                f"- Run directory: `{run_dir}`",
                f"- Model base channels: `{config['model']['base_channels']}`",
                f"- Attention resolutions: `{config['model']['attention_resolutions']}`",
                f"- Noise schedule: `{config['diffusion']['schedule']}`",
                f"- Diffusion steps: `{config['diffusion']['timesteps']}`",
                f"- Training target steps: `{config['train'].get('max_steps', 'epochs-only')}`",
                "",
            ]
        )
    if train_row:
        lines.extend(
            [
                "## Latest Training State",
                "",
                f"- Step: `{train_row['step']}`",
                f"- Epoch: `{train_row['epoch']}`",
                f"- Loss: `{train_row['loss']}`",
                f"- LR: `{train_row['lr']}`",
                "",
            ]
        )
    lines.extend(["## Evaluation Metrics", ""])
    if metrics:
        header = "| weights | sampler | steps | samples | FID | IS | samples/sec |"
        sep = "| --- | --- | ---: | ---: | ---: | ---: | ---: |"
        lines.extend([header, sep])
        for row in metrics:
            is_value = f"{float(row['inception_score_mean']):.3f} +/- {float(row['inception_score_std']):.3f}"
            lines.append(
                f"| {row['weights']} | {row['sampler']} | {row['steps']} | {row['num_samples']} | "
                f"{float(row['fid']):.3f} | {is_value} | {float(row['samples_per_second']):.2f} |"
            )
    else:
        lines.append("No evaluation metrics have been written yet.")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

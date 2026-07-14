import json
import random
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import make_grid, save_image
import yaml


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_config(config: dict, out_dir: str | Path) -> None:
    path = ensure_dir(out_dir) / "config.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def denorm(x: torch.Tensor) -> torch.Tensor:
    return ((x.clamp(-1, 1) + 1) * 0.5).clamp(0, 1)


def save_sample_grid(samples: torch.Tensor, path: str | Path, nrow: int = 8) -> None:
    grid = make_grid(denorm(samples), nrow=nrow)
    save_image(grid, path)


def latest_checkpoint(run_dir: str | Path) -> Path:
    checkpoints = sorted(Path(run_dir).glob("checkpoint_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint_*.pt found in {run_dir}")
    return checkpoints[-1]

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass
class ExperimentConfig:
    run_name: str = "mistral7b-alpaca-qlora"
    model_name: str = "mistralai/Mistral-7B-v0.3"
    dataset_name: str = "yahma/alpaca-cleaned"
    dataset_format: str = "alpaca"
    output_dir: str = "outputs/mistral7b-alpaca-qlora"
    max_seq_length: int = 1024
    validation_size: int = 1000
    max_train_samples: int | None = None
    seed: int = 42

    use_qlora: bool = True
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"

    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    bias: str = "none"

    num_train_epochs: float = 1.0
    max_steps: int = 1200
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 2
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    packing: bool = False
    report_to: str = "none"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["lora_target_modules"] = list(self.lora_target_modules)
        return data


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if overrides:
        raw.update(overrides)

    field_names = {field.name for field in fields(ExperimentConfig)}
    unknown = sorted(set(raw) - field_names)
    if unknown:
        raise ValueError(f"Unknown config keys in {path}: {unknown}")
    if "lora_target_modules" in raw:
        raw["lora_target_modules"] = tuple(raw["lora_target_modules"])
    return ExperimentConfig(**raw)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)

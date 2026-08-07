from __future__ import annotations

from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.config import ExperimentConfig


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype {dtype_name!r}. Choose one of {sorted(mapping)}.")
    return mapping[dtype_name]


def build_bnb_config(config: ExperimentConfig) -> BitsAndBytesConfig | None:
    if not config.load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=resolve_torch_dtype(config.bnb_4bit_compute_dtype),
    )


def load_tokenizer(config: ExperimentConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(config: ExperimentConfig):
    quantization_config = build_bnb_config(config)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=quantization_config,
        torch_dtype=resolve_torch_dtype(config.bnb_4bit_compute_dtype),
        device_map="auto",
    )
    model.config.use_cache = False
    return model


def apply_lora(model, config: ExperimentConfig):
    if config.use_qlora:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.gradient_checkpointing,
        )

    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=list(config.lora_target_modules),
        lora_dropout=config.lora_dropout,
        bias=config.bias,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def load_model_and_tokenizer(config: ExperimentConfig):
    tokenizer = load_tokenizer(config)
    model = load_base_model(config)
    model = apply_lora(model, config)
    return model, tokenizer


def count_parameters(model) -> dict[str, int | float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    percent = 100 * trainable / total if total else 0.0
    return {
        "total": total,
        "trainable": trainable,
        "trainable_percent": percent,
        "frozen": total - trainable,
    }


def save_adapter(model, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)


def merge_adapter(
    base_model_name: str,
    adapter_dir: str | Path,
    output_dir: str | Path,
    torch_dtype: torch.dtype = torch.float16,
) -> None:
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    merged = model.merge_and_unload()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir, safe_serialization=True)

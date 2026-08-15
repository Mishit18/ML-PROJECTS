"""LoRA adapters for attention projections."""

import math
from typing import Iterable, Tuple

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling


def freeze_base_model(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def add_lora_adapters(
    model: nn.Module,
    target_suffixes: Iterable[str] = ("q_proj", "k_proj", "v_proj", "out_proj"),
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
) -> nn.Module:
    freeze_base_model(model)
    targets = tuple(target_suffixes)
    for module_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full_name = f"{module_name}.{child_name}" if module_name else child_name
            if isinstance(child, nn.Linear) and full_name.endswith(targets):
                setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
    return model


def trainable_parameter_summary(model: nn.Module) -> Tuple[int, int, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / total if total else 0.0
    return trainable, total, pct

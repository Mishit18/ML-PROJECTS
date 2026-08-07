from __future__ import annotations

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    """A minimal LoRA wrapper for one frozen nn.Linear layer."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)

        in_features = base_layer.in_features
        out_features = base_layer.out_features
        self.lora_a = nn.Parameter(torch.empty(rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
        self.reset_parameters()

        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frozen_out = self.base_layer(x)
        update = (self.dropout(x) @ self.lora_a.T) @ self.lora_b.T
        return frozen_out + self.scaling * update

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        delta_w = self.scaling * (self.lora_b @ self.lora_a)
        return self.base_layer.weight + delta_w.to(self.base_layer.weight.dtype)


def lora_parameter_count(in_features: int, out_features: int, rank: int) -> int:
    return rank * (in_features + out_features)


def dense_parameter_count(in_features: int, out_features: int) -> int:
    return in_features * out_features

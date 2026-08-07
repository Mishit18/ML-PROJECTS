import torch
from torch import nn

from src.manual_lora import LoRALinear, dense_parameter_count, lora_parameter_count


def test_lora_linear_preserves_shape_and_freezes_base():
    base = nn.Linear(5, 7)
    layer = LoRALinear(base, rank=2, alpha=4, dropout=0.0)
    x = torch.randn(3, 5)

    y = layer(x)

    assert y.shape == (3, 7)
    assert not layer.base_layer.weight.requires_grad
    assert layer.lora_a.requires_grad
    assert layer.lora_b.requires_grad


def test_lora_parameter_count_is_smaller_than_dense():
    assert lora_parameter_count(4096, 4096, 8) == 65536
    assert lora_parameter_count(4096, 4096, 8) < dense_parameter_count(4096, 4096)

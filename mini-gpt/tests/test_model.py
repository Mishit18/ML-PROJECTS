import os
import sys

import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.attention import MultiHeadAttention
from model.gpt import GPT, GPTConfig
from model.lora import LoRALinear, add_lora_adapters, trainable_parameter_summary
from training.loss import compute_language_modeling_loss


def tiny_config(**overrides):
    cfg = dict(
        vocab_size=32,
        max_seq_len=8,
        d_model=16,
        num_layers=2,
        num_heads=4,
        d_ff=32,
        dropout=0.0,
        bias=True,
        position_embedding_type="rope",
        norm_type="rmsnorm",
        num_kv_heads=2,
        use_flash_attention=True,
        ffn_activation="swiglu",
    )
    cfg.update(overrides)
    return GPTConfig(**cfg)


def test_causal_mask_blocks_future_tokens():
    torch.manual_seed(0)
    attn = MultiHeadAttention(
        d_model=16,
        num_heads=4,
        dropout=0.0,
        use_flash_attention=False,
        use_rope=False,
    ).eval()
    x = torch.randn(1, 4, 16)
    x_changed = x.clone()
    x_changed[:, 3, :] += 100.0

    out, _ = attn(x)
    out_changed, _ = attn(x_changed)

    torch.testing.assert_close(out[:, :3, :], out_changed[:, :3, :], atol=1e-5, rtol=1e-5)


def test_logits_shape_is_correct():
    model = GPT(tiny_config()).eval()
    input_ids = torch.randint(0, 32, (3, 5))
    outputs = model(input_ids)
    assert outputs["logits"].shape == (3, 5, 32)


def test_kv_cache_output_matches_non_cache_output():
    torch.manual_seed(0)
    model = GPT(tiny_config()).eval()
    input_ids = torch.randint(0, 32, (2, 6))

    with torch.no_grad():
        full_logits = model(input_ids, use_cache=False)["logits"]
        kv_caches = None
        cached_logits = []
        for pos in range(input_ids.size(1)):
            outputs = model(
                input_ids[:, pos : pos + 1],
                kv_caches=kv_caches,
                use_cache=True,
                start_pos=pos,
            )
            kv_caches = outputs["kv_caches"]
            cached_logits.append(outputs["logits"])
        cached_logits = torch.cat(cached_logits, dim=1)

    torch.testing.assert_close(full_logits, cached_logits, atol=1e-4, rtol=1e-4)


def test_generation_does_not_exceed_context_length():
    model = GPT(tiny_config(max_seq_len=6)).eval()
    input_ids = torch.randint(0, 32, (1, 4))
    generated = model.generate(input_ids, max_new_tokens=10, top_k=1, use_cache=True)
    assert generated.size(1) == 6


def test_loss_ignores_padding_correctly():
    logits = torch.tensor(
        [
            [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
            [[0.0, 5.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.0, 5.0]],
        ]
    )
    labels = torch.tensor([[0, 1, -100], [1, -100, -100]])
    loss = compute_language_modeling_loss(logits, labels, ignore_index=-100)

    valid_logits = torch.stack([logits[0, 0], logits[0, 1], logits[1, 0]])
    valid_labels = torch.tensor([0, 1, 1])
    expected = F.cross_entropy(valid_logits, valid_labels)
    torch.testing.assert_close(loss, expected)


def test_lora_freezes_base_and_keeps_trainable_params_small():
    model = GPT(tiny_config(vocab_size=50257, d_model=64, num_heads=4, num_kv_heads=2, d_ff=128))
    add_lora_adapters(model, rank=4, alpha=8)
    trainable, total, pct = trainable_parameter_summary(model)

    assert pct < 1.0
    assert trainable > 0
    assert total > trainable

    lora_modules = [module for module in model.modules() if isinstance(module, LoRALinear)]
    assert lora_modules
    assert all(not param.requires_grad for module in lora_modules for param in module.base.parameters())
    assert all(module.lora_a.weight.requires_grad and module.lora_b.weight.requires_grad for module in lora_modules)

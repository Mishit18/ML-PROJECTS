"""
Multi-Head Self-Attention implementation from first principles.

This module implements the core attention mechanism:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

With:
- Multi-head splitting
- Causal masking for autoregressive generation
- Numerical stability considerations
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.size(-1) // 2]
    x2 = x[..., x.size(-1) // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings used by modern decoder-only LMs."""

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE head dimension must be even")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, start_pos: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(-2)
        cos = self.cos_cached[:, :, start_pos : start_pos + seq_len, :].to(dtype=q.dtype, device=q.device)
        sin = self.sin_cached[:, :, start_pos : start_pos + seq_len, :].to(dtype=q.dtype, device=q.device)
        return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.
    
    Architecture:
    1. Project input to Q, K, V
    2. Split into multiple heads
    3. Compute scaled dot-product attention per head
    4. Concatenate heads
    5. Final output projection
    
    Shape notation:
        B = batch_size
        T = sequence_length
        d = d_model (embedding dimension)
        h = num_heads
        d_k = d_head = d // h (dimension per head)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        num_kv_heads: Optional[int] = None,
        use_flash_attention: bool = True,
        use_rope: bool = False,
        max_seq_len: int = 1024,
    ):
        """
        Args:
            d_model: Model dimension (embedding size)
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in projections
        """
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads for GQA")
        self.d_head = d_model // num_heads  # Dimension per head
        self.kv_dim = self.num_kv_heads * self.d_head
        self.use_flash_attention = use_flash_attention and hasattr(F, "scaled_dot_product_attention")
        self.use_rope = use_rope
        
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, self.kv_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, self.kv_dim, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.rope = RotaryEmbedding(self.d_head, max_seq_len=max_seq_len) if use_rope else None
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Scaling factor for numerical stability
        self.scale = 1.0 / math.sqrt(self.d_head)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of multi-head attention.
        
        Args:
            x: Input tensor of shape (B, T, d)
            attention_mask: Optional mask of shape (B, T) where 1=attend, 0=ignore
            kv_cache: Optional cached (K, V) from previous steps for inference
            use_cache: Whether to return K, V for caching
        
        Returns:
            output: Attention output of shape (B, T, d)
            new_kv_cache: Optional (K, V) cache if use_cache=True
        """
        B, T, d = x.shape
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 2. Reshape for multi-head attention
        # (B, T, d) -> (B, T, h, d_k) -> (B, h, T, d_k)
        q = q.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.d_head).transpose(1, 2)
        if self.rope is not None:
            q, k = self.rope(q, k, start_pos=start_pos)
        
        # 3. Handle KV cache for inference (autoregressive generation)
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            # Concatenate cached K, V with new K, V
            k = torch.cat([k_cache, k], dim=2)  # (B, h, T_prev + T, d_k)
            v = torch.cat([v_cache, v], dim=2)
        k_cache_out, v_cache_out = k, v
        
        T_kv = k.size(2)
        causal_mask = self._get_causal_mask(T, T_kv, device=x.device).to(torch.bool)
        if attention_mask is not None:
            causal_mask = causal_mask & attention_mask[:, None, None, :].to(torch.bool)

        if self.num_kv_heads != self.num_heads:
            repeats = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        if self.use_flash_attention:
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=causal_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=False,
            )
        else:
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn_scores = attn_scores.masked_fill(causal_mask == 0, float("-inf"))
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            attn_weights = self.attn_dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)
        
        # 9. Concatenate heads
        # (B, h, T, d_k) -> (B, T, h, d_k) -> (B, T, d)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, d)
        
        # 10. Final output projection
        output = self.out_proj(attn_output)
        output = self.resid_dropout(output)
        
        # 11. Return cache if requested
        new_kv_cache = (k_cache_out, v_cache_out) if use_cache else None
        
        return output, new_kv_cache
    
    def _get_causal_mask(self, T_q: int, T_kv: int, device: torch.device) -> torch.Tensor:
        """
        Create causal mask to prevent attending to future positions.
        
        Args:
            T_q: Query sequence length
            T_kv: Key/value sequence length
            device: Device to create mask on
        
        Returns:
            Causal mask of shape (1, 1, T_q, T_kv)
            where mask[i, j] = 1 if position i can attend to position j
        """
        mask = torch.tril(torch.ones(T_q, T_kv, device=device))
        
        if T_kv > T_q:
            offset = T_kv - T_q
            mask = torch.ones(T_q, T_kv, device=device)
            mask[:, offset:] = torch.tril(torch.ones(T_q, T_q, device=device))
        
        return mask.view(1, 1, T_q, T_kv)


class CausalSelfAttention(MultiHeadAttention):
    """
    Alias for MultiHeadAttention with causal masking.
    This is the standard attention used in GPT-style models.
    """
    pass

"""
Multi-Head Self-Attention implementation from first principles.

This module implements the core attention mechanism:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

With:
- Multi-head splitting
- Causal masking for autoregressive generation
- Numerical stability considerations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


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
        self.d_head = d_model // num_heads  # Dimension per head
        
        # QKV projections (combined for efficiency)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
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
        
        # 1. Project to Q, K, V
        # qkv: (B, T, 3*d)
        qkv = self.qkv_proj(x)
        
        # Split into Q, K, V: each (B, T, d)
        q, k, v = qkv.split(self.d_model, dim=-1)
        
        # 2. Reshape for multi-head attention
        # (B, T, d) -> (B, T, h, d_k) -> (B, h, T, d_k)
        q = q.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        
        # 3. Handle KV cache for inference (autoregressive generation)
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            # Concatenate cached K, V with new K, V
            k = torch.cat([k_cache, k], dim=2)  # (B, h, T_prev + T, d_k)
            v = torch.cat([v_cache, v], dim=2)
        
        # 4. Compute attention scores
        # Q @ K^T: (B, h, T, d_k) @ (B, h, d_k, T_kv) -> (B, h, T, T_kv)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 5. Apply causal mask
        T_kv = k.size(2)
        causal_mask = self._get_causal_mask(T, T_kv, device=x.device)
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # 6. Apply padding mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # 7. Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 8. Apply attention to values
        # (B, h, T, T_kv) @ (B, h, T_kv, d_k) -> (B, h, T, d_k)
        attn_output = torch.matmul(attn_weights, v)
        
        # 9. Concatenate heads
        # (B, h, T, d_k) -> (B, T, h, d_k) -> (B, T, d)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, d)
        
        # 10. Final output projection
        output = self.out_proj(attn_output)
        output = self.resid_dropout(output)
        
        # 11. Return cache if requested
        new_kv_cache = (k, v) if use_cache else None
        
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

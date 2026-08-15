"""
Transformer block implementation.

Each block contains:
1. Pre-LayerNorm
2. Multi-Head Self-Attention
3. Residual connection
4. Pre-LayerNorm
5. Feed-Forward Network (MLP)
6. Residual connection

This follows the GPT-2 architecture (Pre-LN variant).
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from .attention import CausalSelfAttention


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    Architecture:
        FFN(x) = GELU(xW1 + b1)W2 + b2
    
    Typically expands dimension by 4x in the hidden layer.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        bias: bool = True,
        activation: str = "swiglu",
    ):
        """
        Args:
            d_model: Model dimension
            d_ff: Feed-forward hidden dimension (typically 4 * d_model)
            dropout: Dropout probability
            bias: Whether to use bias
        """
        super().__init__()
        
        self.activation_name = activation
        if activation == "swiglu":
            self.fc1 = nn.Linear(d_model, 2 * d_ff, bias=bias)
        else:
            self.fc1 = nn.Linear(d_model, d_ff, bias=bias)
        self.fc2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        
        # GELU activation (used in GPT-2, GPT-3)
        # Alternative: ReLU (original Transformer)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape (B, T, d_model)
        
        Returns:
            Output of shape (B, T, d_model)
        """
        x = self.fc1(x)
        if self.activation_name == "swiglu":
            gate, value = x.chunk(2, dim=-1)
            x = nn.functional.silu(gate) * value
        else:
            x = self.activation(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.dropout(x)
        
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer decoder block.
    
    Architecture (Pre-LN variant):
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
    
    This is the GPT-2 style architecture, which is more stable than Post-LN.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        bias: bool = True,
        norm_type: str = "rmsnorm",
        num_kv_heads: Optional[int] = None,
        use_flash_attention: bool = True,
        use_rope: bool = False,
        max_seq_len: int = 1024,
        ffn_activation: str = "swiglu",
    ):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout probability
            bias: Whether to use bias in linear layers
        """
        super().__init__()
        
        norm_cls = RMSNorm if norm_type == "rmsnorm" else nn.LayerNorm
        self.ln1 = norm_cls(d_model)
        self.ln2 = norm_cls(d_model)
        
        # Multi-head self-attention
        self.attention = CausalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            num_kv_heads=num_kv_heads,
            use_flash_attention=use_flash_attention,
            use_rope=use_rope,
            max_seq_len=max_seq_len,
        )
        
        # Feed-forward network
        self.ffn = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            bias=bias,
            activation=ffn_activation,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor of shape (B, T, d_model)
            attention_mask: Optional attention mask of shape (B, T)
            kv_cache: Optional cached (K, V) from previous forward pass
            use_cache: Whether to return K, V cache
        
        Returns:
            output: Output tensor of shape (B, T, d_model)
            new_kv_cache: Optional (K, V) cache if use_cache=True
        """
        normed = self.ln1(x)
        
        attn_output, new_kv_cache = self.attention(
            normed,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            use_cache=use_cache,
            start_pos=start_pos,
        )
        
        x = x + attn_output
        
        normed = self.ln2(x)
        ffn_output = self.ffn(normed)
        x = x + ffn_output
        
        return x, new_kv_cache


class RMSNorm(nn.Module):
    """Root-mean-square normalization used in LLaMA/Mistral-style blocks."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return self.weight * x * torch.rsqrt(variance + self.eps)

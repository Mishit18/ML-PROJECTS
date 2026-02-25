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
    ):
        """
        Args:
            d_model: Model dimension
            d_ff: Feed-forward hidden dimension (typically 4 * d_model)
            dropout: Dropout probability
            bias: Whether to use bias
        """
        super().__init__()
        
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
        
        # Layer normalization (Pre-LN)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # Multi-head self-attention
        self.attention = CausalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
        )
        
        # Feed-forward network
        self.ffn = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            bias=bias,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
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
        )
        
        x = x + attn_output
        
        normed = self.ln2(x)
        ffn_output = self.ffn(normed)
        x = x + ffn_output
        
        return x, new_kv_cache

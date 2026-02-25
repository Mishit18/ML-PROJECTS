"""
Self-attention mechanisms for spatial feature processing.

Implements multi-head self-attention that allows the model to capture
long-range dependencies in the spatial dimensions of feature maps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    """
    Multi-head self-attention block for spatial features.
    
    Applies self-attention across spatial locations, allowing the model
    to capture global context. Uses group normalization and residual connections.
    
    Args:
        channels: Number of input/output channels
        num_heads: Number of attention heads
    """
    
    def __init__(self, channels, num_heads=4):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(32, channels)
        
        # QKV projection
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        
        # Output projection
        self.proj_out = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        """
        Args:
            x: (batch, channels, height, width) feature map
            
        Returns:
            (batch, channels, height, width) attended features
        """
        batch, channels, height, width = x.shape
        residual = x
        
        # Normalize
        x = self.norm(x)
        
        # Compute Q, K, V
        qkv = self.qkv(x)  # (batch, 3*channels, height, width)
        qkv = qkv.reshape(batch, 3, self.num_heads, self.head_dim, height * width)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, batch, num_heads, height*width, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (batch, num_heads, hw, hw)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # (batch, num_heads, hw, head_dim)
        out = out.permute(0, 1, 3, 2)  # (batch, num_heads, head_dim, hw)
        out = out.reshape(batch, channels, height, width)
        
        # Output projection
        out = self.proj_out(out)
        
        return out + residual


class SpatialTransformer(nn.Module):
    """
    Transformer block with cross-attention support (for future extensions).
    
    Currently implements self-attention but structured to support
    cross-attention for text conditioning or other modalities.
    
    Args:
        channels: Number of input/output channels
        num_heads: Number of attention heads
        depth: Number of transformer layers
    """
    
    def __init__(self, channels, num_heads=4, depth=1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        self.norm = nn.GroupNorm(32, channels)
        self.proj_in = nn.Conv2d(channels, channels, 1)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(channels, num_heads)
            for _ in range(depth)
        ])
        
        self.proj_out = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        """
        Args:
            x: (batch, channels, height, width) feature map
            
        Returns:
            (batch, channels, height, width) transformed features
        """
        batch, channels, height, width = x.shape
        residual = x
        
        x = self.norm(x)
        x = self.proj_in(x)
        
        # Reshape to sequence: (batch, hw, channels)
        x = x.reshape(batch, channels, height * width).permute(0, 2, 1)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Reshape back to spatial: (batch, channels, height, width)
        x = x.permute(0, 2, 1).reshape(batch, channels, height, width)
        x = self.proj_out(x)
        
        return x + residual


class TransformerBlock(nn.Module):
    """
    Single transformer block with self-attention and feedforward.
    
    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
    """
    
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, dim) sequence
            
        Returns:
            (batch, seq_len, dim) transformed sequence
        """
        # Self-attention with residual
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        
        # Feedforward with residual
        x = x + self.mlp(self.norm2(x))
        
        return x

"""
Building blocks for the UNet architecture.

Implements residual blocks with timestep and class conditioning,
along with downsampling and upsampling operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_group_norm(num_channels, num_groups=32):
    """
    Create GroupNorm with automatic group adjustment.
    
    Handles cases where num_channels < num_groups or not evenly divisible.
    Prevents failures with small channel counts.
    
    Args:
        num_channels: Number of channels to normalize
        num_groups: Desired number of groups (will be adjusted if needed)
        
    Returns:
        nn.GroupNorm layer with appropriate number of groups
    """
    # Start with desired number of groups, but don't exceed channels
    groups = min(num_groups, num_channels)
    
    # Find largest divisor of num_channels that's <= groups
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    
    # Ensure at least 1 group
    groups = max(1, groups)
    
    return nn.GroupNorm(groups, num_channels)


class ResidualBlock(nn.Module):
    """
    Residual block with timestep and optional class conditioning.
    
    Architecture:
        x -> GroupNorm -> SiLU -> Conv -> [+time_emb] -> [+class_emb] ->
             GroupNorm -> SiLU -> Dropout -> Conv -> [+residual]
    
    Supports adaptive group normalization (scale-shift) conditioned on timestep.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        time_emb_dim: Timestep embedding dimension
        class_emb_dim: Class embedding dimension (None if not using)
        dropout: Dropout probability
        use_scale_shift_norm: Use adaptive normalization
    """
    
    def __init__(
        self,
        in_channels,
        out_channels,
        time_emb_dim,
        class_emb_dim=None,
        dropout=0.0,
        use_scale_shift_norm=True,
    ):
        super().__init__()
        self.use_scale_shift_norm = use_scale_shift_norm
        
        # First convolution path
        self.norm1 = get_group_norm(in_channels, num_groups=32)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        # Timestep embedding projection
        if use_scale_shift_norm:
            self.time_emb_proj = nn.Linear(time_emb_dim, out_channels * 2)
        else:
            self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)
        
        # Optional class embedding projection
        self.class_emb_proj = None
        if class_emb_dim is not None:
            if use_scale_shift_norm:
                self.class_emb_proj = nn.Linear(class_emb_dim, out_channels * 2)
            else:
                self.class_emb_proj = nn.Linear(class_emb_dim, out_channels)
        
        # Second convolution path
        self.norm2 = get_group_norm(out_channels, num_groups=32)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Residual connection
        if in_channels != out_channels:
            self.residual_proj = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_proj = nn.Identity()
    
    def forward(self, x, time_emb, class_emb=None):
        """
        Args:
            x: (batch, in_channels, height, width) input features
            time_emb: (batch, time_emb_dim) timestep embeddings
            class_emb: (batch, class_emb_dim) class embeddings (optional)
            
        Returns:
            (batch, out_channels, height, width) output features
        """
        residual = self.residual_proj(x)
        
        # First conv block
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add timestep embedding
        time_emb_out = self.time_emb_proj(F.silu(time_emb))
        
        # Add class embedding if provided
        if class_emb is not None and self.class_emb_proj is not None:
            class_emb_out = self.class_emb_proj(F.silu(class_emb))
            time_emb_out = time_emb_out + class_emb_out
        
        # Apply conditioning
        if self.use_scale_shift_norm:
            # Adaptive group normalization: scale and shift
            scale, shift = time_emb_out.chunk(2, dim=1)
            h = h + shift[:, :, None, None]
            h = h * (1 + scale[:, :, None, None])
        else:
            # Simple addition
            h = h + time_emb_out[:, :, None, None]
        
        # Second conv block
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + residual


class Downsample(nn.Module):
    """
    Downsampling operation using strided convolution.
    
    Reduces spatial dimensions by factor of 2 while preserving channels.
    
    Args:
        channels: Number of channels
    """
    
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, height, width)
            
        Returns:
            (batch, channels, height//2, width//2)
        """
        return self.conv(x)


class Upsample(nn.Module):
    """
    Upsampling operation using nearest-neighbor interpolation + convolution.
    
    Increases spatial dimensions by factor of 2 while preserving channels.
    
    Args:
        channels: Number of channels
    """
    
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, height, width)
            
        Returns:
            (batch, channels, height*2, width*2)
        """
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        return x


class DownBlock(nn.Module):
    """
    Downsampling block: multiple residual blocks + optional attention + downsample.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        time_emb_dim: Timestep embedding dimension
        class_emb_dim: Class embedding dimension
        num_layers: Number of residual blocks
        dropout: Dropout probability
        use_attention: Whether to use attention
        num_heads: Number of attention heads
        use_scale_shift_norm: Use adaptive normalization
        downsample: Whether to downsample at the end
    """
    
    def __init__(
        self,
        in_channels,
        out_channels,
        time_emb_dim,
        class_emb_dim=None,
        num_layers=2,
        dropout=0.0,
        use_attention=False,
        num_heads=4,
        use_scale_shift_norm=True,
        downsample=True,
    ):
        super().__init__()
        self.use_attention = use_attention
        self.downsample = downsample
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                time_emb_dim,
                class_emb_dim,
                dropout,
                use_scale_shift_norm,
            )
            for i in range(num_layers)
        ])
        
        # Attention blocks
        if use_attention:
            from .attention import AttentionBlock
            self.attn_blocks = nn.ModuleList([
                AttentionBlock(out_channels, num_heads)
                for _ in range(num_layers)
            ])
        
        # Downsampling
        if downsample:
            self.downsample_op = Downsample(out_channels)
    
    def forward(self, x, time_emb, class_emb=None):
        """
        Args:
            x: (batch, in_channels, height, width)
            time_emb: (batch, time_emb_dim)
            class_emb: (batch, class_emb_dim)
            
        Returns:
            output: (batch, out_channels, height//2, width//2) if downsample else same spatial dims
            skip: (batch, out_channels, height, width) features before downsampling
        """
        for i, res_block in enumerate(self.res_blocks):
            x = res_block(x, time_emb, class_emb)
            if self.use_attention:
                x = self.attn_blocks[i](x)
        
        skip = x
        
        if self.downsample:
            x = self.downsample_op(x)
        
        return x, skip


class UpBlock(nn.Module):
    """
    Upsampling block: multiple residual blocks + optional attention + upsample.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        time_emb_dim: Timestep embedding dimension
        class_emb_dim: Class embedding dimension
        num_layers: Number of residual blocks
        dropout: Dropout probability
        use_attention: Whether to use attention
        num_heads: Number of attention heads
        use_scale_shift_norm: Use adaptive normalization
        upsample: Whether to upsample at the end
    """
    
    def __init__(
        self,
        in_channels,
        out_channels,
        time_emb_dim,
        class_emb_dim=None,
        num_layers=2,
        dropout=0.0,
        use_attention=False,
        num_heads=4,
        use_scale_shift_norm=True,
        upsample=True,
    ):
        super().__init__()
        self.use_attention = use_attention
        self.upsample = upsample
        
        # Residual blocks (first block takes concatenated skip connection)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(
                in_channels + out_channels if i == 0 else out_channels,
                out_channels,
                time_emb_dim,
                class_emb_dim,
                dropout,
                use_scale_shift_norm,
            )
            for i in range(num_layers)
        ])
        
        # Attention blocks
        if use_attention:
            from .attention import AttentionBlock
            self.attn_blocks = nn.ModuleList([
                AttentionBlock(out_channels, num_heads)
                for _ in range(num_layers)
            ])
        
        # Upsampling
        if upsample:
            self.upsample_op = Upsample(out_channels)
    
    def forward(self, x, skip, time_emb, class_emb=None):
        """
        Args:
            x: (batch, in_channels, height, width)
            skip: (batch, out_channels, height, width) skip connection from encoder
            time_emb: (batch, time_emb_dim)
            class_emb: (batch, class_emb_dim)
            
        Returns:
            (batch, out_channels, height*2, width*2) if upsample else same spatial dims
        """
        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)
        
        for i, res_block in enumerate(self.res_blocks):
            x = res_block(x, time_emb, class_emb)
            if self.use_attention:
                x = self.attn_blocks[i](x)
        
        if self.upsample:
            x = self.upsample_op(x)
        
        return x

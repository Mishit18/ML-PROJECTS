"""
UNet architecture for noise prediction in diffusion models.

Implements a complete UNet with:
- Encoder-decoder structure with skip connections
- Timestep conditioning via sinusoidal embeddings
- Optional class conditioning for classifier-free guidance
- Multi-resolution attention mechanisms
"""

import torch
import torch.nn as nn
from .embeddings import SinusoidalPositionEmbeddings, TimestepEmbedding, ClassEmbedding
from .blocks import ResidualBlock, Downsample, Upsample
from .attention import AttentionBlock


class UNet(nn.Module):
    """
    UNet for predicting noise ε_θ(x_t, t, c) in diffusion models.
    
    Architecture follows DDPM with improvements:
    - Sinusoidal timestep embeddings injected at every layer
    - Self-attention at multiple resolutions
    - Classifier-free guidance support via class conditioning
    
    Args:
        image_size: Input image size (assumes square images)
        in_channels: Number of input channels (3 for RGB)
        model_channels: Base channel count (scaled by channel_mult)
        out_channels: Number of output channels (same as in_channels)
        num_res_blocks: Number of residual blocks per resolution
        attention_resolutions: List of resolutions to apply attention
        channel_mult: Channel multipliers for each resolution level
        num_heads: Number of attention heads
        dropout: Dropout probability
        use_scale_shift_norm: Use adaptive group normalization
        num_classes: Number of classes for conditioning (None for unconditional)
        class_dropout_prob: Probability of dropping class condition (for CFG)
    """
    
    def __init__(
        self,
        image_size=32,
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(16, 8),
        channel_mult=(1, 2, 2, 2),
        num_heads=4,
        dropout=0.0,
        use_scale_shift_norm=True,
        num_classes=None,
        class_dropout_prob=0.1,
    ):
        super().__init__()
        
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_classes = num_classes
        
        # Timestep embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            TimestepEmbedding(model_channels, time_embed_dim),
        )
        
        # Class embedding (for classifier-free guidance)
        self.class_embed = None
        class_emb_dim = None
        if num_classes is not None:
            class_emb_dim = model_channels * 4
            self.class_embed = ClassEmbedding(num_classes, class_emb_dim, class_dropout_prob)
        
        # Input convolution
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Encoder (downsampling path)
        self.down_blocks = nn.ModuleList()
        self.down_attentions = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        current_channels = model_channels
        current_resolution = image_size
        
        for level, mult in enumerate(channel_mult):
            out_channels_level = model_channels * mult
            
            # Residual blocks at this level
            level_blocks = nn.ModuleList()
            level_attentions = nn.ModuleList()
            
            for _ in range(num_res_blocks):
                level_blocks.append(
                    ResidualBlock(
                        in_channels=current_channels,
                        out_channels=out_channels_level,
                        time_emb_dim=time_embed_dim,
                        class_emb_dim=class_emb_dim,
                        dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                )
                current_channels = out_channels_level
                
                # Add attention if needed
                if current_resolution in attention_resolutions:
                    level_attentions.append(AttentionBlock(current_channels, num_heads))
                else:
                    level_attentions.append(nn.Identity())
            
            self.down_blocks.append(level_blocks)
            self.down_attentions.append(level_attentions)
            
            # Downsample (except at the last level)
            if level != len(channel_mult) - 1:
                self.down_samples.append(Downsample(current_channels))
                current_resolution //= 2
            else:
                self.down_samples.append(nn.Identity())
        
        # Middle (bottleneck)
        self.middle_block1 = ResidualBlock(
            current_channels,
            current_channels,
            time_embed_dim,
            class_emb_dim,
            dropout,
            use_scale_shift_norm,
        )
        
        if current_resolution in attention_resolutions:
            self.middle_attention = AttentionBlock(current_channels, num_heads)
        else:
            self.middle_attention = nn.Identity()
        
        self.middle_block2 = ResidualBlock(
            current_channels,
            current_channels,
            time_embed_dim,
            class_emb_dim,
            dropout,
            use_scale_shift_norm,
        )
        
        # Decoder (upsampling path)
        self.up_blocks = nn.ModuleList()
        self.up_attentions = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_channels_level = model_channels * mult
            
            # Residual blocks at this level
            level_blocks = nn.ModuleList()
            level_attentions = nn.ModuleList()
            
            for i in range(num_res_blocks):
                # First block at each level takes concatenated skip connection
                if i == 0:
                    in_ch = current_channels + out_channels_level
                else:
                    in_ch = out_channels_level
                
                level_blocks.append(
                    ResidualBlock(
                        in_channels=in_ch,
                        out_channels=out_channels_level,
                        time_emb_dim=time_embed_dim,
                        class_emb_dim=class_emb_dim,
                        dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                )
                current_channels = out_channels_level
                
                # Add attention if needed
                if current_resolution in attention_resolutions:
                    level_attentions.append(AttentionBlock(current_channels, num_heads))
                else:
                    level_attentions.append(nn.Identity())
            
            self.up_blocks.append(level_blocks)
            self.up_attentions.append(level_attentions)
            
            # Upsample (except at the first level)
            if level != 0:
                self.up_samples.append(Upsample(current_channels))
                current_resolution *= 2
            else:
                self.up_samples.append(nn.Identity())
        
        # Output convolution
        self.output_conv = nn.Sequential(
            nn.GroupNorm(32, current_channels),
            nn.SiLU(),
            nn.Conv2d(current_channels, self.out_channels, 3, padding=1),
        )
    
    def forward(self, x, timesteps, class_labels=None, force_drop_class=False):
        """
        Predict noise ε_θ(x_t, t, c).
        
        Args:
            x: (batch, in_channels, height, width) noisy input
            timesteps: (batch,) timestep values in [0, T-1]
            class_labels: (batch,) class indices (optional)
            force_drop_class: If True, use unconditional generation
            
        Returns:
            (batch, out_channels, height, width) predicted noise
        """
        batch_size = x.shape[0]
        
        # Input validation
        assert x.ndim == 4, f"Expected 4D input (B,C,H,W), got {x.ndim}D with shape {x.shape}"
        assert x.shape[1] == self.in_channels, \
            f"Expected {self.in_channels} input channels, got {x.shape[1]}"
        assert x.shape[2] == x.shape[3] == self.image_size, \
            f"Expected {self.image_size}×{self.image_size} images, got {x.shape[2]}×{x.shape[3]}"
        
        # Timestep validation
        assert timesteps.shape == (batch_size,), \
            f"Expected timesteps shape ({batch_size},), got {timesteps.shape}"
        assert timesteps.dtype in [torch.long, torch.int, torch.int32, torch.int64], \
            f"Timesteps must be integer type, got {timesteps.dtype}"
        
        # Class label validation
        if class_labels is not None:
            assert self.num_classes is not None, \
                "Model not configured for class conditioning (num_classes=None)"
            assert class_labels.shape == (batch_size,), \
                f"Expected class_labels shape ({batch_size},), got {class_labels.shape}"
        
        # Embed timesteps
        time_emb = self.time_embed(timesteps)
        
        # Embed class labels
        class_emb = None
        if self.class_embed is not None and class_labels is not None:
            class_emb = self.class_embed(class_labels, force_drop=force_drop_class)
        
        # Input projection
        h = self.input_conv(x)
        
        # Encoder - collect skip connections
        skips = []
        for level_blocks, level_attentions, downsample in zip(
            self.down_blocks, self.down_attentions, self.down_samples
        ):
            for i, (block, attn) in enumerate(zip(level_blocks, level_attentions)):
                h = block(h, time_emb, class_emb)
                h = attn(h)
                # Save skip connection from last block at each level
                if i == len(level_blocks) - 1:
                    skips.append(h)
            h = downsample(h)
        
        # Middle
        h = self.middle_block1(h, time_emb, class_emb)
        h = self.middle_attention(h)
        h = self.middle_block2(h, time_emb, class_emb)
        
        # Decoder - use skip connections
        for level_blocks, level_attentions, upsample in zip(
            self.up_blocks, self.up_attentions, self.up_samples
        ):
            for i, (block, attn) in enumerate(zip(level_blocks, level_attentions)):
                if i == 0:
                    # Concatenate skip connection for first block at each level
                    skip = skips.pop()
                    h = torch.cat([h, skip], dim=1)
                h = block(h, time_emb, class_emb)
                h = attn(h)
            
            # Upsample after processing blocks at this level
            h = upsample(h)
        
        # Output
        h = self.output_conv(h)
        
        return h

import math
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal embeddings like Transformer positions, indexed by diffusion step."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def norm_groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class TimeMLP(nn.Module):
    def __init__(self, base_channels: int, time_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.GroupNorm(norm_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(norm_groups(out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}")
        self.norm = nn.GroupNorm(norm_groups(channels), channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        y = self.norm(x).flatten(2).transpose(1, 2)
        y, _ = self.attn(y, y, y, need_weights=False)
        y = self.proj(y).transpose(1, 2).reshape(b, c, h, w)
        return x + y


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class TimestepSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, t_emb)
            else:
                x = layer(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_mults: Iterable[int] = (1, 2, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: Iterable[int] = (16, 8),
        num_heads: int = 4,
        dropout: float = 0.1,
        image_size: int = 32,
        time_emb_mult: int = 4,
    ):
        super().__init__()
        channel_mults = tuple(channel_mults)
        attention_resolutions = set(attention_resolutions)
        time_dim = base_channels * time_emb_mult

        self.base_channels = base_channels
        self.time_mlp = TimeMLP(base_channels, time_dim)
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        skip_channels = [base_channels]
        ch = base_channels
        resolution = image_size
        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, out_ch, time_dim, dropout)]
                ch = out_ch
                if resolution in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads))
                self.down_blocks.append(TimestepSequential(*layers))
                skip_channels.append(ch)
            if level != len(channel_mults) - 1:
                self.down_blocks.append(TimestepSequential(Downsample(ch)))
                skip_channels.append(ch)
                resolution //= 2

        self.middle = TimestepSequential(
            ResBlock(ch, ch, time_dim, dropout),
            AttentionBlock(ch, num_heads),
            ResBlock(ch, ch, time_dim, dropout),
        )

        self.up_blocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for block_idx in range(num_res_blocks + 1):
                layers = [ResBlock(ch + skip_channels.pop(), out_ch, time_dim, dropout)]
                ch = out_ch
                if resolution in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads))
                if level != 0 and block_idx == num_res_blocks:
                    layers.append(Upsample(ch))
                    resolution *= 2
                self.up_blocks.append(TimestepSequential(*layers))

        self.out = nn.Sequential(
            nn.GroupNorm(norm_groups(ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, in_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = timestep_embedding(timesteps, self.base_channels)
        t_emb = self.time_mlp(t_emb)

        h = self.input_conv(x)
        skips = [h]
        for block in self.down_blocks:
            h = block(h, t_emb)
            skips.append(h)

        h = self.middle(h, t_emb)

        for block in self.up_blocks:
            h = torch.cat([h, skips.pop()], dim=1)
            h = block(h, t_emb)
        return self.out(h)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

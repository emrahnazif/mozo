# SPDX-License-Identifier: MIT
"""DaViT, Florence-2's vision tower: an image to a 24 x 24 grid of features.

Derived from ``DaViT`` and its parts in Microsoft's ``modeling_florence2.py`` (see
``PROVENANCE.md``), reduced to ``forward_features_unpool`` -- the classification head, final norm
and drop-path are training-only and Florence-2 deletes the first two itself. Module names follow
upstream's so the published state dict loads strictly.

Each stage is a strided convolution (``ConvEmbed``) and pairs of blocks: a *spatial* block (window
attention over 12 x 12 tokens) and a *channel* block (attention across channel groups, the whole
image at once). Every attention and feed-forward sits behind a depthwise 3 x 3 convolution.

Two layouts are kept exactly as upstream writes them, because a convolution on a permuted view is a
channels-last tensor and can take a different kernel than on a contiguous one: ``ConvEmbed``
reshapes then permutes (what ``einops.rearrange`` does), and ``DepthWiseConv2d`` transposes then
views.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import VisionSpec

__all__ = ["DaViT"]


class _PreNorm(nn.Module):
    """``x + fn(norm(x))``; ``norm`` may be absent (the depthwise convolutions)."""

    def __init__(self, norm: nn.Module | None, fn: nn.Module) -> None:
        super().__init__()
        self.norm = norm
        self.fn = fn

    def forward(self, x, size):
        shortcut = x
        x, size = self.fn(self.norm(x) if self.norm is not None else x, size)
        return shortcut + x, size


class _Net(nn.Module):
    """``fc1 -> GELU -> fc2``, named as upstream's ``Mlp.net`` (an ``nn.Sequential``)."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class _Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.net = _Net(dim, hidden)

    def forward(self, x, size):
        return self.net(x), size


class _DepthWiseConv2d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, stride=1)

    def forward(self, x, size):
        batch, _, channels = x.shape
        height, width = size
        x = self.dw(x.transpose(1, 2).view(batch, channels, height, width))
        size = (x.size(-2), x.size(-1))
        return x.flatten(2).transpose(1, 2), size


class _ConvEmbed(nn.Module):
    def __init__(self, patch: int, stride: int, padding: int, in_chans: int, dim: int,
                 prenorm: bool) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(in_chans if prenorm else dim)
        self.prenorm = prenorm

    def forward(self, x, size):
        height, width = size
        if x.dim() == 3:
            if self.prenorm:
                x = self.norm(x)
            # einops.rearrange(x, 'b (h w) c -> b c h w'): a reshape, then a permuted view.
            x = x.reshape(x.shape[0], height, width, x.shape[-1]).permute(0, 3, 1, 2)
        x = self.proj(x)
        batch, channels, height, width = x.shape
        # einops.rearrange(x, 'b c h w -> b (h w) c')
        x = x.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        if not self.prenorm:
            x = self.norm(x)
        return x, (height, width)


class _ChannelAttention(nn.Module):
    def __init__(self, dim: int, groups: int) -> None:
        super().__init__()
        self.groups = groups
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, size):
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.groups, channels // self.groups)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        q = q * (float(tokens) ** -0.5)
        attention = (q.transpose(-1, -2) @ k).softmax(dim=-1)
        x = (attention @ v.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj(x), size


def _window_partition(x, window: int):
    batch, height, width, channels = x.shape
    x = x.view(batch, height // window, window, width // window, window, channels)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window, window, channels)


def _window_reverse(windows, batch: int, window: int, height: int, width: int):
    x = windows.view(batch, height // window, width // window, window, window, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, height, width, -1)


class _WindowAttention(nn.Module):
    def __init__(self, dim: int, heads: int, window: int) -> None:
        super().__init__()
        self.window = window
        self.heads = heads
        self.scale = float(dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, size):
        height, width = size
        batch, _, channels = x.shape
        x = x.view(batch, height, width, channels)
        pad_r = (self.window - width % self.window) % self.window
        pad_b = (self.window - height % self.window) % self.window
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, padded_h, padded_w, _ = x.shape

        x = _window_partition(x, self.window).view(-1, self.window * self.window, channels)
        windows, tokens, _ = x.shape
        qkv = self.qkv(x).reshape(windows, tokens, 3, self.heads, channels // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        q = q * self.scale
        attention = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        x = (attention @ v).transpose(1, 2).reshape(windows, tokens, channels)
        x = self.proj(x)

        x = _window_reverse(x.view(-1, self.window, self.window, channels), batch, self.window,
                            padded_h, padded_w)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :height, :width, :].contiguous()
        return x.view(batch, height * width, channels), size


class _SpatialBlock(nn.Module):
    def __init__(self, dim: int, heads: int, window: int) -> None:
        super().__init__()
        self.conv1 = _PreNorm(None, _DepthWiseConv2d(dim))
        self.window_attn = _PreNorm(nn.LayerNorm(dim), _WindowAttention(dim, heads, window))
        self.conv2 = _PreNorm(None, _DepthWiseConv2d(dim))
        self.ffn = _PreNorm(nn.LayerNorm(dim), _Mlp(dim, dim * 4))

    def forward(self, x, size):
        x, size = self.conv1(x, size)
        x, size = self.window_attn(x, size)
        x, size = self.conv2(x, size)
        return self.ffn(x, size)


class _ChannelBlock(nn.Module):
    def __init__(self, dim: int, groups: int) -> None:
        super().__init__()
        self.conv1 = _PreNorm(None, _DepthWiseConv2d(dim))
        self.channel_attn = _PreNorm(nn.LayerNorm(dim), _ChannelAttention(dim, groups))
        self.conv2 = _PreNorm(None, _DepthWiseConv2d(dim))
        self.ffn = _PreNorm(nn.LayerNorm(dim), _Mlp(dim, dim * 4))

    def forward(self, x, size):
        x, size = self.conv1(x, size)
        x, size = self.channel_attn(x, size)
        x, size = self.conv2(x, size)
        return self.ffn(x, size)


class _Pair(nn.Module):
    """One spatial block then one channel block (upstream's inner ``MySequential``)."""

    def __init__(self, dim: int, heads: int, groups: int, window: int) -> None:
        super().__init__()
        self.spatial_block = _SpatialBlock(dim, heads, window)
        self.channel_block = _ChannelBlock(dim, groups)

    def forward(self, x, size):
        x, size = self.spatial_block(x, size)
        return self.channel_block(x, size)


class DaViT(nn.Module):
    """``pixel_values (B, 3, 768, 768)`` to ``(B, 576, dims[-1])`` features, row-major 24 x 24."""

    def __init__(self, spec: VisionSpec) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            _ConvEmbed(spec.patch[i], spec.stride[i], spec.padding[i],
                       3 if i == 0 else spec.dims[i - 1], spec.dims[i], spec.prenorm[i])
            for i in range(4))
        # ``blocks[stage][pair]`` -- keys ``blocks.{i}.{j}.spatial_block...``, as published.
        self.blocks = nn.ModuleList(
            nn.ModuleList(_Pair(spec.dims[i], spec.heads[i], spec.groups[i], spec.window)
                          for _ in range(spec.depths[i]))
            for i in range(4))

    def forward(self, x: torch.Tensor, hooks: dict | None = None) -> torch.Tensor:
        size = (x.size(2), x.size(3))
        for stage, (conv, pairs) in enumerate(zip(self.convs, self.blocks)):
            x, size = conv(x, size)
            for pair in pairs:
                x, size = pair(x, size)
            if hooks is not None:
                hooks[f"vision_stage_{stage}"] = x
        return x

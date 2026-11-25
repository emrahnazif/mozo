# SPDX-License-Identifier: Apache-2.0
"""Swin-T as timm builds it: ``swin_tiny_patch4_window7_224`` at 640, ``always_partition``.

Extracted from ``timm.models.swin_transformer`` (timm 1.0.28, Apache-2.0; Swin itself is
Microsoft's, MIT), keeping only the inference path the published checkpoint runs through
transformers' ``TimmBackbone``. This is timm's arrangement of Swin, which differs from Microsoft's
original in ways that move numbers:

- **The map is rolled before it is padded**, and the shift mask is computed on the padded grid.
  160, 80, 40 and 20 are not multiples of 7, so every stage pads, and Microsoft's code pads first.
- **Patch merging concatenates in ``(row, col)`` order** ``permute(0, 1, 3, 4, 2, 5)``, not
  Microsoft's ``x0, x1, x2, x3``.
- **Window attention is the explicit path**: ``q * scale``, ``q @ k.T``, add the relative position
  bias and the mask, softmax. timm switches to ``scaled_dot_product_attention`` only when the
  ``TIMM_FUSED_ATTN`` environment variable is 2 or more, which is not the default, and the two are
  different bits.
- **The shift masks are built once, for 640x640**, as timm does when ``strict_img_size`` is set.
  No other input size runs, and none is asked to.
- **Features come out channels-last** (``B, H, W, C``), unnormalised: ``features_only`` drops the
  classifier's final norm.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import SwinSpec

__all__ = ["Swin"]


def window_partition(x: Tensor, window: int) -> Tensor:
    batch, height, width, channels = x.shape
    x = x.view(batch, height // window, window, width // window, window, channels)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window, window, channels)


def window_reverse(windows: Tensor, window: int, height: int, width: int) -> Tensor:
    channels = windows.shape[-1]
    x = windows.view(-1, height // window, width // window, window, window, channels)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, height, width, channels)


def relative_position_index(window: int) -> Tensor:
    coords = torch.stack(torch.meshgrid(torch.arange(window), torch.arange(window), indexing="ij"))
    flat = torch.flatten(coords, 1)
    relative = (flat[:, :, None] - flat[:, None, :]).permute(1, 2, 0).contiguous()
    relative[:, :, 0] += window - 1
    relative[:, :, 1] += window - 1
    relative[:, :, 0] *= 2 * window - 1
    return relative.sum(-1)


def shift_mask(size: int, window: int, shift: int) -> Tensor:
    """timm's ``get_attn_mask``: 0 inside a shifted region, -100 across one, on the padded grid."""
    padded = math.ceil(size / window) * window
    image = torch.zeros((1, padded, padded, 1))
    count = 0
    for rows in ((0, -window), (-window, -shift), (-shift, None)):
        for cols in ((0, -window), (-window, -shift), (-shift, None)):
            image[:, rows[0]:rows[1], cols[0]:cols[1], :] = count
            count += 1
    windows = window_partition(image, window).view(-1, window * window)
    mask = windows.unsqueeze(1) - windows.unsqueeze(2)
    return mask.masked_fill(mask != 0, float(-100.0)).masked_fill(mask == 0, float(0.0))


class WindowAttention(nn.Module):
    def __init__(self, dim: int, heads: int, window: int):
        super().__init__()
        self.heads = heads
        self.window_area = window * window
        self.scale = (dim // heads) ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window - 1) ** 2, heads))
        self.register_buffer("relative_position_index", relative_position_index(window),
                             persistent=False)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        batch, length, channels = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_area, self.window_area, -1)
        attn = attn + bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        if mask is not None:
            windows = mask.shape[0]
            attn = attn.view(-1, windows, self.heads, length, length) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.heads, length, length)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(batch, length, -1)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, size: int, window: int, shift: int, spec: SwinSpec):
        super().__init__()
        self.window, self.shift = window, shift
        self.norm1 = nn.LayerNorm(dim, eps=spec.layer_norm_eps)
        self.attn = WindowAttention(dim, heads, window)
        self.norm2 = nn.LayerNorm(dim, eps=spec.layer_norm_eps)
        self.mlp = Mlp(dim, int(dim * spec.mlp_ratio))
        self.register_buffer("attn_mask", shift_mask(size, window, shift) if shift else None,
                             persistent=False)

    def _attn(self, x: Tensor) -> Tensor:
        batch, height, width, channels = x.shape
        window, shift = self.window, self.shift
        shifted = torch.roll(x, shifts=(-shift, -shift), dims=(1, 2)) if shift else x
        pad_h = (window - height % window) % window
        pad_w = (window - width % window) % window
        shifted = F.pad(shifted, (0, 0, 0, pad_w, 0, pad_h))
        _, padded_h, padded_w, _ = shifted.shape
        windows = window_partition(shifted, window).view(-1, window * window, channels)
        windows = self.attn(windows, self.attn_mask).view(-1, window, window, channels)
        shifted = window_reverse(windows, window, padded_h, padded_w)[:, :height, :width, :]
        shifted = shifted.contiguous()
        return torch.roll(shifted, shifts=(shift, shift), dims=(1, 2)) if shift else shifted

    def forward(self, x: Tensor) -> Tensor:
        batch, height, width, channels = x.shape
        x = x + self._attn(self.norm1(x))
        x = x.reshape(batch, -1, channels)
        x = x + self.mlp(self.norm2(x))
        return x.reshape(batch, height, width, channels)


class PatchMerging(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim, eps=eps)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, height, width, channels = x.shape
        x = F.pad(x, (0, 0, 0, width % 2, 0, height % 2))
        _, height, width, _ = x.shape
        x = x.reshape(batch, height // 2, 2, width // 2, 2, channels).permute(0, 1, 3, 4, 2, 5)
        return self.reduction(self.norm(x.flatten(3)))


class Stage(nn.Module):
    def __init__(self, index: int, size: int, spec: SwinSpec):
        super().__init__()
        dim = spec.widths[index]
        self.downsample = PatchMerging(spec.widths[index - 1], spec.layer_norm_eps) if index else None
        self.blocks = nn.Sequential(*(
            Block(dim, spec.heads[index], size, spec.window,
                  0 if block % 2 == 0 else spec.window // 2, spec)
            for block in range(spec.depths[index])))

    def forward(self, x: Tensor) -> Tensor:
        if self.downsample is not None:
            x = self.downsample(x)
        return self.blocks(x)


class PatchEmbed(nn.Module):
    def __init__(self, spec: SwinSpec):
        super().__init__()
        self.proj = nn.Conv2d(3, spec.embed_dim, spec.patch_size, spec.patch_size)
        self.norm = nn.LayerNorm(spec.embed_dim, eps=spec.layer_norm_eps)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.proj(x).permute(0, 2, 3, 1))


class Swin(nn.Module):
    """``(B, 3, 640, 640)`` normalised pixels -> the ``out_stages`` maps, channels-last.

    Module names are timm's ``FeatureListNet`` names (``patch_embed``, ``layers_0`` ...), so the
    checkpoint's keys load onto them after one prefix rename.
    """

    def __init__(self, spec: SwinSpec):
        super().__init__()
        self.spec = spec
        self.patch_embed = PatchEmbed(spec)
        grid = spec.image_size // spec.patch_size
        for index in range(len(spec.depths)):
            self.add_module(f"layers_{index}", Stage(index, grid // 2**index, spec))

    def forward(self, pixels: Tensor) -> list[Tensor]:
        x = self.patch_embed(pixels)
        features = []
        for index in range(len(self.spec.depths)):
            x = getattr(self, f"layers_{index}")(x)
            if index in self.spec.out_stages:
                features.append(x)
        return features

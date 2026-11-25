# ------------------------------------------------------------------------
# BiRefNet -- Bilateral Reference for High-Resolution Dichotomous Image Segmentation
# Copyright (c) 2024 ZhengPeng. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
#
# The Swin Transformer in this file derives from Microsoft's implementation
# (https://github.com/microsoft/Swin-Transformer, MIT, written by Ze Liu,
# Yutong Lin, Yixuan Wei), by way of BiRefNet's models/backbones/swin_v1.py.
# ------------------------------------------------------------------------
"""Swin v1 backbone, returning the four normed stage outputs.

Extracted from the ``### models/backbones/swin_v1.py`` section of upstream's remote-code
``birefnet.py``. Everything removed is inert at inference, and the gate proves it:

* **``timm`` is gone.** ``DropPath`` is the identity at eval and carries no parameters,
  ``trunc_normal_`` is initialisation the strict load overwrites, ``to_2tuple`` is one line.
* **Every ``nn.Dropout`` is gone**: all are ``p=0`` and the model only runs at eval.
* **No ``torch.utils.checkpoint``, ``_freeze_stages``, ``ape`` or ``pos_drop``**: training
  facilities, or disabled at every call site (``ape=False``).
* **The ``SDPA_enabled`` branch is gone.** Upstream's remote code pins it ``False``; its SDPA branch
  would also drop the relative position bias and the shift mask, so it is not merely a faster
  path to the same numbers.

Kept exactly, because each moves numbers: ``q * scale`` *before* ``q @ k^T``; the shift mask
filled with ``-100.0`` rather than ``-inf``; ``F.pad`` to window multiples (Swin-T at window 7 pads
every stage of a 1024 input; Swin-L at 12 never does); the mask built from
``torch.ceil(torch.tensor(H) / window)`` as upstream builds it.
"""

from __future__ import annotations

__all__ = ["SwinTransformer"]

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BackboneSpec


class Mlp(nn.Module):
    """Two linear layers with GELU between them."""

    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """``(B, H, W, C)`` -> ``(num_windows * B, window_size, window_size, C)``."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """``(num_windows * B, window_size, window_size, C)`` -> ``(B, H, W, C)``."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class WindowAttention(nn.Module):
    """Window multi-head self-attention with a relative position bias."""

    def __init__(self, dim: int, window_size: int, num_heads: int) -> None:
        super().__init__()
        self.window_size = (window_size, window_size)
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        # Persistent, as upstream registers it, so the checkpoint carries it and the strict load
        # holds the computed index against the published one.
        coords = torch.stack(torch.meshgrid(
            [torch.arange(window_size), torch.arange(window_size)], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        area = self.window_size[0] * self.window_size[1]
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(area, area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)


class SwinTransformerBlock(nn.Module):
    """One (shifted) window attention block and its MLP, pre-norm."""

    def __init__(self, dim: int, num_heads: int, window_size: int, shift_size: int) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * 4.0))

    def forward(self, x: torch.Tensor, H: int, W: int, mask_matrix: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Pad to multiples of the window. Zero for Swin-L at every published size; Swin-T pads.
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class PatchMerging(nn.Module):
    """2x2 neighbourhood concatenation, norm, linear: halves the grid, doubles the width."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        if (H % 2 == 1) or (W % 2 == 1):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        return self.reduction(self.norm(x))


class BasicLayer(nn.Module):
    """One stage: blocks alternating plain and shifted windows, then an optional merge."""

    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int,
                 downsample: bool) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim, num_heads, window_size,
                                 0 if (i % 2 == 0) else window_size // 2)
            for i in range(depth)])
        self.downsample = PatchMerging(dim) if downsample else None

    def _mask(self, H: int, W: int, x: torch.Tensor) -> torch.Tensor:
        """The shift mask, built exactly as upstream builds it on every call."""
        # Upstream turns the int into a tensor "for torch.compile in PyTorch 2.5"; the ceil of a
        # float32 quotient is the same integer here, and it is kept as written.
        Hp = torch.ceil(torch.tensor(H) / self.window_size).to(torch.int64) * self.window_size
        Wp = torch.ceil(torch.tensor(W) / self.window_size).to(torch.int64) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        slices = (slice(0, -self.window_size),
                  slice(-self.window_size, -self.shift_size),
                  slice(-self.shift_size, None))
        cnt = 0
        for h in slices:
            for w in slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)).to(x.dtype)

    def forward(self, x: torch.Tensor, H: int, W: int):
        attn_mask = self._mask(H, W, x)
        for blk in self.blocks:
            x = blk(x, H, W, attn_mask)
        if self.downsample is not None:
            return x, H, W, self.downsample(x, H, W), (H + 1) // 2, (W + 1) // 2
        return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    """4x4 strided convolution, then LayerNorm over channels."""

    def __init__(self, embed_dim: int, patch_size: int = 4) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.size()
        if W % self.patch_size != 0:
            x = F.pad(x, (0, self.patch_size - W % self.patch_size))
        if H % self.patch_size != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size - H % self.patch_size))
        x = self.proj(x)
        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)


class SwinTransformer(nn.Module):
    """Swin v1 returning ``(x1, x2, x3, x4)`` at strides 4, 8, 16, 32, each normed, NCHW."""

    def __init__(self, spec: BackboneSpec) -> None:
        super().__init__()
        self.num_layers = len(spec.depths)
        self.num_features = spec.channels
        self.patch_embed = PatchEmbed(spec.embed_dim)
        self.layers = nn.ModuleList([
            BasicLayer(spec.channels[i], spec.depths[i], spec.num_heads[i], spec.window_size,
                       downsample=i < self.num_layers - 1)
            for i in range(self.num_layers)])
        for i in range(self.num_layers):
            self.add_module(f"norm{i}", nn.LayerNorm(spec.channels[i]))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.patch_embed(x)
        Wh, Ww = x.size(2), x.size(3)
        outs = []
        x = x.flatten(2).transpose(1, 2)
        for i in range(self.num_layers):
            x_out, H, W, x, Wh, Ww = self.layers[i](x, Wh, Ww)
            x_out = getattr(self, f"norm{i}")(x_out)
            outs.append(x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous())
        return tuple(outs)

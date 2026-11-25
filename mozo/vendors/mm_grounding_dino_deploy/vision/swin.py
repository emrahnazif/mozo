# SPDX-License-Identifier: Apache-2.0
"""The Swin Transformer backbone, as ``transformers``' ``SwinBackbone`` runs it.

Not Microsoft's Swin and not the copy inside IDEA Research's Grounding DINO: the checkpoints this
package loads were converted for ``transformers``, and its Swin is a re-implementation with its
own arithmetic order -- separate query, key and value projections rather than one fused ``qkv``,
the attention scaled after the product rather than the query before it, padding applied per layer
rather than once per stage. Each is the same model and a different number. Parameter names follow
``transformers``' so the published tensors load by name.

Only the backbone path is carried: patch embedding, the four stages, and a LayerNorm on each
stage handed to the neck. ``SwinBackbone`` always partitions into windows
(``always_partition=True``), so the "shrink the window when the feature map is smaller than it"
branch of ``SwinLayer`` is never taken here and is not carried. Drop-path and dropout are
training-only.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["SwinBackbone"]


def _window_partition(x: Tensor, window: int) -> Tensor:
    batch, height, width, channels = x.shape
    x = x.view(batch, height // window, window, width // window, window, channels)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window, window, channels)


def _window_reverse(windows: Tensor, window: int, height: int, width: int) -> Tensor:
    channels = windows.shape[-1]
    x = windows.view(-1, height // window, width // window, window, window, channels)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, height, width, channels)


class _PatchEmbeddings(nn.Module):
    def __init__(self, patch: int, channels: int) -> None:
        super().__init__()
        self.patch = patch
        self.projection = nn.Conv2d(3, channels, kernel_size=patch, stride=patch)

    def forward(self, pixels: Tensor) -> tuple[Tensor, tuple[int, int]]:
        _, _, height, width = pixels.shape
        # Right and bottom zero padding up to a multiple of the patch -- width first, then height,
        # in two pads, as upstream does.
        if width % self.patch != 0:
            pixels = F.pad(pixels, (0, self.patch - width % self.patch))
        if height % self.patch != 0:
            pixels = F.pad(pixels, (0, 0, 0, self.patch - height % self.patch))
        embedded = self.projection(pixels)
        _, _, height, width = embedded.shape
        return embedded.flatten(2).transpose(1, 2), (height, width)


class _Embeddings(nn.Module):
    def __init__(self, patch: int, channels: int) -> None:
        super().__init__()
        self.patch_embeddings = _PatchEmbeddings(patch, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, pixels: Tensor) -> tuple[Tensor, tuple[int, int]]:
        embedded, dims = self.patch_embeddings(pixels)
        return self.norm(embedded), dims


class _SelfAttention(nn.Module):
    """Windowed self-attention with a learned relative-position bias."""

    def __init__(self, dim: int, heads: int, window: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.window = window
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window - 1) * (2 * window - 1), heads))
        # Persistent, as upstream registers it: the checkpoint carries it, and the strict load
        # holds this geometry against it rather than trusting a recomputation.
        self.register_buffer("relative_position_index", torch.zeros(window**2, window**2,
                                                                    dtype=torch.long))
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, hidden: Tensor, mask: Tensor | None) -> Tensor:
        batch, tokens, _ = hidden.shape
        shape = (batch, tokens, -1, self.head_dim)
        query = self.query(hidden).view(shape).transpose(1, 2)
        key = self.key(hidden).view(shape).transpose(1, 2)
        value = self.value(hidden).view(shape).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(self.window**2, self.window**2, -1).permute(2, 0, 1).contiguous()
        scores = scores + bias.unsqueeze(0)

        if mask is not None:
            windows = mask.shape[0]
            scores = scores.view(batch // windows, windows, self.heads, tokens, tokens)
            scores = scores + mask.unsqueeze(1).unsqueeze(0)
            scores = scores.view(-1, self.heads, tokens, tokens)

        context = torch.matmul(scores.softmax(dim=-1), value)
        context = context.permute(0, 2, 1, 3).contiguous()
        return context.view(context.shape[:-2] + (self.heads * self.head_dim,))


class _Dense(nn.Module):
    """A Linear behind an attribute named ``dense``, which is where the checkpoint keeps it."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.dense = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        return self.dense(x)


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int, window: int) -> None:
        super().__init__()
        self.self = _SelfAttention(dim, heads, window)
        self.output = _Dense(dim, dim)

    def forward(self, hidden: Tensor, mask: Tensor | None) -> Tensor:
        return self.output(self.self(hidden, mask))


class _Block(nn.Module):
    """One Swin layer: (shifted) window attention, then an MLP, each pre-norm with a residual."""

    def __init__(self, dim: int, heads: int, window: int, shift: int) -> None:
        super().__init__()
        self.window = window
        self.shift = shift
        self.layernorm_before = nn.LayerNorm(dim)
        self.attention = _Attention(dim, heads, window)
        self.layernorm_after = nn.LayerNorm(dim)
        self.intermediate = _Dense(dim, 4 * dim)
        self.output = _Dense(4 * dim, dim)

    def _mask(self, height: int, width: int, dtype: torch.dtype, device) -> Tensor | None:
        """The shifted-window mask: -100 between tokens that the roll made neighbours."""
        if self.shift == 0:
            return None
        image = torch.zeros((1, height, width, 1), dtype=dtype, device=device)
        slices = (slice(0, -self.window), slice(-self.window, -self.shift),
                  slice(-self.shift, None))
        count = 0
        for rows in slices:
            for cols in slices:
                image[:, rows, cols, :] = count
                count += 1
        windows = _window_partition(image, self.window).view(-1, self.window * self.window)
        mask = windows.unsqueeze(1) - windows.unsqueeze(2)
        return mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)

    def forward(self, hidden: Tensor, dims: tuple[int, int]) -> Tensor:
        height, width = dims
        batch, _, channels = hidden.shape
        shortcut = hidden

        hidden = self.layernorm_before(hidden).view(batch, height, width, channels)
        # Padded after the norm, to a multiple of the window, right and bottom.
        pad_right = (self.window - width % self.window) % self.window
        pad_bottom = (self.window - height % self.window) % self.window
        hidden = F.pad(hidden, (0, 0, 0, pad_right, 0, pad_bottom))
        _, height_pad, width_pad, _ = hidden.shape

        if self.shift > 0:
            hidden = torch.roll(hidden, shifts=(-self.shift, -self.shift), dims=(1, 2))
        windows = _window_partition(hidden, self.window).view(-1, self.window**2, channels)
        mask = self._mask(height_pad, width_pad, hidden.dtype, windows.device)
        attended = self.attention(windows, mask).view(-1, self.window, self.window, channels)
        hidden = _window_reverse(attended, self.window, height_pad, width_pad)
        if self.shift > 0:
            hidden = torch.roll(hidden, shifts=(self.shift, self.shift), dims=(1, 2))
        if pad_right > 0 or pad_bottom > 0:
            hidden = hidden[:, :height, :width, :].contiguous()

        hidden = shortcut + hidden.view(batch, height * width, channels)
        # Exact erf GELU, which is what the config's ``"gelu"`` names.
        return hidden + self.output(F.gelu(self.intermediate(self.layernorm_after(hidden))))


class _PatchMerging(nn.Module):
    """Halve the resolution: concatenate each 2x2 neighbourhood, normalise, project."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, hidden: Tensor, dims: tuple[int, int]) -> Tensor:
        height, width = dims
        batch, _, channels = hidden.shape
        hidden = hidden.view(batch, height, width, channels)
        if height % 2 == 1 or width % 2 == 1:
            hidden = F.pad(hidden, (0, 0, 0, width % 2, 0, height % 2))
        hidden = torch.cat([hidden[:, 0::2, 0::2, :], hidden[:, 1::2, 0::2, :],
                            hidden[:, 0::2, 1::2, :], hidden[:, 1::2, 1::2, :]], -1)
        return self.reduction(self.norm(hidden.view(batch, -1, 4 * channels)))


class _Stage(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, window: int, downsample: bool) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            _Block(dim, heads, window, shift=0 if i % 2 == 0 else window // 2)
            for i in range(depth))
        self.downsample = _PatchMerging(dim) if downsample else None


class _Encoder(nn.Module):
    def __init__(self, embed_dim: int, depths: tuple[int, ...], heads: tuple[int, ...],
                 window: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            _Stage(embed_dim * 2**i, depth, heads[i], window, downsample=i < len(depths) - 1)
            for i, depth in enumerate(depths))


class SwinBackbone(nn.Module):
    """Swin, returning the normalised output of the requested stages.

    Args:
        embed_dim: Width of the first stage; each later stage doubles it.
        depths: Blocks per stage.
        num_heads: Attention heads per stage.
        window_size: Side of the attention window, in tokens.
        out_stages: Which stages (0-based) to return. Each gets its own LayerNorm, named
            ``stage{n+1}`` as upstream names it.

    Returns from ``forward``:
        One ``(batch, channels, h, w)`` map per requested stage, in order. Each is the stage's
        output *before* its patch merging.
    """

    def __init__(self, embed_dim: int, depths: tuple[int, ...], num_heads: tuple[int, ...],
                 window_size: int, out_stages: tuple[int, ...]) -> None:
        super().__init__()
        self.out_stages = tuple(out_stages)
        self.embeddings = _Embeddings(patch=4, channels=embed_dim)
        self.encoder = _Encoder(embed_dim, depths, num_heads, window_size)
        self.hidden_states_norms = nn.ModuleDict(
            {f"stage{i + 1}": nn.LayerNorm(embed_dim * 2**i) for i in self.out_stages})

    def forward(self, pixels: Tensor) -> list[Tensor]:
        hidden, dims = self.embeddings(pixels)
        outputs = []
        for index, stage in enumerate(self.encoder.layers):
            height, width = dims
            for block in stage.blocks:
                hidden = block(hidden, dims)
            if index in self.out_stages:
                batch, _, channels = hidden.shape
                normed = self.hidden_states_norms[f"stage{index + 1}"](hidden)
                outputs.append(normed.view(batch, height, width, channels)
                               .permute(0, 3, 1, 2).contiguous())
            if stage.downsample is not None:
                hidden = stage.downsample(hidden, dims)
                dims = ((height + 1) // 2, (width + 1) // 2)
        return outputs

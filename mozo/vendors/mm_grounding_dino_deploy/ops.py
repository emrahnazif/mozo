# SPDX-License-Identifier: Apache-2.0
"""Multi-scale deformable attention, the ``grid_sample`` way, and plain multi-head attention.

Deformable attention samples a handful of learned offsets around a reference point on every level
of the image pyramid, instead of attending to every location. ``transformers`` can swap in a
compiled kernel from its kernel hub; the pure-torch path is the one carried, because it runs on
every device mozo targets and it is the path the parity gate compares.

**Level sizes travel as Python integers** (``level_shapes``) beside the ``spatial_shapes`` tensor.
``torch.export`` cannot follow a value read back off a tensor, and the read would be a device sync
per call. Upstream carries the same pair (``spatial_shapes_list``) for the same reason.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["MultiScaleDeformableAttention", "MultiheadAttention"]


def _sample(value: Tensor, level_shapes: list[tuple[int, int]], locations: Tensor,
            weights: Tensor) -> Tensor:
    """Bilinearly sample *value* at *locations* on each level and sum with *weights*."""
    batch, _, heads, head_dim = value.shape
    _, queries, _, levels, points, _ = locations.shape
    per_level = value.split([height * width for height, width in level_shapes], dim=1)
    grids = 2 * locations - 1
    sampled = []
    for level, (height, width) in enumerate(level_shapes):
        level_value = (per_level[level].flatten(2).transpose(1, 2)
                       .reshape(batch * heads, head_dim, height, width))
        level_grid = grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampled.append(F.grid_sample(level_value, level_grid, mode="bilinear",
                                     padding_mode="zeros", align_corners=False))
    weights = weights.transpose(1, 2).reshape(batch * heads, 1, queries, levels * points)
    output = ((torch.stack(sampled, dim=-2).flatten(-2) * weights).sum(-1)
              .view(batch, heads * head_dim, queries))
    return output.transpose(1, 2).contiguous()


class MultiScaleDeformableAttention(nn.Module):
    """Deformable attention over a feature pyramid.

    Args:
        d_model: Model width.
        n_heads: Attention heads.
        n_levels: Pyramid levels.
        n_points: Sampling points per head per level.
    """

    def __init__(self, d_model: int, n_heads: int, n_levels: int, n_points: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        hidden: Tensor,
        position: Tensor,
        memory: Tensor,
        valid: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_shapes: list[tuple[int, int]],
    ) -> Tensor:
        """Attend from *hidden* (+ *position*) into *memory*.

        Args:
            hidden: ``(batch, queries, d_model)``.
            position: Added to *hidden* before the offsets and weights are predicted.
            memory: ``(batch, tokens, d_model)``, the flattened pyramid.
            valid: ``(batch, tokens)``, True where *memory* is picture rather than padding.
            reference_points: ``(batch, queries, levels, 2)`` points or ``(..., 4)`` boxes.
            spatial_shapes: ``(levels, 2)`` long tensor of ``(height, width)``.
            level_shapes: The same sizes as Python integers.
        """
        hidden = hidden + position
        batch, queries, _ = hidden.shape
        _, tokens, _ = memory.shape

        value = self.value_proj(memory).masked_fill(~valid[..., None], float(0))
        value = value.view(batch, tokens, self.n_heads, self.d_model // self.n_heads)
        offsets = self.sampling_offsets(hidden).view(
            batch, queries, self.n_heads, self.n_levels, self.n_points, 2)
        weights = self.attention_weights(hidden).view(
            batch, queries, self.n_heads, self.n_levels * self.n_points)
        weights = F.softmax(weights, -1).view(
            batch, queries, self.n_heads, self.n_levels, self.n_points)

        if reference_points.shape[-1] == 2:
            normaliser = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            locations = (reference_points[:, :, None, :, None, :]
                         + offsets / normaliser[None, None, None, :, None, :])
        else:
            locations = (reference_points[:, :, None, :, None, :2]
                         + offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5)

        return self.output_proj(_sample(value, level_shapes, locations, weights))


class MultiheadAttention(nn.Module):
    """``transformers``' re-implementation of ``nn.MultiheadAttention`` (``batch_first``).

    Not ``nn.MultiheadAttention`` itself, which IDEA Research's Grounding DINO uses: that one
    scales the query before the product, this one divides the product after, and the two are
    different numbers. Separate ``query``/``key``/``value`` projections, named as the checkpoint
    names them.

    Args:
        dim: Model width.
        heads: Attention heads.
    """

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor,
                mask: Tensor | None = None) -> Tensor:
        """Attend; *mask*, if given, is added to the scores before the softmax."""
        batch, length, _ = queries.shape
        query = self.query(queries).view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        key = self.key(keys).view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        value = self.value(values).view(batch, -1, self.heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        context = torch.matmul(F.softmax(scores, dim=-1), value)
        context = context.permute(0, 2, 1, 3).contiguous()
        return self.out_proj(context.view(context.size()[:-2] + (self.heads * self.head_dim,)))

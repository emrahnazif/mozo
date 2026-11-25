# SPDX-License-Identifier: Apache-2.0
"""The vision tower: patches in, image tokens (and three DeepStack feature sets) out.

Extracted from ``transformers/models/qwen3_vl/modeling_qwen3_vl.py`` (v5.24.1:
``Qwen3VLVisionModel`` and its parts) and ``transformers/vision_utils.py`` (the position and
interpolation helpers it calls). Module and parameter names are upstream's, so the published
``model.visual.*`` keys load strictly.

What is particular to it, and easy to get almost right:

- **The learned 48x48 position table is resampled to each image's grid** by bilinear interpolation
  with ``align_corners=True``, in merge-window order. Upstream 5.x computes the taps and weights
  itself rather than calling ``F.interpolate``; 4.57 did it differently, and the two disagree in
  the last bits, so this follows 5.24.1 line for line.
- **The rotary embedding is axial 2-D**: ``head_dim // 4`` frequencies, applied to (row, column)
  and laid out ``[row, column, row, column]`` across the head.
- **Three blocks (5, 11, 17) feed the text decoder** through their own mergers, which normalise
  *after* gathering a 2x2 window (``use_postshuffle_norm``) where the main merger normalises before.
- **The MLP's GELU is the tanh approximation; the mergers' is exact.** Upstream's
  ``gelu_pytorch_tanh`` and ``nn.GELU()``.
- **Attention is per image**, split by cumulative lengths; with one image the split has one piece,
  but the call shape is upstream's.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import VisionSpec

__all__ = ["VisionTower"]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """``apply_rotary_pos_emb_vision``: computed in float32 whatever the tower runs in."""
    q_dtype, k_dtype = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q_dtype), k_embed.to(k_dtype)


class PatchEmbed(nn.Module):
    def __init__(self, spec: VisionSpec) -> None:
        super().__init__()
        self.spec = spec
        kernel = [spec.temporal_patch, spec.patch, spec.patch]
        self.proj = nn.Conv3d(3, spec.width, kernel_size=kernel, stride=kernel, bias=True)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        s = self.spec
        pixels = pixels.view(-1, 3, s.temporal_patch, s.patch, s.patch)
        return self.proj(pixels.to(dtype=self.proj.weight.dtype)).view(-1, s.width)


class Attention(nn.Module):
    def __init__(self, spec: VisionSpec) -> None:
        super().__init__()
        self.heads = spec.heads
        self.qkv = nn.Linear(spec.width, spec.width * 3, bias=True)
        self.proj = nn.Linear(spec.width, spec.width)
        self.scaling = spec.head_dim ** -0.5

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor,
                rotary: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        length = x.shape[0]
        q, k, v = self.qkv(x).reshape(length, 3, self.heads, -1).permute(1, 0, 2, 3).unbind(0)
        q, k = _apply_rotary(q, k, *rotary)
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        pieces = [torch.split(t, lengths, dim=2) for t in (q, k, v)]
        out = [
            F.scaled_dot_product_attention(qi, ki, vi, attn_mask=None, dropout_p=0.0,
                                           scale=self.scaling, is_causal=False)
            .transpose(1, 2).contiguous()
            for qi, ki, vi in zip(*pieces)
        ]
        out = torch.cat(out, dim=1)
        return self.proj(out.reshape(length, -1).contiguous())


class MLP(nn.Module):
    def __init__(self, spec: VisionSpec) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(spec.width, spec.mlp, bias=True)
        self.linear_fc2 = nn.Linear(spec.mlp, spec.width, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(F.gelu(self.linear_fc1(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, spec: VisionSpec) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(spec.width, eps=spec.norm_eps)
        self.norm2 = nn.LayerNorm(spec.width, eps=spec.norm_eps)
        self.attn = Attention(spec)
        self.mlp = MLP(spec)

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor,
                rotary: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cu_seqlens, rotary)
        return x + self.mlp(self.norm2(x))


class PatchMerger(nn.Module):
    """A 2x2 window of patch features to one text-width token."""

    def __init__(self, spec: VisionSpec, postshuffle: bool) -> None:
        super().__init__()
        self.hidden = spec.width * spec.merge ** 2
        self.postshuffle = postshuffle
        self.norm = nn.LayerNorm(self.hidden if postshuffle else spec.width, eps=spec.norm_eps)
        self.linear_fc1 = nn.Linear(self.hidden, self.hidden)
        self.linear_fc2 = nn.Linear(self.hidden, spec.out_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.view(-1, self.hidden) if self.postshuffle else x).view(-1, self.hidden)
        return self.linear_fc2(F.gelu(self.linear_fc1(x)))


def _axis_taps(index: torch.Tensor, size: torch.Tensor, side: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinear taps and weights along one axis, ``align_corners=True``, clamped at the border."""
    index = index.to(torch.float32)
    src = index * (side - 1) / torch.clamp(size - 1, min=1)
    floor = torch.floor(src)
    offsets = torch.arange(0, 2, device=index.device)
    taps = (floor.long()[:, None] + offsets).clamp(0, side - 1)
    distance = (src[:, None] - floor[:, None] - offsets).abs()
    return taps, (1 - distance).clamp(min=0)


def interpolation(grid: tuple[int, int, int], side: int, merge: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``get_vision_interpolation_indices_and_weights`` for one image, bilinear, align_corners."""
    grid_thw = torch.tensor([grid], dtype=torch.long)
    counts = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
    heights = torch.repeat_interleave(grid_thw[:, 1], counts)
    widths = torch.repeat_interleave(grid_thw[:, 2], counts)
    starts = torch.repeat_interleave(F.pad(counts.cumsum(0)[:-1], (1, 0)), counts)
    within = (torch.arange(counts.sum()) - starts) % (heights * widths)
    blocks_w = widths // merge
    in_col = within % merge
    in_row = (within // merge) % merge
    block_col = (within // (merge * merge)) % blocks_w
    block_row = within // (merge * merge * blocks_w)
    row = block_row * merge + in_row
    col = block_col * merge + in_col
    h_taps, h_weights = _axis_taps(row, heights, side)
    w_taps, w_weights = _axis_taps(col, widths, side)
    n = h_taps.shape[1]
    indices = (h_taps[:, :, None] * side + w_taps[:, None, :]).reshape(-1, n * n)
    weights = (h_weights[:, :, None] * w_weights[:, None, :]).reshape(-1, n * n)
    return indices, weights


def rotary_positions(grid: tuple[int, int, int], merge: int) -> torch.Tensor:
    """``get_vision_position_ids``: (row, column) of every patch, merge-window order. ``(N, 2)``."""
    t, h, w = grid
    hpos, wpos = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    shape = (h // merge, merge, w // merge, merge)
    hpos = hpos.reshape(shape).transpose(1, 2).flatten()
    wpos = wpos.reshape(shape).transpose(1, 2).flatten()
    return torch.stack([hpos, wpos], dim=-1).repeat(t, 1)


class VisionTower(nn.Module):
    """``Qwen3VLVisionModel``: ``(pixels, grid)`` to ``(tokens, [deepstack x3])``."""

    def __init__(self, spec: VisionSpec) -> None:
        super().__init__()
        self.spec = spec
        self.patch_embed = PatchEmbed(spec)
        self.pos_embed = nn.Embedding(spec.grid * spec.grid, spec.width)
        self.register_buffer("inv_freq", self.rotary_frequencies(), persistent=False)
        self.blocks = nn.ModuleList(Block(spec) for _ in range(spec.depth))
        self.merger = PatchMerger(spec, postshuffle=False)
        self.deepstack_merger_list = nn.ModuleList(
            PatchMerger(spec, postshuffle=True) for _ in spec.deepstack)

    def rotary_frequencies(self) -> torch.Tensor:
        """Upstream's ``inv_freq``, computed on the CPU in float32 as upstream computes it."""
        spatial = self.spec.head_dim // 2
        return 1.0 / (self.spec.rope_theta ** (torch.arange(0, spatial, 2, dtype=torch.float) / spatial))

    def _rotary(self, x: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = positions[..., None].float() * self.inv_freq.to(device=x.device, dtype=torch.float)
        cos, sin = freqs.cos() * 1.0, freqs.sin() * 1.0

        def recompose(f: torch.Tensor) -> torch.Tensor:
            hw = torch.cat([f[:, 0], f[:, 1]], dim=-1)
            return torch.cat([hw, hw], dim=-1)

        return recompose(cos), recompose(sin)

    def forward(self, pixels: torch.Tensor, grid: tuple[int, int, int],
                record: dict | None = None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the tower. *record*, when given, collects every stage for the gate."""
        s = self.spec
        device = pixels.device
        indices, weights = interpolation(grid, s.grid, s.merge)
        indices, weights = indices.to(device), weights.to(device)
        positions = rotary_positions(grid, s.merge).to(device)
        cu_seqlens = F.pad(torch.tensor([grid[1] * grid[2]], dtype=torch.int32)
                           .repeat_interleave(grid[0]).cumsum(0, dtype=torch.int32), (1, 0), value=0)

        x = self.patch_embed(pixels)
        pos = (self.pos_embed(indices) * weights[:, :, None]).sum(1)
        x = x + pos.to(x.dtype)
        rotary = self._rotary(x, positions)
        if record is not None:
            record["patch_embed"] = x
        deepstack = []
        for number, block in enumerate(self.blocks):
            x = block(x, cu_seqlens, rotary)
            if record is not None:
                record[f"vision_block_{number}"] = x
            if number in s.deepstack:
                deepstack.append(self.deepstack_merger_list[s.deepstack.index(number)](x))
        merged = self.merger(x)
        if record is not None:
            record["merger"] = merged
            for i, feature in enumerate(deepstack):
                record[f"deepstack_{i}"] = feature
        return merged, deepstack

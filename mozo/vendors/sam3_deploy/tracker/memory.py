# SPDX-License-Identifier: Apache-2.0
"""What carries an object from one frame to the next: memory attention and the memory encoder.

Derived from ``transformers/models/sam3_tracker_video`` (Apache-2.0), SAM 3's own video tracker
-- the same provenance as the click head. Two places follow the published model's arithmetic
rather than that source, each found by comparing numbers (``PROVENANCE.md``):

- **The rotary encoding is complex**, as in the image trunk (:func:`..vision.vit.apply_rotary`):
  ``transformers`` rotates with real ``cos``/``sin`` and notes the 1e-07 drift itself.
- **Keys that skip the rotation are rotated in place.** Object pointers ride at the end of the
  memory and are excluded from the rotary encoding; the rotated part is written back into the key
  tensor rather than concatenated with the rest, which leaves the key in the layout the attention
  kernel was measured with.

**The memory encoder** turns a frame's mask and image features into a 64-channel memory: the mask
is resized to 1152x1152 (antialiased), downsampled by four stride-2 convolutions to the 72x72 grid,
added to a projection of the image features, fused by two ConvNeXt blocks and projected down.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..click.layers import LayerNorm2d
from ..config import TRACKER, TrackerSpec
from ..position import SinePositionEmbedding

__all__ = ["MemoryAttention", "MemoryEncoder", "axial_rotary"]


def axial_rotary(dim: int, side: int, theta: float, device=None) -> Tensor:
    """The 2-D rotary table for a ``side x side`` grid: x frequencies, then y.

    Returns:
        ``(side * side, dim // 2)`` complex64, row-major over the grid.
    """
    bands = 1.0 / (theta ** (torch.arange(0, dim, 4, device=device)[: dim // 4].float() / dim))
    index = torch.arange(side * side, dtype=torch.float32, device=device)
    x = (index % side).float()
    y = torch.div(index, side, rounding_mode="floor").float()
    angles_x = torch.outer(x, bands)
    angles_y = torch.outer(y, bands)
    return torch.cat([torch.polar(torch.ones_like(angles_x), angles_x),
                      torch.polar(torch.ones_like(angles_y), angles_y)], dim=-1)


def _rotate(x: Tensor, table: Tensor) -> Tensor:
    """Multiply each channel pair of ``(B, heads, N, C)`` by ``table`` ``(N, C // 2)``."""
    paired = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(paired * table.view(1, 1, *table.shape)).flatten(3).type_as(x)


class RotaryAttention(nn.Module):
    """Single-head attention with the 2-D rotary encoding on queries and keys.

    Args:
        hidden: Query width (and output width).
        kv_dim: Key/value width -- the memory's 64 for cross-attention.
        repeat_keys: Keys span several frames' grids; repeat the table once per grid.
    """

    def __init__(self, hidden: int, kv_dim: int | None = None, repeat_keys: bool = False):
        super().__init__()
        kv_dim = kv_dim or hidden
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(kv_dim, hidden)
        self.v_proj = nn.Linear(kv_dim, hidden)
        self.o_proj = nn.Linear(hidden, hidden)
        self.repeat_keys = repeat_keys

    def forward(self, query: Tensor, key: Tensor, value: Tensor, table: Tensor,
                key_table: Tensor | None = None, unrotated: int = 0) -> Tensor:
        # One head: ``(B, N, C)`` -> ``(B, 1, N, C)``, as a reshape-and-transpose view.
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        b = q.shape[0]
        q = q.reshape(b, q.shape[1], 1, -1).transpose(1, 2)
        k = k.reshape(b, k.shape[1], 1, -1).transpose(1, 2)
        v = v.reshape(b, v.shape[1], 1, -1).transpose(1, 2)

        rotated = k.shape[-2] - unrotated
        q = _rotate(q, table)
        if rotated:
            k[:, :, :rotated] = _rotate(k[:, :, :rotated], table if key_table is None else key_table)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, -1, out.shape[-1])
        return self.o_proj(out)


class MemoryAttentionLayer(nn.Module):
    """Pre-norm self-attention over the frame, cross-attention to the memory, then a feed-forward."""

    def __init__(self, spec: TrackerSpec = TRACKER):
        super().__init__()
        self.self_attn = RotaryAttention(spec.hidden)
        self.cross_attn_image = RotaryAttention(spec.hidden, spec.memory_dim, repeat_keys=True)
        self.linear1 = nn.Linear(spec.hidden, spec.attention_intermediate)
        self.linear2 = nn.Linear(spec.attention_intermediate, spec.hidden)
        self.layer_norm1 = nn.LayerNorm(spec.hidden)
        self.layer_norm2 = nn.LayerNorm(spec.hidden)
        self.layer_norm3 = nn.LayerNorm(spec.hidden)

    def forward(self, x: Tensor, memory: Tensor, keys: Tensor, table: Tensor, key_table: Tensor,
                unrotated: int) -> Tensor:
        h = self.layer_norm1(x)
        x = x + self.self_attn(h, h, h, table)
        h = self.layer_norm2(x)
        x = x + self.cross_attn_image(h, keys, memory, table, key_table, unrotated)
        h = self.layer_norm3(x)
        return x + self.linear2(F.relu(self.linear1(h)))


class MemoryAttention(nn.Module):
    """Condition the current frame's features on an object's memories and pointers.

    Args:
        spec: Tracker geometry.
    """

    def __init__(self, spec: TrackerSpec = TRACKER):
        super().__init__()
        self.layers = nn.ModuleList(MemoryAttentionLayer(spec) for _ in range(spec.attention_layers))
        self.layer_norm = nn.LayerNorm(spec.hidden)
        self.dim = spec.hidden
        self.theta = spec.rope_theta
        self._tables: dict[tuple, Tensor] = {}

    def table(self, tokens: int, device) -> Tensor:
        """The rotary table for a square grid of ``tokens`` positions, built once per size."""
        key = (tokens, torch.device(device))
        if key not in self._tables:
            self._tables[key] = axial_rotary(self.dim, int(tokens ** 0.5), self.theta, device)
        return self._tables[key]

    def forward(self, features: Tensor, positions: Tensor, memory: Tensor, memory_pos: Tensor,
                unrotated: int = 0) -> Tensor:
        """Fuse.

        Args:
            features: ``(HW, B, C)`` the current frame's coarsest features, sequence first.
            positions: ``(HW, B, C)`` their position encoding.
            memory: ``(M, B, 64)`` memories, object-pointer tokens last.
            memory_pos: ``(M, B, 64)`` their encodings.
            unrotated: How many trailing memory tokens are object pointers (no rotation).

        Returns:
            ``(HW, B, C)``.
        """
        x = (features + 0.1 * positions).transpose(0, 1)
        memory = memory.transpose(0, 1)
        memory_pos = memory_pos.transpose(0, 1)
        table = self.table(x.shape[1], x.device)
        # The same for every layer, so built once: the keys (memory plus its encoding) and the
        # rotary table tiled once per memory frame the keys span.
        keys = memory + memory_pos
        key_table = table.repeat((memory.shape[1] - unrotated) // x.shape[1], 1)
        for layer in self.layers:
            x = layer(x, memory, keys, table, key_table, unrotated)
        return self.layer_norm(x).transpose(0, 1)


class DownSamplerLayer(nn.Module):
    """A stride-2 convolution, a channel LayerNorm and a GELU."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=3, stride=2, padding=1)
        self.layer_norm = LayerNorm2d(cout)

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(self.layer_norm(self.conv(x)))


class MaskDownSampler(nn.Module):
    """1152x1152 mask -> 72x72x256: four stride-2 stages, each quadrupling the channels."""

    def __init__(self, spec: TrackerSpec = TRACKER):
        super().__init__()
        self.size = spec.mask_input_size
        channels = [4 ** i for i in range(spec.downsampler_layers + 1)]
        self.layers = nn.ModuleList(DownSamplerLayer(channels[i], channels[i + 1])
                                    for i in range(spec.downsampler_layers))
        self.final_conv = nn.Conv2d(channels[-1], spec.hidden, kernel_size=1)

    def forward(self, masks: Tensor) -> Tensor:
        if list(masks.shape[-2:]) != [self.size, self.size]:
            masks = F.interpolate(masks.float(), size=(self.size, self.size), mode="bilinear",
                                  align_corners=False, antialias=True)
        for layer in self.layers:
            masks = layer(masks)
        return self.final_conv(masks)


class FuserBlock(nn.Module):
    """A ConvNeXt block: depthwise conv, LayerNorm, pointwise MLP (as linears), scaled residual."""

    def __init__(self, spec: TrackerSpec = TRACKER):
        super().__init__()
        dim = spec.hidden
        self.depthwise_conv = nn.Conv2d(dim, dim, kernel_size=spec.fuser_kernel,
                                        padding=spec.fuser_kernel // 2, groups=dim)
        self.layer_norm = LayerNorm2d(dim)
        self.pointwise_conv1 = nn.Linear(dim, 4 * dim)
        self.pointwise_conv2 = nn.Linear(4 * dim, dim)
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        h = self.layer_norm(self.depthwise_conv(x)).permute(0, 2, 3, 1)
        h = self.pointwise_conv2(F.gelu(self.pointwise_conv1(h)))
        return x + (self.scale * h).permute(0, 3, 1, 2)


class MemoryEncoder(nn.Module):
    """Encode a frame's mask with its image features into a memory and its position encoding."""

    def __init__(self, spec: TrackerSpec = TRACKER):
        super().__init__()
        self.mask_downsampler = MaskDownSampler(spec)
        self.feature_projection = nn.Conv2d(spec.hidden, spec.hidden, kernel_size=1)
        self.memory_fuser = nn.ModuleList(FuserBlock(spec) for _ in range(spec.fuser_layers))
        self.projection = nn.Conv2d(spec.hidden, spec.memory_dim, kernel_size=1)
        self.position = SinePositionEmbedding(spec.memory_dim // 2)

    def forward(self, features: Tensor, masks: Tensor) -> tuple[Tensor, Tensor]:
        """Encode.

        Args:
            features: ``(B, 256, 72, 72)`` the frame's coarsest features.
            masks: ``(B, 1, H, W)`` mask values already scaled for the encoder (not logits).

        Returns:
            ``(B, 64, 72, 72)`` memory and its ``(B, 64, 72, 72)`` position encoding.
        """
        masks = self.mask_downsampler(masks)
        x = self.feature_projection(features) + masks
        for block in self.memory_fuser:
            x = block(x)
        x = self.projection(x)
        pos = self.position(x.shape[-2], x.shape[-1], x.device, torch.float32)
        return x, pos.repeat(x.shape[0], 1, 1, 1).to(x.dtype)

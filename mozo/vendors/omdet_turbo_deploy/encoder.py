# SPDX-License-Identifier: Apache-2.0
"""The hybrid encoder: RT-DETR's, on Swin's three levels.

Extracted from ``transformers``' ``OmDetTurboHybridEncoder`` (Apache-2.0), which is RT-DETR's
encoder: project each Swin level to 256 channels, run one transformer layer over the coarsest
(stride 32) with a 2-D sine-cosine position embedding, then a top-down FPN and a bottom-up PAN of
CSP blocks. Text plays no part here; it enters in the decoder.

Upstream details kept as they are:

- **The position embedding meshes ``(width, height)`` with ``indexing="ij"``**, so its first axis
  is x. The grid is square at 640 and the order still decides which half of the channels is which.
- **The attention is spelled out** (``q @ k.T / sqrt(d)``, softmax, ``@ v``), not
  ``scaled_dot_product_attention``: upstream's ``OmDetTurboMultiheadAttention`` divides the scores,
  where SDPA scales the query.
- **The RepVGG blocks stay as two convolutions summed**, the training form; upstream never fuses
  them, and a fused 3x3 is the same algebra in different bits.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import Spec

__all__ = ["HybridEncoder", "MultiheadAttention", "ConvNorm"]


class ConvNorm(nn.Module):
    """Convolution without bias, batch norm, optional activation (RT-DETR's ``ConvNormLayer``)."""

    def __init__(self, spec: Spec, cin: int, cout: int, kernel: int, stride: int,
                 padding: int | None = None, activation: type[nn.Module] | None = None):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel, stride,
                              padding=(kernel - 1) // 2 if padding is None else padding, bias=False)
        self.norm = nn.BatchNorm2d(cout, spec.batch_norm_eps)
        self.activation = activation() if activation is not None else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):
    def __init__(self, spec: Spec):
        super().__init__()
        width = spec.d_model
        self.conv1 = ConvNorm(spec, width, width, 3, 1, padding=1)
        self.conv2 = ConvNorm(spec, width, width, 1, 1, padding=0)
        self.activation = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.conv1(x) + self.conv2(x))


class CSPRepLayer(nn.Module):
    def __init__(self, spec: Spec):
        super().__init__()
        width = spec.d_model
        self.conv1 = ConvNorm(spec, 2 * width, width, 1, 1, activation=nn.SiLU)
        self.conv2 = ConvNorm(spec, 2 * width, width, 1, 1, activation=nn.SiLU)
        self.bottlenecks = nn.Sequential(*(RepVggBlock(spec) for _ in range(spec.csp_blocks)))

    def forward(self, x: Tensor) -> Tensor:
        # ``conv3`` is the identity: hidden_expansion is 1.
        return self.bottlenecks(self.conv1(x)) + self.conv2(x)


class MultiheadAttention(nn.Module):
    """``OmDetTurboMultiheadAttention``: batch-first, separate projections, divide-then-add-mask."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor,
                mask: Tensor | None = None) -> Tensor:
        batch = queries.shape[0]
        query = self.query(queries).view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        key = self.key(keys).view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        value = self.value(values).view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        context = torch.matmul(scores.softmax(dim=-1), value)
        context = context.permute(0, 2, 1, 3).contiguous()
        return self.out_proj(context.view(context.shape[:-2] + (-1,)))


class EncoderLayer(nn.Module):
    def __init__(self, spec: Spec):
        super().__init__()
        width = spec.d_model
        self.self_attn = MultiheadAttention(width, spec.encoder_heads)
        self.self_attn_layer_norm = nn.LayerNorm(width, eps=spec.layer_norm_eps)
        self.fc1 = nn.Linear(width, spec.encoder_feedforward)
        self.fc2 = nn.Linear(spec.encoder_feedforward, width)
        self.final_layer_norm = nn.LayerNorm(width, eps=spec.layer_norm_eps)

    def forward(self, x: Tensor, position: Tensor) -> Tensor:
        query = key = x + position
        x = self.self_attn_layer_norm(x + self.self_attn(query, key, x))
        return self.final_layer_norm(x + self.fc2(F.relu(self.fc1(x))))


class Encoder(nn.Module):
    """One stack of transformer layers. Named to match the checkpoint (``encoder.encoder.0``)."""

    def __init__(self, spec: Spec):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(spec)])


def sincos_position(width: int, height: int, dim: int, temperature: float,
                    device: torch.device, dtype: torch.dtype) -> Tensor:
    """``build_2d_sincos_position_embedding``, upstream's argument order and meshing."""
    grid_w = torch.arange(int(width), dtype=dtype, device=device)
    grid_h = torch.arange(int(height), dtype=dtype, device=device)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
    quarter = dim // 4
    omega = torch.arange(quarter, dtype=dtype, device=device) / quarter
    omega = 1.0 / (temperature**omega)
    out_w = grid_w.flatten()[..., None] @ omega[None]
    out_h = grid_h.flatten()[..., None] @ omega[None]
    return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]


class HybridEncoder(nn.Module):
    """Swin's three channels-first levels -> three 256-channel levels (strides 8, 16, 32)."""

    def __init__(self, spec: Spec):
        super().__init__()
        self.spec = spec
        width = spec.d_model
        self.channel_projection_layers = nn.ModuleList(
            nn.Sequential(nn.Conv2d(cin, width, kernel_size=(1, 1), bias=False),
                          nn.BatchNorm2d(width))
            for cin in spec.encoder_in_channels)
        self.encoder = nn.ModuleList([Encoder(spec)])
        levels = len(spec.encoder_in_channels)
        self.lateral_convs = nn.ModuleList(
            ConvNorm(spec, width, width, 1, 1, activation=nn.GELU) for _ in range(levels - 1))
        self.fpn_blocks = nn.ModuleList(CSPRepLayer(spec) for _ in range(levels - 1))
        self.downsample_convs = nn.ModuleList(
            ConvNorm(spec, width, width, 3, 2, activation=nn.GELU) for _ in range(levels - 1))
        self.pan_blocks = nn.ModuleList(CSPRepLayer(spec) for _ in range(levels - 1))

    def forward(self, features: list[Tensor]) -> list[Tensor]:
        spec = self.spec
        projected = [project(feature)
                     for project, feature in zip(self.channel_projection_layers, features)]

        index = spec.encoder_projection_index
        height, width = projected[index].shape[2:]
        flat = projected[index].flatten(2).permute(0, 2, 1)
        position = sincos_position(width, height, spec.d_model, spec.positional_temperature,
                                   flat.device, flat.dtype).to(flat.device, flat.dtype)
        for layer in self.encoder[0].layers:
            flat = layer(flat, position)
        projected[index] = flat.permute(0, 2, 1).reshape(-1, spec.d_model, height, width).contiguous()

        levels = len(spec.encoder_in_channels)
        fpn = [projected[-1]]
        for idx in range(levels - 1, 0, -1):
            high = self.lateral_convs[levels - 1 - idx](fpn[0])
            fpn[0] = high
            upsampled = F.interpolate(high, scale_factor=2.0, mode="nearest")
            fpn.insert(0, self.fpn_blocks[levels - 1 - idx](
                torch.concat([upsampled, projected[idx - 1]], dim=1)))

        pan = [fpn[0]]
        for idx in range(levels - 1):
            down = self.downsample_convs[idx](pan[-1])
            pan.append(self.pan_blocks[idx](torch.concat([down, fpn[idx + 1]], dim=1)))
        return pan

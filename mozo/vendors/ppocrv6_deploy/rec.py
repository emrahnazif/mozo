# SPDX-License-Identifier: Apache-2.0
# Extracted from transformers 5.20.2, src/transformers/models/pp_ocrv6_small_rec/ and
# pp_ocrv6_tiny_rec/modeling_*.py. Copyright 2026 The PaddlePaddle Team and The HuggingFace Inc.
# team. Apache-2.0.
"""The text recognisers: PP-LCNetV4 over a 48-pixel line, pooled to one row, then a CTC head.

``medium`` and ``small`` put a two-block SVTR encoder before the classifier
(``PPOCRV6SmallRec``; medium is the same class, wider). ``tiny`` replaces it with two 1-D
convolutions (``PPOCRV6TinyRec``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Recogniser
from .lcnetv4 import LCNetV4

_LAYER_NORM_EPS = 1e-6  # the svtr configs' ``layer_norm_eps``, not torch's 1e-5


class _Body(nn.Module):
    def __init__(self, spec: Recogniser) -> None:
        super().__init__()
        self.backbone = LCNetV4(spec.backbone)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(self.backbone(x)[-1], (3, 2))


class SvtrConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, int],
                 groups: int = 1) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                     stride=1, padding=(kernel_size[0] // 2, kernel_size[1] // 2),
                                     bias=False, groups=groups)
        self.normalization = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.normalization(self.convolution(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        # q and v carry a bias, k does not: upstream concatenates a zero k-bias in the middle,
        # and the checkpoint stores the concatenated vector.
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.projection = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.size()
        qkv = self.qkv(x).reshape(batch, length, 3, self.heads, dim // self.heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # Upstream dispatches through transformers' attention interface, which PaddleOCR leaves
        # at its default: scaled_dot_product_attention. Eager attention is other arithmetic.
        out = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], attn_mask=None,
                                             dropout_p=0.0, scale=self.scale, is_causal=False)
        out = out.transpose(1, 2).contiguous()
        return self.projection(out.reshape(batch, length, -1).contiguous())


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.activation = nn.SiLU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(x)))


class SvtrBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.self_attn = Attention(dim, heads)
        self.layer_norm1 = nn.LayerNorm(dim, eps=_LAYER_NORM_EPS)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.layer_norm2 = nn.LayerNorm(dim, eps=_LAYER_NORM_EPS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))


class SvtrEncoder(nn.Module):
    def __init__(self, spec: Recogniser) -> None:
        super().__init__()
        channels = spec.backbone.stage_channels[-1]
        hidden = spec.hidden_size
        self.conv_block = nn.ModuleList([
            SvtrConv(channels, hidden, (1, 1)),                          # the residual
            SvtrConv(channels, hidden, (1, 1)),                          # reduce
            SvtrConv(hidden, hidden, spec.conv_kernel, groups=hidden),   # local
        ])
        self.svtr_block = nn.ModuleList(
            SvtrBlock(hidden, spec.heads, spec.mlp_ratio) for _ in range(spec.depth))
        self.norm = nn.LayerNorm(hidden, eps=_LAYER_NORM_EPS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv_block[0](x)
        x = self.conv_block[1](x)
        x = x + self.conv_block[2](x)
        batch, channels, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        for block in self.svtr_block:
            x = block(x)
        x = self.norm(x)
        x = x.view(batch, height, width, channels).permute(0, 3, 1, 2)
        x = x + residual
        return x.squeeze(2).transpose(1, 2)


class SvtrHead(nn.Module):
    def __init__(self, spec: Recogniser) -> None:
        super().__init__()
        self.encoder = SvtrEncoder(spec)
        self.head = nn.Linear(spec.hidden_size, spec.classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class ConvCtcHead(nn.Module):
    def __init__(self, spec: Recogniser) -> None:
        super().__init__()
        channels = spec.backbone.stage_channels[-1]
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=5, padding=2, groups=channels,
                               bias=False)
        self.norm1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm1d(channels)
        self.act_fn = nn.Hardswish()
        self.fc1 = nn.Linear(channels, spec.hidden_size)
        self.fc2 = nn.Linear(spec.hidden_size, spec.classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(2)
        x = self.act_fn(self.norm1(self.conv1(x)))
        x = self.act_fn(self.norm2(self.conv2(x)))
        return self.fc2(self.fc1(x.transpose(1, 2)))


class TextRecogniser(nn.Module):
    """Normalised ``(N, 3, 48, W)`` BGR in, per-step class probabilities ``(N, W/8, classes)``."""

    def __init__(self, spec: Recogniser) -> None:
        super().__init__()
        self.model = _Body(spec)
        self.head = SvtrHead(spec) if spec.head == "svtr" else ConvCtcHead(spec)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        logits = self.head(self.model(pixels))
        return F.softmax(logits, dim=2, dtype=torch.float32).to(logits.dtype)

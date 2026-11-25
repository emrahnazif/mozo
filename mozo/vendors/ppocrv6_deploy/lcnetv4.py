# SPDX-License-Identifier: Apache-2.0
# Extracted from transformers 5.20.2, src/transformers/models/pp_lcnet_v4/modeling_pp_lcnet_v4.py
# Copyright 2026 The PaddlePaddle Team and The HuggingFace Inc. team. Apache-2.0.
"""PP-LCNetV4, the backbone under every PP-OCRv6 detector and recogniser.

Module and attribute names are upstream's, because the published checkpoints' keys are:
``encoder.convolution.stem1.convolution.weight`` greps over there unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Backbone


class ConvLayer(nn.Module):
    """Convolution, batch norm, then an optional activation. Never bias, upstream's ``bias=False``."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1,
                 groups: int = 1, activation: str | None = "relu") -> None:
        super().__init__()
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                     stride=stride, groups=groups,
                                     padding=(kernel_size - 1) // 2, bias=False)
        self.normalization = nn.BatchNorm2d(out_channels)
        # Every published variant has ``use_learnable_affine_block: false``, so upstream's
        # ``lab`` is an ``nn.Identity`` and is not carried.
        self.activation = nn.ReLU() if activation == "relu" else nn.Identity()
        if activation not in ("relu", None):
            raise ValueError(f"unsupported activation {activation!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.normalization(self.convolution(x)))


class SqueezeExcitation(nn.Module):
    def __init__(self, channel: int, reduction: int) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Upstream builds this as a ModuleList of [conv, ReLU, conv, Hardsigmoid], so the two
        # convolutions are ``convolutions.0`` and ``convolutions.2`` in the checkpoint.
        self.convolutions = nn.ModuleList([
            nn.Conv2d(channel, channel // reduction, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Hardsigmoid(),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.avg_pool(x)
        for layer in self.convolutions:
            x = layer(x)
        return residual * x


class LargeStem(nn.Module):
    def __init__(self, spec: Backbone) -> None:
        super().__init__()
        c0, c1, c2 = spec.stem_channels
        s = spec.stem_strides
        self.stem1 = ConvLayer(c0, c1, 3, stride=s[0])
        self.stem2a = ConvLayer(c1, c1 // 2, 2, stride=s[1])
        self.stem2b = ConvLayer(c1 // 2, c1, 2, stride=s[2])
        self.stem3 = ConvLayer(c1 * 2, c1, 3, stride=s[3])
        self.stem4 = ConvLayer(c1, c2, 1, stride=s[4])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem1(x)
        x = F.pad(x, (0, 1, 0, 1))
        branch = self.stem2a(x)
        branch = F.pad(branch, (0, 1, 0, 1))
        branch = self.stem2b(branch)
        pooled = self.pool(x)
        x = torch.cat([pooled, branch], dim=1)
        x = self.stem3(x)
        return self.stem4(x)


class SmallStem(nn.Module):
    """The tiny recogniser's stem: two stride-2 convolutions with a GELU between them."""

    def __init__(self, spec: Backbone) -> None:
        super().__init__()
        c0, c1, c2 = spec.stem_channels
        self.conv1 = ConvLayer(c0, c1, 3, stride=2, activation=None)
        self.act_fn = nn.GELU()
        self.conv2 = ConvLayer(c1, c2, 3, stride=2, activation=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.act_fn(self.conv1(x)))


class DepthwiseSeparable(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride, kernel_size: int,
                 use_se: bool, reduction: int) -> None:
        super().__init__()
        # ``stride`` is 1, 2 or (2, 1); upstream tests ``stride == 1``, which a tuple never meets.
        self.has_residual = in_channels == out_channels and stride == 1
        # Where upstream's training graph has a re-parameterisable depthwise branch, the
        # published checkpoint holds it already folded: one biased convolution, no norm.
        if self.has_residual:
            self.token_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                        stride=stride, padding=kernel_size // 2,
                                        groups=in_channels)
        else:
            self.token_conv = ConvLayer(in_channels, in_channels, kernel_size, stride=stride,
                                        groups=in_channels, activation=None)
        self.token_squeeze_excitation = (SqueezeExcitation(in_channels, reduction) if use_se
                                         else nn.Identity())
        self.channel_conv1 = ConvLayer(in_channels, in_channels * 2, 1, activation=None)
        self.channel_act_fn = nn.GELU()
        self.channel_conv2 = ConvLayer(in_channels * 2, out_channels, 1, activation=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_conv(x)
        x = self.token_squeeze_excitation(x)
        residual = x
        x = self.channel_conv1(x)
        x = self.channel_act_fn(x)
        x = self.channel_conv2(x)
        return residual + x if self.has_residual else x


class Stage(nn.Module):
    def __init__(self, blocks, reduction: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            DepthwiseSeparable(cin, cout, stride, kernel, use_se, reduction)
            for kernel, cin, cout, stride, use_se in blocks
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class Encoder(nn.Module):
    def __init__(self, spec: Backbone) -> None:
        super().__init__()
        self.convolution = LargeStem(spec) if spec.stem_type == "large" else SmallStem(spec)
        self.blocks = nn.ModuleList([Stage(stage, spec.reduction) for stage in spec.blocks])


class LCNetV4(nn.Module):
    """Returns the four stage outputs, shallowest first -- what upstream's backbone hands its
    ``out_features`` stage1 to stage4."""

    def __init__(self, spec: Backbone) -> None:
        super().__init__()
        self.encoder = Encoder(spec)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.encoder.convolution(x)
        stages = []
        for stage in self.encoder.blocks:
            x = stage(x)
            stages.append(x)
        return stages

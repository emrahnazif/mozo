# SPDX-License-Identifier: Apache-2.0
# Extracted from transformers 5.20.2, src/transformers/models/pp_ocrv6_medium_det/ and
# pp_ocrv6_small_det/modeling_*.py. Copyright 2026 The PaddlePaddle Team and The HuggingFace Inc.
# team. Apache-2.0.
"""The text detectors: PP-LCNetV4, a feature pyramid, and a DB head that draws a text-probability
map at the input's resolution.

Two necks. ``medium`` has a large-kernel path-aggregation neck with intra-class blocks
(``PPOCRV6MediumDet``); ``small`` and ``tiny`` share a lighter residual-SE neck
(``PPOCRV6SmallDet``; tiny is the same class at other widths). The head is the same for all.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Detector
from .lcnetv4 import LCNetV4


class ConvBatchNorm(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 1, activation: bool = True, bias: bool = False,
                 transpose: bool = False) -> None:
        super().__init__()
        if transpose:
            self.convolution = nn.ConvTranspose2d(in_channels, out_channels,
                                                  kernel_size=kernel_size, stride=stride)
        else:
            self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, bias=bias)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act_fn = nn.ReLU() if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act_fn(self.norm(self.convolution(x)))


class Head(nn.Module):
    """DB's probability branch. Upstream's inference graph carries no threshold branch."""

    def __init__(self, channels: int, kernels: tuple[int, int, int]) -> None:
        super().__init__()
        quarter = channels // 4
        self.conv_down = ConvBatchNorm(channels, quarter, kernels[0], padding=kernels[0] // 2)
        self.conv_up = ConvBatchNorm(quarter, quarter, kernels[1], stride=2, transpose=True)
        self.conv_final = nn.ConvTranspose2d(quarter, 1, kernel_size=kernels[2], stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.conv_final(self.conv_up(self.conv_down(x))))


# -- medium: large-kernel PAN with intra-class blocks ----------------------------------------------

#: ``intraclass_block_config`` in the medium detector's config: Conv2d (kernel, stride, padding).
_INTRACLASS = {
    "reduce_channel": (1, 1, 0),
    "vertical_long_to_small_conv_longratio": ((7, 1), (1, 1), (3, 0)),
    "vertical_long_to_small_conv_midratio": ((5, 1), (1, 1), (2, 0)),
    "vertical_long_to_small_conv_shortratio": ((3, 1), (1, 1), (1, 0)),
    "horizontal_small_to_long_conv_longratio": ((1, 7), (1, 1), (0, 3)),
    "horizontal_small_to_long_conv_midratio": ((1, 5), (1, 1), (0, 2)),
    "horizontal_small_to_long_conv_shortratio": ((1, 3), (1, 1), (0, 1)),
    "symmetric_conv_long_longratio": ((7, 7), (1, 1), (3, 3)),
    "symmetric_conv_long_midratio": ((5, 5), (1, 1), (2, 2)),
    "symmetric_conv_long_shortratio": ((3, 3), (1, 1), (1, 1)),
}


class IntraclassBlock(nn.Module):
    def __init__(self, channels: int, reduce_factor: int = 2) -> None:
        super().__init__()
        reduced = channels // reduce_factor
        self.conv_reduce_channel = nn.Conv2d(channels, reduced, *_INTRACLASS["reduce_channel"])
        for name, geometry in _INTRACLASS.items():
            if name != "reduce_channel":
                setattr(self, name, nn.Conv2d(reduced, reduced, *geometry))
        self.conv_final = ConvBatchNorm(reduced, channels, 1, stride=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv_reduce_channel(x)
        # The order of the three terms is upstream's: float addition does not commute in bits.
        x = (self.symmetric_conv_long_longratio(x)
             + self.vertical_long_to_small_conv_longratio(x)
             + self.horizontal_small_to_long_conv_longratio(x))
        x = (self.symmetric_conv_long_midratio(x)
             + self.vertical_long_to_small_conv_midratio(x)
             + self.horizontal_small_to_long_conv_midratio(x))
        x = (self.symmetric_conv_long_shortratio(x)
             + self.vertical_long_to_small_conv_shortratio(x)
             + self.horizontal_small_to_long_conv_shortratio(x))
        return residual + self.conv_final(x)


class LargeKernelNeck(nn.Module):
    def __init__(self, stage_channels: tuple[int, ...], channels: int) -> None:
        super().__init__()
        quarter = channels // 4
        self.input_channel_adjustment_convolution = nn.ModuleList(
            nn.Conv2d(c, channels, kernel_size=1, bias=False) for c in stage_channels)
        self.input_feature_projection_convolution = nn.ModuleList(
            nn.Conv2d(channels, quarter, kernel_size=9, padding=4, bias=True)
            for _ in stage_channels)
        self.path_aggregation_head_convolution = nn.ModuleList(
            nn.Conv2d(quarter, quarter, kernel_size=3, padding=1, stride=2, bias=False)
            for _ in stage_channels[1:])
        self.path_aggregation_lateral_convolution = nn.ModuleList(
            nn.Conv2d(quarter, quarter, kernel_size=9, padding=4, bias=True)
            for _ in stage_channels)
        self.intraclass_blocks = nn.ModuleList(IntraclassBlock(quarter) for _ in stage_channels)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        n = len(features)
        adjusted = [conv(f) for conv, f in zip(self.input_channel_adjustment_convolution, features)]

        top_down = [None] * n
        top_down[n - 1] = adjusted[n - 1]
        for i in range(n - 2, -1, -1):
            top_down[i] = adjusted[i] + F.interpolate(top_down[i + 1], scale_factor=2,
                                                      mode="nearest")

        projected = [conv(top_down[i] if i < n - 1 else adjusted[-1])
                     for i, conv in enumerate(self.input_feature_projection_convolution)]

        bottom_up = [projected[0]]
        for i in range(1, n):
            bottom_up.append(projected[i] + self.path_aggregation_head_convolution[i - 1](
                bottom_up[i - 1]))

        lateral = [conv(projected[0] if i == 0 else bottom_up[i])
                   for i, conv in enumerate(self.path_aggregation_lateral_convolution)]
        refined = [block(f) for block, f in zip(self.intraclass_blocks, lateral)]
        upsampled = [F.interpolate(f, scale_factor=s, mode="nearest") if s > 1 else f
                     for f, s in zip(refined, (1, 2, 4, 8))]
        return torch.cat(upsampled[::-1], dim=1)


# -- small and tiny: residual squeeze-excitation neck ----------------------------------------------

class NeckSqueezeExcitation(nn.Module):
    """Not the backbone's SE: a hard sigmoid written out as ``clamp(0.2x + 0.5)``."""

    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, stride=1, padding=0)
        self.act_fn = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv2(self.act_fn(self.conv1(self.avg_pool(x))))
        x = torch.clamp(0.2 * x + 0.5, min=0.0, max=1.0)
        return residual * x


class ResidualSE(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 reduction: int) -> None:
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                 padding=kernel_size // 2, bias=False)
        self.squeeze_excitation_block = NeckSqueezeExcitation(out_channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        return x + self.squeeze_excitation_block(x)


class DepthwiseSE(nn.Module):
    def __init__(self, channels: int, kernel_size: int, reduction: int) -> None:
        super().__init__()
        self.depthwise_convolution = nn.Conv2d(channels, channels, kernel_size=kernel_size,
                                               stride=1, padding=kernel_size // 2,
                                               groups=channels, bias=True)
        self.squeeze_excitation_module = NeckSqueezeExcitation(channels // 4, reduction)
        self.pointwise_convolution = nn.Conv2d(channels, channels // 4, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pointwise_convolution(self.depthwise_convolution(x))
        return x + self.squeeze_excitation_module(x)


class ResidualSENeck(nn.Module):
    def __init__(self, stage_channels: tuple[int, ...], channels: int, kernel_size: int,
                 reduction: int = 4) -> None:
        super().__init__()
        self.insert_conv = nn.ModuleList(
            ResidualSE(c, channels, 1, reduction) for c in stage_channels)
        self.input_conv = nn.ModuleList(
            DepthwiseSE(channels, kernel_size, reduction) for _ in stage_channels)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        fused = [conv(f) for conv, f in zip(self.insert_conv, features)]
        for i in range(2, -1, -1):
            fused[i] = fused[i] + F.interpolate(fused[i + 1], scale_factor=2, mode="nearest")
        refined = [conv(f) for conv, f in zip(self.input_conv, fused)]
        upsampled = [F.interpolate(f, scale_factor=s, mode="nearest") if s != 1 else f
                     for f, s in zip(refined, (1, 2, 4, 8))]
        return torch.cat(upsampled[::-1], dim=1)


class _Body(nn.Module):
    def __init__(self, spec: Detector) -> None:
        super().__init__()
        self.backbone = LCNetV4(spec.backbone)
        stages = spec.backbone.stage_channels
        if spec.neck == "large_kernel":
            self.neck = LargeKernelNeck(stages, spec.neck_channels)
        else:
            self.neck = ResidualSENeck(stages, spec.neck_channels, spec.dilated_kernel_size)


class TextDetector(nn.Module):
    """Normalised ``(N, 3, H, W)`` BGR in, text probability ``(N, 1, H, W)`` out."""

    def __init__(self, spec: Detector) -> None:
        super().__init__()
        self.model = _Body(spec)
        self.head = Head(spec.neck_channels, spec.kernel_list)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.head(self.model.neck(self.model.backbone(pixels)))

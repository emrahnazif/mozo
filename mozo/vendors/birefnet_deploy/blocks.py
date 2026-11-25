# ------------------------------------------------------------------------
# BiRefNet -- Bilateral Reference for High-Resolution Dichotomous Image Segmentation
# Copyright (c) 2024 ZhengPeng. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
"""The decoder's building blocks: deformable ASPP, the decoder and lateral blocks, the patch convs.

Extracted from the ``deform_conv.py``, ``decoder_blocks.py``, ``lateral_blocks.py``, ``aspp.py``
and ``SimpleConvs`` sections of upstream's remote-code ``birefnet.py``, with each block reduced
to the one configuration every published checkpoint uses (see :mod:`.config`). Module and
attribute names are upstream's, because they are the checkpoint's keys.

Removed, all inert at inference: ``nn.Dropout(0.5)`` at the end of ``ASPPDeformable`` (the
identity at eval), the plain-``ASPP`` and ``ResBlk`` alternatives no checkpoint selects, and the
``nn.init.constant_`` calls the strict load overwrites.

**The deformable convolution is torchvision's**, ``torchvision.ops.deform_conv2d``, called exactly
as upstream calls it. It is the one operator in this model that ONNX Runtime cannot run and
TensorRT runs only through a plugin, so :mod:`.graph` carries a decomposition that an exporter
can swap in. The torch runtime never uses it: this module is what the gate holds equal to
upstream.
"""

from __future__ import annotations

__all__ = ["ASPPDeformable", "BasicDecBlk", "BasicLatBlk", "DeformableConv2d", "SimpleConvs"]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


class DeformableConv2d(nn.Module):
    """Modulated deformable convolution (DCNv2): offsets and a ``2 * sigmoid`` mask, both learned.

    The modulator is ``2 * sigmoid(...)``, so a mask value ranges over ``(0, 2)`` rather than
    DCNv2's usual ``(0, 1)`` -- upstream's choice, which makes an untrained modulator (zero
    weights, zero bias) the identity.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int) -> None:
        super().__init__()
        self.stride = (1, 1)
        self.padding = padding
        k = (kernel_size, kernel_size)
        self.offset_conv = nn.Conv2d(in_channels, 2 * k[0] * k[1], kernel_size=k, stride=1,
                                     padding=padding, bias=True)
        self.modulator_conv = nn.Conv2d(in_channels, k[0] * k[1], kernel_size=k, stride=1,
                                        padding=padding, bias=True)
        self.regular_conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=1,
                                      padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset_conv(x)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        return self.deform(x, offset, modulator)

    def deform(self, x: torch.Tensor, offset: torch.Tensor, modulator: torch.Tensor) -> torch.Tensor:
        """The sampling convolution alone. An exporter replaces this method, and nothing else."""
        return deform_conv2d(input=x, offset=offset, weight=self.regular_conv.weight,
                             bias=self.regular_conv.bias, padding=self.padding, mask=modulator,
                             stride=self.stride)


class _ASPPModuleDeformable(nn.Module):
    """Deformable conv, BatchNorm, ReLU."""

    def __init__(self, in_channels: int, planes: int, kernel_size: int, padding: int) -> None:
        super().__init__()
        self.atrous_conv = DeformableConv2d(in_channels, planes, kernel_size, padding)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.atrous_conv(x)))


class ASPPDeformable(nn.Module):
    """Four deformable branches (1x1, then 1x1, 3x3, 7x7) and a global-pool branch, fused 1x1.

    The first two branches are both 1x1: upstream builds ``aspp1`` and then a ``ModuleList`` over
    ``parallel_block_sizes=[1, 3, 7]``. Two 1x1 deformable branches is not a typo to fix.
    """

    def __init__(self, in_channels: int, parallel_block_sizes: tuple[int, ...] = (1, 3, 7)) -> None:
        super().__init__()
        planes = 256
        self.aspp1 = _ASPPModuleDeformable(in_channels, planes, 1, padding=0)
        self.aspp_deforms = nn.ModuleList([
            _ASPPModuleDeformable(in_channels, planes, size, padding=int(size // 2))
            for size in parallel_block_sizes])
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, planes, 1, stride=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True))
        self.conv1 = nn.Conv2d(planes * (2 + len(parallel_block_sizes)), in_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.aspp1(x)
        x_aspp_deforms = [aspp_deform(x) for aspp_deform in self.aspp_deforms]
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x1.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, *x_aspp_deforms, x5), dim=1)
        return self.relu(self.bn1(self.conv1(x)))


class BasicDecBlk(nn.Module):
    """3x3 conv to 64, BN, ReLU, deformable ASPP, 3x3 conv out, BN. No residual."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        inter_channels = 64
        self.conv_in = nn.Conv2d(in_channels, inter_channels, 3, 1, padding=1)
        self.relu_in = nn.ReLU(inplace=True)
        self.dec_att = ASPPDeformable(in_channels=inter_channels)
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, 1, padding=1)
        self.bn_in = nn.BatchNorm2d(inter_channels)
        self.bn_out = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu_in(self.bn_in(self.conv_in(x)))
        x = self.dec_att(x)
        return self.bn_out(self.conv_out(x))


class BasicLatBlk(nn.Module):
    """A 1x1 convolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SimpleConvs(nn.Module):
    """Two 3x3 convolutions with nothing between them -- no norm, no activation, as upstream."""

    def __init__(self, in_channels: int, out_channels: int, inter_channels: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, inter_channels, 3, 1, 1)
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_out(self.conv1(x))

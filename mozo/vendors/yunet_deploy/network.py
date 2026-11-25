# SPDX-License-Identifier: Apache-2.0
# Derived from libfacedetection.train, Copyright (c) 2022-2026, Shiqi Yu (BSD-3-Clause: see LICENSE).
"""YuNet's network: a depthwise-separable backbone, a tiny top-down FPN and a decoupled head.

Extracted from ``yunet_train/models/{layers,backbone,neck}.py`` and ``tasks/face/head.py`` at
commit ``dca340aa``. Module and attribute names are upstream's, so the published state dict loads
strictly with no renaming. What is not carried: weight initialisation (every parameter is
overwritten by the checkpoint), the ``stacked_convs`` branches (0 in both published geometries),
and the ``torch.onnx.is_in_onnx_export()`` branch, which flattens and sigmoids the outputs inside
the graph for OpenCV -- mozo's graphs return the same raw maps as this module, so the decode is
one piece of code whatever ran the network.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import KEYPOINTS, Spec

__all__ = ["YuNet"]


class ConvDPUnit(nn.Module):
    """A 1x1 conv then a 3x3 depthwise conv, both biased, then (optionally) BN and ReLU.

    Note the order: the pointwise conv comes *first*, the reverse of a MobileNet block.
    """

    def __init__(self, in_channels: int, out_channels: int, withBNRelu: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True, groups=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True,
                               groups=out_channels)
        self.withBNRelu = withBNRelu
        if withBNRelu:
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        if self.withBNRelu:
            x = self.bn(x)
            x = self.relu(x)
        return x


class Conv_head(nn.Module):
    """The stem: a strided 3x3 conv with BN and ReLU, then one :class:`ConvDPUnit`."""

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 2, 1, bias=True, groups=1)
        self.conv2 = ConvDPUnit(mid_channels, out_channels, True)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.relu1(self.bn1(self.conv1(x))))


class Conv4layerBlock(nn.Module):
    """Two :class:`ConvDPUnit`, the first keeping the width, the second changing it."""

    def __init__(self, in_channels: int, out_channels: int, withBNRelu: bool = True):
        super().__init__()
        self.conv1 = ConvDPUnit(in_channels, in_channels, True)
        self.conv2 = ConvDPUnit(in_channels, out_channels, withBNRelu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class YuNetBackbone(nn.Module):
    """Six stages, ``model0`` .. ``model5``; a 2x2 max pool follows the stages in
    ``downsample_idx``, and the stages in ``out_idx`` are read *before* their pool."""

    def __init__(self, spec: Spec):
        super().__init__()
        self.layer_num = len(spec.stage_channels)
        self.downsample_idx = spec.downsample_idx
        self.out_idx = spec.out_idx
        self.model0 = Conv_head(*spec.stage_channels[0])
        for i in range(1, self.layer_num):
            self.add_module(f"model{i}", Conv4layerBlock(*spec.stage_channels[i]))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        out = []
        for i in range(self.layer_num):
            x = getattr(self, f"model{i}")(x)
            if i in self.out_idx:
                out.append(x)
            if i in self.downsample_idx:
                x = F.max_pool2d(x, 2)
        return out


class TFPN(nn.Module):
    """Top-down: from the coarsest level, each level is convolved and its nearest x2 upsampling
    *added* to the next finer level, which is convolved in turn."""

    def __init__(self, spec: Spec):
        super().__init__()
        widths = (spec.channels,) * len(spec.out_idx)
        self.lateral_convs = nn.ModuleList(ConvDPUnit(c, c, True) for c in widths)

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        feats = list(feats)
        for i in range(len(feats) - 1, 0, -1):
            feats[i] = self.lateral_convs[i](feats[i])
            feats[i - 1] = feats[i - 1] + F.interpolate(feats[i], scale_factor=2.0,
                                                        mode="nearest")
        feats[0] = self.lateral_convs[0](feats[0])
        return feats


class YuNetHead(nn.Module):
    """Per level: optionally one shared :class:`ConvDPUnit`, then four bare (no BN, no ReLU)
    :class:`ConvDPUnit` branches -- class score, box, objectness, landmarks."""

    def __init__(self, spec: Spec):
        super().__init__()
        self.shared_stack_convs = spec.shared_stacked_convs
        width = spec.channels
        levels = range(len(spec.strides))
        if self.shared_stack_convs > 0:
            self.multi_level_share_convs = nn.ModuleList(
                nn.Sequential(*(ConvDPUnit(width, width) for _ in range(self.shared_stack_convs)))
                for _ in levels)
        self.multi_level_cls = nn.ModuleList(ConvDPUnit(width, 1, False) for _ in levels)
        self.multi_level_bbox = nn.ModuleList(ConvDPUnit(width, 4, False) for _ in levels)
        self.multi_level_kps = nn.ModuleList(ConvDPUnit(width, KEYPOINTS * 2, False)
                                             for _ in levels)
        self.multi_level_obj = nn.ModuleList(ConvDPUnit(width, 1, False) for _ in levels)

    def forward(self, feats: list[torch.Tensor]):
        if self.shared_stack_convs > 0:
            feats = [convs(feat) for feat, convs in zip(feats, self.multi_level_share_convs)]
        cls_preds = [convs(feat) for feat, convs in zip(feats, self.multi_level_cls)]
        bbox_preds = [convs(feat) for feat, convs in zip(feats, self.multi_level_bbox)]
        obj_preds = [convs(feat) for feat, convs in zip(feats, self.multi_level_obj)]
        kps_preds = [convs(feat) for feat, convs in zip(feats, self.multi_level_kps)]
        return cls_preds, bbox_preds, obj_preds, kps_preds


class YuNet(nn.Module):
    """The whole network: ``(1, 3, H, W)`` BGR float in 0..255, ``H`` and ``W`` multiples of 32.

    Returns four lists of three ``NCHW`` maps (strides 8, 16, 32), raw logits and offsets:
    ``(cls, bbox, obj, kps)`` with 1, 4, 1 and 10 channels -- upstream's order.
    """

    def __init__(self, spec: Spec):
        super().__init__()
        self.backbone = YuNetBackbone(spec)
        self.neck = TFPN(spec)
        self.bbox_head = YuNetHead(spec)

    def forward(self, img: torch.Tensor):
        return self.bbox_head(self.neck(self.backbone(img)))

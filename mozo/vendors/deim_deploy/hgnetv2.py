"""
reference
- https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py

Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.

Upstream's ``engine/backbone/hgnetv2.py`` and ``FrozenBatchNorm2d`` from
``engine/backbone/common.py``, reduced to inference: no ImageNet-pretrained download, no
parameter freezing, no initialisers, and only the arch configs a released checkpoint uses
(B0, B2, B4, B5; B1, B3 and B6 were dropped).
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenBatchNorm2d(nn.Module):
    """copy and modified from https://github.com/facebookresearch/detr/blob/master/models/backbone.py
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.

    Not the same arithmetic as an eval-mode ``nn.BatchNorm2d``: this folds the statistics into
    one scale and one shift and applies ``x * scale + bias``. The Objects365-pretrained X
    checkpoint was trained (and is evaluated) with it, the COCO-only checkpoints are not.
    """

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        n = num_features
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps
        self.num_features = n

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class LearnableAffineBlock(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale_value]), requires_grad=False)
        self.bias = nn.Parameter(torch.tensor([bias_value]), requires_grad=False)

    def forward(self, x):
        return self.scale * x + self.bias


class ConvBNAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride=1, groups=1, use_act=True, use_lab=False):
        super().__init__()
        self.use_act = use_act
        self.use_lab = use_lab
        self.conv = nn.Conv2d(
            in_chs,
            out_chs,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_chs)
        if self.use_act:
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()
        if self.use_act and self.use_lab:
            self.lab = LearnableAffineBlock()
        else:
            self.lab = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.lab(x)
        return x


class LightConvBNAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, use_lab=False):
        super().__init__()
        self.conv1 = ConvBNAct(in_chs, out_chs, kernel_size=1, use_act=False, use_lab=use_lab)
        self.conv2 = ConvBNAct(out_chs, out_chs, kernel_size=kernel_size, groups=out_chs,
                               use_act=True, use_lab=use_lab)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class StemBlock(nn.Module):
    # for HGNetv2
    def __init__(self, in_chs, mid_chs, out_chs, use_lab=False):
        super().__init__()
        self.stem1 = ConvBNAct(in_chs, mid_chs, kernel_size=3, stride=2, use_lab=use_lab)
        self.stem2a = ConvBNAct(mid_chs, mid_chs // 2, kernel_size=2, stride=1, use_lab=use_lab)
        self.stem2b = ConvBNAct(mid_chs // 2, mid_chs, kernel_size=2, stride=1, use_lab=use_lab)
        self.stem3 = ConvBNAct(mid_chs * 2, mid_chs, kernel_size=3, stride=2, use_lab=use_lab)
        self.stem4 = ConvBNAct(mid_chs, out_chs, kernel_size=1, stride=1, use_lab=use_lab)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)

    def forward(self, x):
        x = self.stem1(x)
        x = F.pad(x, (0, 1, 0, 1))
        x2 = self.stem2a(x)
        x2 = F.pad(x2, (0, 1, 0, 1))
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HG_Block(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, layer_num, kernel_size=3, residual=False,
                 light_block=False, use_lab=False):
        super().__init__()
        self.residual = residual

        self.layers = nn.ModuleList()
        for i in range(layer_num):
            if light_block:
                self.layers.append(
                    LightConvBNAct(in_chs if i == 0 else mid_chs, mid_chs, kernel_size=kernel_size,
                                   use_lab=use_lab))
            else:
                self.layers.append(
                    ConvBNAct(in_chs if i == 0 else mid_chs, mid_chs, kernel_size=kernel_size,
                              stride=1, use_lab=use_lab))

        # feature aggregation: the "se" form, which is ``HG_Stage``'s default and which
        # ``HGNetv2`` never overrides, although ``HG_Block`` itself defaults to "ese".
        total_chs = in_chs + layer_num * mid_chs
        aggregation_squeeze_conv = ConvBNAct(total_chs, out_chs // 2, kernel_size=1, stride=1,
                                             use_lab=use_lab)
        aggregation_excitation_conv = ConvBNAct(out_chs // 2, out_chs, kernel_size=1, stride=1,
                                                use_lab=use_lab)
        self.aggregation = nn.Sequential(aggregation_squeeze_conv, aggregation_excitation_conv)

        # drop_path is 0 in every released config, where upstream builds an Identity.
        self.drop_path = nn.Identity()

    def forward(self, x):
        identity = x
        output = [x]
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = torch.cat(output, dim=1)
        x = self.aggregation(x)
        if self.residual:
            x = self.drop_path(x) + identity
        return x


class HG_Stage(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, block_num, layer_num, downsample=True,
                 light_block=False, kernel_size=3, use_lab=False):
        super().__init__()
        if downsample:
            self.downsample = ConvBNAct(in_chs, in_chs, kernel_size=3, stride=2, groups=in_chs,
                                        use_act=False, use_lab=use_lab)
        else:
            self.downsample = nn.Identity()

        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                HG_Block(
                    in_chs if i == 0 else out_chs,
                    mid_chs,
                    out_chs,
                    layer_num,
                    residual=False if i == 0 else True,
                    kernel_size=kernel_size,
                    light_block=light_block,
                    use_lab=use_lab,
                )
            )
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class HGNetv2(nn.Module):
    """HGNetV2 backbone.

    Args:
        name: Which arch config -- ``B0`` (N, S), ``B2`` (M), ``B4`` (L), ``B5`` (X).
        use_lab: Whether to use LearnableAffineBlock in network.
        return_idx: Which stages' outputs feed the encoder.
        freeze_norm: Swap every BatchNorm2d for :class:`FrozenBatchNorm2d`, as the checkpoint
            was trained with. Changes the arithmetic, so it is part of the variant's geometry.
    """

    arch_configs = {
        'B0': {
            'stem_channels': [3, 16, 16],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [16, 16, 64, 1, False, False, 3, 3],
                "stage2": [64, 32, 256, 1, True, False, 3, 3],
                "stage3": [256, 64, 512, 2, True, True, 5, 3],
                "stage4": [512, 128, 1024, 1, True, True, 5, 3],
            },
        },
        'B2': {
            'stem_channels': [3, 24, 32],
            'stage_config': {
                "stage1": [32, 32, 96, 1, False, False, 3, 4],
                "stage2": [96, 64, 384, 1, True, False, 3, 4],
                "stage3": [384, 128, 768, 3, True, True, 5, 4],
                "stage4": [768, 256, 1536, 1, True, True, 5, 4],
            },
        },
        'B4': {
            'stem_channels': [3, 32, 48],
            'stage_config': {
                "stage1": [48, 48, 128, 1, False, False, 3, 6],
                "stage2": [128, 96, 512, 1, True, False, 3, 6],
                "stage3": [512, 192, 1024, 3, True, True, 5, 6],
                "stage4": [1024, 384, 2048, 1, True, True, 5, 6],
            },
        },
        'B5': {
            'stem_channels': [3, 32, 64],
            'stage_config': {
                "stage1": [64, 64, 128, 1, False, False, 3, 6],
                "stage2": [128, 128, 512, 2, True, False, 3, 6],
                "stage3": [512, 256, 1024, 5, True, True, 5, 6],
                "stage4": [1024, 512, 2048, 2, True, True, 5, 6],
            },
        },
    }

    def __init__(self, name, use_lab=False, return_idx=(1, 2, 3), freeze_norm=False):
        super().__init__()
        self.use_lab = use_lab
        self.return_idx = list(return_idx)

        stem_channels = self.arch_configs[name]['stem_channels']
        stage_config = self.arch_configs[name]['stage_config']

        self.stem = StemBlock(in_chs=stem_channels[0], mid_chs=stem_channels[1],
                              out_chs=stem_channels[2], use_lab=use_lab)

        self.stages = nn.ModuleList()
        for k in stage_config:
            (in_channels, mid_channels, out_channels, block_num, downsample, light_block,
             kernel_size, layer_num) = stage_config[k]
            self.stages.append(
                HG_Stage(in_channels, mid_channels, out_channels, block_num, layer_num, downsample,
                         light_block, kernel_size, use_lab))

        if freeze_norm:
            self._freeze_norm(self)

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def forward(self, x) -> List[torch.Tensor]:
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs

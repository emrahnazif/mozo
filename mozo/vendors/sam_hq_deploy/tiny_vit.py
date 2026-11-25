# SPDX-License-Identifier: Apache-2.0
# --------------------------------------------------------
# TinyViT Model Architecture
# Copyright (c) 2022 Microsoft
# Adapted from LeViT and Swin Transformer
#   LeViT: (https://github.com/facebookresearch/levit)
#   Swin: (https://github.com/microsoft/swin-transformer)
# Build the TinyViT Model
# --------------------------------------------------------
"""MobileSAM's TinyViT image encoder, as HQ-SAM's ``vit_tiny`` ("Light HQ-SAM") builds it.

Reduced from ``segment_anything/modeling/tiny_vit_sam.py``. The arithmetic of every module that
runs is upstream's, line for line. What is gone, and why, is listed in ``PROVENANCE.md``; the two
removals that are not just dead code:

- **timm.** Upstream imports ``DropPath``, ``to_2tuple``, ``trunc_normal_`` and
  ``register_model`` from it. Stochastic depth is zero at inference, so every ``DropPath`` was an
  ``nn.Identity`` already; the other three are weight init and a model registry.
- **The cached attention bias.** Upstream's ``Attention.train(False)`` gathers
  ``attention_biases[:, attention_bias_idxs]`` into a buffer ``ab`` *at the moment ``eval()`` is
  called*. ``build_sam_vit_t`` calls ``eval()`` before it loads the checkpoint, so a model used
  straight from the builder attends with the zero biases it was initialised with; upstream's own
  demo calls ``eval()`` a second time, after the load, which is what makes it right. Here the
  same gather is done in ``forward`` from the loaded parameter, so there is no order in which it
  can be stale. It is the same indexing either way, so the numbers are the same.
"""

import itertools
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import LayerNorm2d

__all__ = ["TinyViT"]


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))


class PatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim, resolution, activation):
        super().__init__()
        self.patches_resolution = (resolution // 4, resolution // 4)
        n = embed_dim
        self.seq = nn.Sequential(
            Conv2d_BN(in_chans, n // 2, 3, 2, 1),
            activation(),
            Conv2d_BN(n // 2, n, 3, 2, 1),
        )

    def forward(self, x):
        return self.seq(x)


class MBConv(nn.Module):
    def __init__(self, in_chans, out_chans, expand_ratio, activation):
        super().__init__()
        hidden_chans = int(in_chans * expand_ratio)
        self.conv1 = Conv2d_BN(in_chans, hidden_chans, ks=1)
        self.act1 = activation()
        self.conv2 = Conv2d_BN(hidden_chans, hidden_chans,
                               ks=3, stride=1, pad=1, groups=hidden_chans)
        self.act2 = activation()
        self.conv3 = Conv2d_BN(hidden_chans, out_chans, ks=1)
        self.act3 = activation()

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x += shortcut
        x = self.act3(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, out_dim, activation):
        super().__init__()
        self.input_resolution = input_resolution
        self.act = activation()
        self.conv1 = Conv2d_BN(dim, out_dim, 1, 1, 0)
        # MobileSAM's change to TinyViT: the last merge keeps the resolution, so the encoder
        # ends at 64x64 (stride 16, SAM's embedding grid) rather than 32x32. Upstream keys it on
        # the output width, and so does this.
        stride_c = 2
        if out_dim == 320 or out_dim == 448 or out_dim == 576:
            stride_c = 1
        self.conv2 = Conv2d_BN(out_dim, out_dim, 3, stride_c, 1, groups=out_dim)
        self.conv3 = Conv2d_BN(out_dim, out_dim, 1, 1, 0)

    def forward(self, x):
        if x.ndim == 3:
            H, W = self.input_resolution
            B = len(x)
            # (B, C, H, W)
            x = x.view(B, H, W, -1).permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ConvLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, activation, downsample=None, out_dim=None,
                 conv_expand_ratio=4.):
        super().__init__()
        self.blocks = nn.ModuleList([
            MBConv(dim, dim, conv_expand_ratio, activation) for _ in range(depth)])
        self.downsample = (None if downsample is None else
                           downsample(input_resolution, dim=dim, out_dim=out_dim,
                                      activation=activation))

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.act = act_layer()

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads=8, attn_ratio=4, resolution=(14, 14)):
        super().__init__()
        # (h, w)
        assert isinstance(resolution, tuple) and len(resolution) == 2
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        h = self.dh + nh_kd * 2

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, h)
        self.proj = nn.Linear(self.dh, dim)

        points = list(itertools.product(
            range(resolution[0]), range(resolution[1])))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N),
                             persistent=False)

    def forward(self, x):  # x (B,N,C)
        B, N, _ = x.shape

        # Normalization
        x = self.norm(x)

        qkv = self.qkv(x)
        # (B, N, num_heads, d)
        q, k, v = qkv.view(B, N, self.num_heads, -
                           1).split([self.key_dim, self.key_dim, self.d], dim=3)
        # (B, num_heads, N, d)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Upstream reads a buffer gathered at ``eval()`` time; see the module docstring.
        attn = (
            (q @ k.transpose(-2, -1)) * self.scale
            +
            self.attention_biases[:, self.attention_bias_idxs]
        )
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = self.proj(x)
        return x


class TinyViTBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, mlp_ratio=4.,
                 local_conv_size=3, activation=nn.GELU):
        super().__init__()
        self.input_resolution = input_resolution
        assert window_size > 0, 'window_size must be greater than 0'
        self.window_size = window_size

        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        head_dim = dim // num_heads

        window_resolution = (window_size, window_size)
        self.attn = Attention(dim, head_dim, num_heads,
                              attn_ratio=1, resolution=window_resolution)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=activation)

        pad = local_conv_size // 2
        self.local_conv = Conv2d_BN(
            dim, dim, ks=local_conv_size, stride=1, pad=pad, groups=dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        res_x = x
        if H == self.window_size and W == self.window_size:
            x = self.attn(x)
        else:
            x = x.view(B, H, W, C)
            pad_b = (self.window_size - H %
                     self.window_size) % self.window_size
            pad_r = (self.window_size - W %
                     self.window_size) % self.window_size
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_size
            nW = pW // self.window_size
            # window partition
            x = x.view(B, nH, self.window_size, nW, self.window_size, C).transpose(2, 3).reshape(
                B * nH * nW, self.window_size * self.window_size, C)
            x = self.attn(x)
            # window reverse
            x = x.view(B, nH, nW, self.window_size, self.window_size,
                       C).transpose(2, 3).reshape(B, pH, pW, C)

            if padding:
                x = x[:, :H, :W].contiguous()

            x = x.view(B, L, C)

        x = res_x + x

        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.local_conv(x)
        x = x.view(B, C, L).transpose(1, 2)

        x = x + self.mlp(x)
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4.,
                 downsample=None, local_conv_size=3, activation=nn.GELU, out_dim=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            TinyViTBlock(dim=dim, input_resolution=input_resolution,
                         num_heads=num_heads, window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         local_conv_size=local_conv_size,
                         activation=activation)
            for _ in range(depth)])
        self.downsample = (None if downsample is None else
                           downsample(input_resolution, dim=dim, out_dim=out_dim,
                                      activation=activation))

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class TinyViT(nn.Module):
    """The encoder: a convolutional stage, three attention stages, and SAM's neck.

    Returns the ``(B, 256, 64, 64)`` image embedding and, as a one-element list, stage 1's output
    as ``(B, 64, 64, 160)`` -- the early feature HQ-SAM's decoder fuses in. Upstream returns it
    in a list because the ViT encoders return one entry per global block; only the first is read.

    The classifier head TinyViT was pretrained with (``norm_head``, ``head``) is in the checkpoint
    and never called; it is not built here, and the loader drops its keys by name.
    """

    def __init__(self, img_size, embed_dims, depths, num_heads, window_sizes, mlp_ratio=4.,
                 mbconv_expand_ratio=4.0, local_conv_size=3, in_chans=3):
        super().__init__()
        activation = nn.GELU

        self.patch_embed = PatchEmbed(in_chans=in_chans,
                                      embed_dim=embed_dims[0],
                                      resolution=img_size,
                                      activation=activation)

        patches_resolution = self.patch_embed.patches_resolution

        num_layers = len(depths)
        self.layers = nn.ModuleList()
        for i_layer in range(num_layers):
            # Upstream's resolution rule, MobileSAM's edit included: stage 3 runs at stage 2's
            # resolution because the merge into it keeps the stride (see PatchMerging).
            kwargs = dict(dim=embed_dims[i_layer],
                          input_resolution=(patches_resolution[0] // (2 ** (i_layer-1 if i_layer == 3 else i_layer)),
                                            patches_resolution[1] // (2 ** (i_layer-1 if i_layer == 3 else i_layer))),
                          depth=depths[i_layer],
                          downsample=PatchMerging if (
                              i_layer < num_layers - 1) else None,
                          out_dim=embed_dims[min(
                              i_layer + 1, len(embed_dims) - 1)],
                          activation=activation,
                          )
            if i_layer == 0:
                layer = ConvLayer(
                    conv_expand_ratio=mbconv_expand_ratio,
                    **kwargs,
                )
            else:
                layer = BasicLayer(
                    num_heads=num_heads[i_layer],
                    window_size=window_sizes[i_layer],
                    mlp_ratio=mlp_ratio,
                    local_conv_size=local_conv_size,
                    **kwargs)
            self.layers.append(layer)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dims[-1],
                256,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(256),
            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(256),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        # x: (N, C, H, W)
        x = self.patch_embed(x)

        x = self.layers[0](x)

        interm_embeddings = []
        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            x = layer(x)
            if i == 1:
                interm_embeddings.append(x.view(x.shape[0], 64, 64, -1))

        B, _, C = x.size()
        x = x.view(B, 64, 64, C)
        x = x.permute(0, 3, 1, 2)
        x = self.neck(x)
        return x, interm_embeddings

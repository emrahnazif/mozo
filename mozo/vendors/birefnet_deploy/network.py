# ------------------------------------------------------------------------
# BiRefNet -- Bilateral Reference for High-Resolution Dichotomous Image Segmentation
# Copyright (c) 2024 ZhengPeng. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
"""BiRefNet: the Swin encoder run twice, a squeeze block, and the four-stage decoder.

Extracted from the ``### models/birefnet.py`` section of upstream's remote-code ``birefnet.py``,
reduced to the eval path of the one configuration every published checkpoint uses. What is gone,
and why it changes nothing:

* ``transformers.PreTrainedModel`` -- the class is an ``nn.Module`` here; the hub mixin only loads.
* ``kornia.filters.laplacian`` -- it computes the gradient label that only training reads.
* The auxiliary classification head, refiner, ender and backbone freezing -- all disabled in
  upstream's config and absent from every checkpoint.
* ``conv_ms_spvn_{4,3,2}`` and ``gdt_convs_pred_{4,3,2}`` -- the multi-scale supervision and
  gradient-prediction heads. Upstream builds them and runs them only under ``self.training``;
  their tensors are in the checkpoints and are dropped when mozo publishes one (named in
  ``PROVENANCE.md``). The gradient-*attention* convolutions, by contrast, run at inference and are
  kept.
* ``einops.rearrange`` -- one fixed pattern, ``b c (hg h) (wg w) -> b (c hg wg) h w``, written
  as a ``view``/``permute``/``reshape`` in :func:`image2patches`.

Upstream returns ``[logits]`` at eval (a list of one, the last of the multi-scale outputs it would
return in training); this returns the tensor.
"""

from __future__ import annotations

__all__ = ["BiRefNet", "Decoder", "image2patches"]

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import BasicDecBlk, BasicLatBlk, SimpleConvs
from .config import ModelSpec
from .swin import SwinTransformer


def _resize(x: torch.Tensor, size) -> torch.Tensor:
    """Every resize in this model: bilinear, ``align_corners=True``."""
    return F.interpolate(x, size=size, mode='bilinear', align_corners=True)


def image2patches(image: torch.Tensor, patch_ref: torch.Tensor) -> torch.Tensor:
    """Fold the image into channels on the grid that makes it *patch_ref*'s size.

    ``einops.rearrange(image, 'b c (hg h) (wg w) -> b (c hg wg) h w', hg=..., wg=...)`` with the grid
    taken from the two shapes, as upstream's ``image2patches(..., patch_ref=...)`` takes it. The
    channel order is ``c`` outermost, then ``hg``, then ``wg`` -- the order the published
    ``ipt_blk*`` weights expect -- and a rearrangement only moves values, so this is exact.
    """
    grid_h = image.shape[-2] // patch_ref.shape[-2]
    grid_w = image.shape[-1] // patch_ref.shape[-1]
    b, c, H, W = image.shape
    h, w = H // grid_h, W // grid_w
    patches = image.view(b, c, grid_h, h, grid_w, w).permute(0, 1, 2, 4, 3, 5)
    return patches.reshape(b, c * grid_h * grid_w, h, w)


def _gdt_convs(channels: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(channels, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(inplace=True))


class Decoder(nn.Module):
    """Four decoder stages, each fed the image again as patches, each gated by gradient attention.

    Args:
        channels: Upstream's ``lateral_channels_in_collection``, deepest first.
    """

    def __init__(self, channels: tuple[int, int, int, int]) -> None:
        super().__init__()
        c0, c1, c2, c3 = channels
        # Upstream: SimpleConvs(2**k * 3, channels[i] // 8). The input width is the image's three
        # channels times the patch grid's area at that scale -- 32x32 at stride 32, down to 1x1.
        self.ipt_blk5 = SimpleConvs(2 ** 10 * 3, c0 // 8)
        self.ipt_blk4 = SimpleConvs(2 ** 8 * 3, c0 // 8)
        self.ipt_blk3 = SimpleConvs(2 ** 6 * 3, c1 // 8)
        self.ipt_blk2 = SimpleConvs(2 ** 4 * 3, c2 // 8)
        self.ipt_blk1 = SimpleConvs(2 ** 0 * 3, c3 // 8)

        self.decoder_block4 = BasicDecBlk(c0 + c0 // 8, c1)
        self.decoder_block3 = BasicDecBlk(c1 + c0 // 8, c2)
        self.decoder_block2 = BasicDecBlk(c2 + c1 // 8, c3)
        self.decoder_block1 = BasicDecBlk(c3 + c2 // 8, c3 // 2)
        self.conv_out1 = nn.Sequential(nn.Conv2d(c3 // 2 + c3 // 8, 1, 1, 1, 0))

        self.lateral_block4 = BasicLatBlk(c1, c1)
        self.lateral_block3 = BasicLatBlk(c2, c2)
        self.lateral_block2 = BasicLatBlk(c3, c3)

        self.gdt_convs_4 = _gdt_convs(c1)
        self.gdt_convs_3 = _gdt_convs(c2)
        self.gdt_convs_2 = _gdt_convs(c3)
        self.gdt_convs_attn_4 = nn.Sequential(nn.Conv2d(16, 1, 1, 1, 0))
        self.gdt_convs_attn_3 = nn.Sequential(nn.Conv2d(16, 1, 1, 1, 0))
        self.gdt_convs_attn_2 = nn.Sequential(nn.Conv2d(16, 1, 1, 1, 0))

    def forward(self, x: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor,
                x4: torch.Tensor) -> torch.Tensor:
        patches = image2patches(x, x4)
        x4 = torch.cat((x4, self.ipt_blk5(_resize(patches, x4.shape[2:]))), 1)
        p4 = self.decoder_block4(x4)
        p4 = p4 * self.gdt_convs_attn_4(self.gdt_convs_4(p4)).sigmoid()
        _p4 = _resize(p4, x3.shape[2:])
        _p3 = _p4 + self.lateral_block4(x3)

        patches = image2patches(x, _p3)
        _p3 = torch.cat((_p3, self.ipt_blk4(_resize(patches, x3.shape[2:]))), 1)
        p3 = self.decoder_block3(_p3)
        p3 = p3 * self.gdt_convs_attn_3(self.gdt_convs_3(p3)).sigmoid()
        _p3 = _resize(p3, x2.shape[2:])
        _p2 = _p3 + self.lateral_block3(x2)

        patches = image2patches(x, _p2)
        _p2 = torch.cat((_p2, self.ipt_blk3(_resize(patches, x2.shape[2:]))), 1)
        p2 = self.decoder_block2(_p2)
        p2 = p2 * self.gdt_convs_attn_2(self.gdt_convs_2(p2)).sigmoid()
        _p2 = _resize(p2, x1.shape[2:])
        _p1 = _p2 + self.lateral_block2(x1)

        patches = image2patches(x, _p1)
        _p1 = torch.cat((_p1, self.ipt_blk2(_resize(patches, x1.shape[2:]))), 1)
        _p1 = self.decoder_block1(_p1)
        _p1 = _resize(_p1, x.shape[2:])

        # A 1x1 grid at full resolution: the image itself. Kept as upstream wrote it, resize to
        # its own size included.
        patches = image2patches(x, _p1)
        _p1 = torch.cat((_p1, self.ipt_blk1(_resize(patches, x.shape[2:]))), 1)
        return self.conv_out1(_p1)


class BiRefNet(nn.Module):
    """One published BiRefNet. ``forward`` takes the normalised image and returns logits.

    Args:
        spec: The checkpoint's geometry, from :data:`~.config.MODEL_SPECS`.
    """

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.spec = spec
        channels = spec.swin.lateral_channels
        context = channels[1:][::-1]  # upstream's cxt: the three shallower levels, shallowest first
        self.bb = SwinTransformer(spec.swin)
        self.squeeze_module = nn.Sequential(BasicDecBlk(channels[0] + sum(context), channels[0]))
        self.decoder = Decoder(channels)

    def forward_enc(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Both backbone passes, concatenated per level, and the context fused into the deepest."""
        x1, x2, x3, x4 = self.bb(x)
        _, _, H, W = x.shape
        x1_, x2_, x3_, x4_ = self.bb(_resize(x, (H // 2, W // 2)))
        x1 = torch.cat([x1, _resize(x1_, x1.shape[2:])], dim=1)
        x2 = torch.cat([x2, _resize(x2_, x2.shape[2:])], dim=1)
        x3 = torch.cat([x3, _resize(x3_, x3.shape[2:])], dim=1)
        x4 = torch.cat([x4, _resize(x4_, x4.shape[2:])], dim=1)
        x4 = torch.cat((_resize(x1, x4.shape[2:]), _resize(x2, x4.shape[2:]),
                        _resize(x3, x4.shape[2:]), x4), dim=1)
        return x1, x2, x3, x4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, 3, S, S)`` normalised -> ``(B, 1, S, S)`` logits. Apply a sigmoid for the matte."""
        x1, x2, x3, x4 = self.forward_enc(x)
        x4 = self.squeeze_module(x4)
        return self.decoder(x, x1, x2, x3, x4)

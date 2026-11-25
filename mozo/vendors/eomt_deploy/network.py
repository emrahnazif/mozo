# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the timm library by Ross Wightman,
# used under the Apache 2.0 License.
# ---------------------------------------------------------------
"""EoMT's network: a timm DINOv2 ViT with learned queries in its last blocks, and two heads.

Extracted from ``tue-mps/eomt`` at ``aa017c3`` (``models/eomt.py``, ``models/vit.py``,
``models/scale_block.py``) and from the timm 1.0.15 modules those build on
(``timm/models/vision_transformer.py``: ``VisionTransformer``, ``Block``, ``Attention``,
``LayerScale``; ``timm/layers``: ``PatchEmbed``, ``Mlp``, ``LayerNorm2d``), reduced to the
inference path the published DINOv2 checkpoints take. The module names are upstream's, so a
checkpoint in upstream's layout loads with ``strict=True``.

**Masked attention is not built.** Upstream's forward predicts masks before each of the last
blocks and uses them as an attention mask, but every published checkpoint carries
``attn_mask_probs = 0``, which replaces every mask with "attend everywhere"
(``_disable_attn_mask``). The authors' notebook therefore builds the network with
``masked_attn_enabled=False``; both settings give ``torch.equal`` final logits (measured on all
three Phase 0 checkpoints). The buffer stays, so the strict load accounts for it.

**Attention is always ``scaled_dot_product_attention``.** timm picks SDPA or a manual softmax
from ``TIMM_FUSED_ATTN`` at import; upstream's own attention (``EoMT._attn``) follows the
module's flag, which is SDPA by default. One path here, whatever the environment says.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MEAN, PATCH, REGISTERS, STD, Spec

__all__ = ["EoMT", "build"]

_EPS = 1e-6  # timm's ``partial(nn.LayerNorm, eps=1e-6)`` and ``LayerNorm2d``'s default


class PatchEmbed(nn.Module):
    """timm's ``PatchEmbed``: one strided convolution, flattened to tokens."""

    def __init__(self, width: int, image_size: int) -> None:
        super().__init__()
        self.patch_size = (PATCH, PATCH)
        self.grid_size = (image_size // PATCH, image_size // PATCH)
        self.proj = nn.Conv2d(3, width, kernel_size=PATCH, stride=PATCH)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    """timm's ``Attention`` (``qkv_bias=True``, no qk-norm), as ``EoMT._attn`` calls it."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.num_heads = heads
        self.head_dim = width // heads
        self.qkv = nn.Linear(width, width * 3)
        self.proj = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, None, 0.0)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class LayerScale(nn.Module):
    """timm's ``LayerScale``, not in place."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class Mlp(nn.Module):
    """timm's ``Mlp``: fc, exact GELU, fc."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, width * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(width * 4, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    """timm's ``Block``. Its forward is ``EoMT.forward``'s, so it lives there."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width, eps=_EPS)
        self.attn = Attention(width, heads)
        self.ls1 = LayerScale(width)
        self.norm2 = nn.LayerNorm(width, eps=_EPS)
        self.mlp = Mlp(width)
        self.ls2 = LayerScale(width)


class Backbone(nn.Module):
    """timm's ``VisionTransformer`` for ``vit_*_patch14_reg4_dinov2`` at patch 16, ``num_classes=0``.

    ``no_embed_class=True``: the position embedding covers the patches only, and is added before
    the class and register tokens are put in front of them.
    """

    def __init__(self, spec: Spec) -> None:
        super().__init__()
        self.num_prefix_tokens = spec.prefix
        self.patch_embed = PatchEmbed(spec.width, spec.image_size)
        self.cls_token = nn.Parameter(torch.empty(1, 1, spec.width))
        self.reg_token = nn.Parameter(torch.empty(1, REGISTERS, spec.width))
        self.pos_embed = nn.Parameter(torch.empty(1, spec.grid * spec.grid, spec.width))
        self.blocks = nn.Sequential(*[Block(spec.width, spec.heads) for _ in range(spec.depth)])
        self.norm = nn.LayerNorm(spec.width, eps=_EPS)

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed
        return torch.cat([self.cls_token.expand(x.shape[0], -1, -1),
                          self.reg_token.expand(x.shape[0], -1, -1), x], dim=1)


class Encoder(nn.Module):
    """``models/vit.py``'s ``ViT``: the backbone and the pixel statistics, as buffers."""

    def __init__(self, spec: Spec) -> None:
        super().__init__()
        self.backbone = Backbone(spec)
        self.register_buffer("pixel_mean", torch.tensor(MEAN).reshape(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(STD).reshape(1, -1, 1, 1))


class LayerNorm2d(nn.LayerNorm):
    """timm's ``LayerNorm2d`` (not the fast-norm path): channels-last LayerNorm on NCHW."""

    def __init__(self, channels: int) -> None:
        super().__init__(channels, eps=_EPS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class ScaleBlock(nn.Module):
    """``models/scale_block.py``: a 2x transposed convolution, GELU, depthwise 3x3, LayerNorm2d."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(width, width, kernel_size=2, stride=2)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1, groups=width, bias=False)
        self.norm = LayerNorm2d(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv2(self.act(self.conv1(x))))


class EoMT(nn.Module):
    """``models/eomt.py``'s ``EoMT``, inference only, last layer only.

    ``forward`` takes pixels already divided by 255 -- the Lightning module's ``forward`` does
    that division, the network the normalisation -- and returns ``(mask_logits, class_logits)``:
    ``(B, queries, 4·grid, 4·grid)`` and ``(B, queries, classes + 1)``, the last class being
    "no object".
    """

    def __init__(self, spec: Spec) -> None:
        super().__init__()
        self.num_q = spec.queries
        self.num_blocks = spec.query_blocks
        self.encoder = Encoder(spec)
        # Training-time masked-attention annealing state; zero in every published checkpoint.
        self.register_buffer("attn_mask_probs", torch.ones(spec.query_blocks))
        self.q = nn.Embedding(spec.queries, spec.width)
        self.class_head = nn.Linear(spec.width, spec.classes + 1)
        self.mask_head = nn.Sequential(
            nn.Linear(spec.width, spec.width), nn.GELU(),
            nn.Linear(spec.width, spec.width), nn.GELU(),
            nn.Linear(spec.width, spec.width),
        )
        num_upscale = max(1, int(math.log2(PATCH)) - 2)
        self.upscale = nn.Sequential(*[ScaleBlock(spec.width) for _ in range(num_upscale)])

    def _predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        backbone = self.encoder.backbone
        q = x[:, : self.num_q, :]
        class_logits = self.class_head(q)
        x = x[:, self.num_q + backbone.num_prefix_tokens:, :]
        x = x.transpose(1, 2).reshape(x.shape[0], -1, *backbone.patch_embed.grid_size)
        mask_logits = torch.einsum("bqc, bchw -> bqhw", self.mask_head(q), self.upscale(x))
        return mask_logits, class_logits

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        backbone = self.encoder.backbone
        x = (x - self.encoder.pixel_mean) / self.encoder.pixel_std
        x = backbone.patch_embed(x)
        x = backbone._pos_embed(x)
        start = len(backbone.blocks) - self.num_blocks
        for i, block in enumerate(backbone.blocks):
            if i == start:
                x = torch.cat((self.q.weight[None, :, :].expand(x.shape[0], -1, -1), x), dim=1)
            x = x + block.ls1(block.attn(block.norm1(x)))
            x = x + block.ls2(block.mlp(block.norm2(x)))
        return self._predict(backbone.norm(x))


def build(spec: Spec) -> EoMT:
    """The network for *spec*, uninitialised."""
    return EoMT(spec)

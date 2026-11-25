# SPDX-License-Identifier: Apache-2.0
"""Per-variant model geometry, replacing upstream's ``build_sam.py``.

Upstream builds each variant by calling ``_build_sam`` (the ViTs) or ``build_sam_vit_t`` (the
TinyViT) with numbers written inline. They are written down here as frozen data, and the strict
load in :mod:`.predictor` is what holds them against the checkpoint: a wrong depth or width leaves
a key with nowhere to go and raises.

Every value was read out of ``segment_anything/build_sam.py`` at the commit recorded in
``PROVENANCE.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["SHARED", "SPECS", "TinySpec", "VitSpec"]


@dataclass(frozen=True)
class VitSpec:
    """A SAM ViT image encoder: the trunk of ``base``, ``large`` and ``huge``.

    Args:
        embed_dim: Token width.
        depth: Transformer blocks.
        num_heads: Attention heads per block.
        global_attn_indexes: The blocks that attend globally rather than within a 14x14 window.
            The first of them is also the early feature HQ-SAM's decoder fuses in.
    """

    embed_dim: int
    depth: int
    num_heads: int
    global_attn_indexes: tuple[int, ...]


@dataclass(frozen=True)
class TinySpec:
    """MobileSAM's TinyViT image encoder: the trunk of ``tiny`` (upstream's "Light HQ-SAM").

    Args:
        embed_dims: Width per stage.
        depths: Blocks per stage. Stage 0 is convolutional (MBConv), the rest attend.
        num_heads: Attention heads per stage; stage 0's entry is unused.
        window_sizes: Attention window per stage; stage 0's entry is unused.
    """

    embed_dims: tuple[int, ...] = (64, 128, 160, 320)
    depths: tuple[int, ...] = (2, 2, 6, 2)
    num_heads: tuple[int, ...] = (2, 4, 5, 10)
    window_sizes: tuple[int, ...] = (7, 7, 14, 7)
    mlp_ratio: float = 4.0
    mbconv_expand_ratio: float = 4.0
    local_conv_size: int = 3

    @property
    def vit_dim(self) -> int:
        """Width of the early feature the HQ decoder fuses: stage 1's output, after its merge."""
        return self.embed_dims[2]


#: What every variant shares. Named rather than inlined so the preprocessing, the prompt scaling
#: and the mask resize all read the same resolution.
SHARED = {
    "image_size": 1024,
    "patch_size": 16,
    "prompt_embed_dim": 256,
    "window_size": 14,
    "mlp_ratio": 4,
    # Pixel statistics on a 0..255 scale: upstream subtracts and divides *before* any rescale,
    # so these are not ImageNet's 0..1 numbers multiplied out -- they are the numbers it uses.
    "pixel_mean": (123.675, 116.28, 103.53),
    "pixel_std": (58.395, 57.12, 57.375),
}

#: Variant name -> geometry. ``tiny`` is the TinyViT; the rest are ViTs.
SPECS: dict[str, VitSpec | TinySpec] = {
    "tiny": TinySpec(),
    "base": VitSpec(embed_dim=768, depth=12, num_heads=12, global_attn_indexes=(2, 5, 8, 11)),
    "large": VitSpec(embed_dim=1024, depth=24, num_heads=16, global_attn_indexes=(5, 11, 17, 23)),
    "huge": VitSpec(embed_dim=1280, depth=32, num_heads=16, global_attn_indexes=(7, 15, 23, 31)),
}

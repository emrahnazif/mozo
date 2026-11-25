# ------------------------------------------------------------------------
# BiRefNet -- Bilateral Reference for High-Resolution Dichotomous Image Segmentation
# Copyright (c) 2024 ZhengPeng. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
"""The geometry of each published BiRefNet, written down rather than inferred.

Upstream keeps it in a ``Config`` class whose every setting is a list indexed by a literal --
``['vgg16', ..., 'swin_v1_l', ...][6]`` -- and that one index is the only difference between the
remote code of the Swin-L repositories and ``BiRefNet_lite``'s (``[3]``, Swin-T). Collecting the
numbers here is what lets the strict load check them (``VENDORING.md`` §5, "geometry that looks
derivable usually is not").

Settings upstream fixes for every published checkpoint, and that this package therefore hardcodes
rather than parameterises: ``mul_scl_ipt='cat'`` (the backbone runs twice, at full and half
resolution, and the two feature pyramids are concatenated), ``cxt_num=3`` (the deepest feature is
concatenated with the three shallower ones), ``squeeze_block='BasicDecBlk_x1'``,
``dec_att='ASPPDeformable'``, ``dec_ipt=True`` with ``dec_ipt_split=True`` (image patches re-enter
the decoder at every scale), ``ms_supervision=True`` with ``out_ref=True`` (the gradient-attention
convolutions run at inference too), ``batch_size=4`` (> 1, so every ``BatchNorm2d`` is real rather
than an ``Identity``), ``dec_channels_inter='fixed'`` (decoder blocks are 64 wide) and
``SDPA_enabled=False`` (eager window attention).
"""

from __future__ import annotations

__all__ = ["BACKBONES", "MEAN", "MODEL_SPECS", "STD", "BackboneSpec", "ModelSpec"]

from dataclasses import dataclass

#: ImageNet statistics, which is what upstream normalises with.
MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class BackboneSpec:
    """One Swin v1, as upstream's ``swin_v1_*`` constructors build it.

    Attributes:
        embed_dim: Width after patch embedding. Stage *i* is ``embed_dim * 2**i``.
        depths: Blocks per stage.
        num_heads: Attention heads per stage.
        window_size: Attention window. 12 for Swin-L, 7 for Swin-T -- and at a 1024 input the
            first stage is 256 tokens a side, which 7 does not divide, so Swin-T pads every stage.
    """

    embed_dim: int
    depths: tuple[int, ...]
    num_heads: tuple[int, ...]
    window_size: int

    @property
    def channels(self) -> tuple[int, ...]:
        """Width of each of the four stage outputs, shallowest first."""
        return tuple(self.embed_dim * 2 ** i for i in range(len(self.depths)))

    @property
    def lateral_channels(self) -> tuple[int, int, int, int]:
        """Upstream's ``lateral_channels_in_collection``: deepest first, doubled by the ``'cat'``
        of the full- and half-resolution passes."""
        return tuple(2 * c for c in reversed(self.channels))  # type: ignore[return-value]


BACKBONES: dict[str, BackboneSpec] = {
    "swin_v1_l": BackboneSpec(embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48),
                              window_size=12),
    "swin_v1_t": BackboneSpec(embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                              window_size=7),
}


@dataclass(frozen=True)
class ModelSpec:
    """One published checkpoint.

    Attributes:
        backbone: Key into :data:`BACKBONES`.
        resolution: The square side the image is resized to. Not a preference: the checkpoints
            were trained at it, and upstream's own ``handler.py`` sets it per checkpoint -- 2048
            for the HR models, 1024 for the rest. It must be a multiple of 32, because the decoder
            folds a ``(resolution / 32)``-square grid of the image into channels and the first
            patch block's width (``2**10 * 3``) assumes exactly 32.
    """

    backbone: str
    resolution: int

    @property
    def swin(self) -> BackboneSpec:
        return BACKBONES[self.backbone]


#: mozo variant -> geometry. The Hugging Face repository each one comes from is a publishing fact
#: and lives in ``tools/fetch/birefnet.py``, not here.
MODEL_SPECS: dict[str, ModelSpec] = {
    "general": ModelSpec("swin_v1_l", 1024),
    "lite": ModelSpec("swin_v1_t", 1024),
    "matting": ModelSpec("swin_v1_l", 1024),
    "hr": ModelSpec("swin_v1_l", 2048),
}

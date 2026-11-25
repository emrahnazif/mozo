# SPDX-License-Identifier: Apache-2.0
"""OmDet-Turbo's geometry, as frozen dataclasses.

Upstream keeps this in ``OmDetTurboConfig`` plus a timm model name and a ``clip_text_model``
config that inherits every number from its class defaults. mozo publishes the checkpoint alone, so
each number is written out here, read off the published ``config.json`` and the defaults it leans
on, and held against the checkpoint by a strict load.

There is one published model. The dataclasses exist anyway so the strict load has something to
check against and a reader has one place to look up a width.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["SPECS", "Spec", "SwinSpec", "TextSpec", "VARIANTS"]


@dataclass(frozen=True)
class SwinSpec:
    """timm's ``swin_tiny_patch4_window7_224``, built at 640 with ``always_partition``.

    Args:
        image_size: The square every image is resized to. The shift masks are computed for this
            size when the model is built, as timm does with ``strict_img_size``, so no other size
            runs.
        patch_size: Side of one patch.
        embed_dim: Width of the first stage; each later stage doubles it.
        depths: Blocks per stage.
        heads: Attention heads per stage.
        window: Window side. 160, 80, 40 and 20 are not multiples of 7, so every stage pads.
        out_stages: Stages whose output the detector reads (strides 8, 16, 32).
    """

    image_size: int = 640
    patch_size: int = 4
    embed_dim: int = 96
    depths: tuple[int, ...] = (2, 2, 6, 2)
    heads: tuple[int, ...] = (3, 6, 12, 24)
    window: int = 7
    mlp_ratio: float = 4.0
    out_stages: tuple[int, ...] = (1, 2, 3)
    layer_norm_eps: float = 1e-5

    @property
    def widths(self) -> tuple[int, ...]:
        return tuple(self.embed_dim * 2**i for i in range(len(self.depths)))


@dataclass(frozen=True)
class TextSpec:
    """CLIP ViT-B/16's text tower, the ``CLIPTextConfig`` defaults.

    Args:
        context_length: Prompt length in tokens, padded with ``<|endoftext|>``.
        projection: Width of ``text_projection``, applied to class prompts only.
    """

    width: int = 512
    layers: int = 12
    heads: int = 8
    intermediate: int = 2048
    vocab_size: int = 49408
    context_length: int = 77
    projection: int = 512
    layer_norm_eps: float = 1e-5


@dataclass(frozen=True)
class Spec:
    """The whole detector. Field names follow ``OmDetTurboConfig``."""

    swin: SwinSpec = SwinSpec()
    text: TextSpec = TextSpec()
    d_model: int = 256
    #: Swin output widths the encoder projects from.
    encoder_in_channels: tuple[int, ...] = (192, 384, 768)
    #: Which projected level the transformer encoder layer runs on (stride 32).
    encoder_projection_index: int = 2
    encoder_heads: int = 8
    encoder_feedforward: int = 2048
    positional_temperature: float = 10000.0
    csp_blocks: int = 3
    decoder_layers: int = 6
    decoder_heads: int = 8
    decoder_points: int = 4
    decoder_feedforward: int = 2048
    levels: int = 3
    queries: int = 900
    class_embed_dim: int = 512
    task_encoder_hidden: int = 1024
    layer_norm_eps: float = 1e-5
    batch_norm_eps: float = 1e-5
    #: Anchor side at the finest level, doubled per level.
    anchor_size: float = 0.05
    #: Anchors closer than this to an edge are invalid.
    anchor_eps: float = 1e-2


#: The variants mozo publishes. Om AI Lab publishes one OmDet-Turbo checkpoint, a Swin-T.
SPECS: dict[str, Spec] = {"tiny": Spec()}

VARIANTS: tuple[str, ...] = tuple(SPECS)

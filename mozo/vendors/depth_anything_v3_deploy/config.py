# ------------------------------------------------------------------------
# Depth Anything 3
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Architecture specifications for the Depth Anything 3 checkpoints this package builds.

Upstream states each model as a YAML file (``configs/da3-small.yaml`` and so on) resolved by
OmegaConf into nested constructor calls; the same numbers are frozen dataclasses here, and the
strict checkpoint load is what holds them to the published tensors.

Three checkpoints, the ones published under Apache-2.0 that predict depth from one image without
a camera: DA3-SMALL and DA3-BASE (upstream's "any-view" models, run on a single view) and
DA3MONO-LARGE (the monocular model). The any-view models alternate local and global attention
from block ``alt_start`` with 2D RoPE and QK-norm, inject a learned camera token there, and
concatenate local and global tokens for the head; the monocular model is a plain DINOv2-L with a
sky head.
"""

from __future__ import annotations

__all__ = ["MODEL_SPECS", "ModelSpec", "get_spec"]

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to build one checkpoint's network.

    Attributes:
        encoder: DINOv2 backbone size, ``vits``, ``vitb`` or ``vitl``.
        out_layers: The four blocks whose outputs feed the head.
        alt_start: First block of alternating local/global attention and of the camera token;
            ``-1`` for none.
        qknorm_start: First block with QK LayerNorm; ``-1`` for none.
        rope_start: First block with 2D rotary position embedding; ``-1`` for none.
        cat_token: Whether the head reads local and global tokens concatenated.
        head_dim_in: Channel width of the tokens reaching the head.
        features: Width of the DPT fusion path.
        out_channels: Per-stage projection widths feeding the four fusion blocks.
        sky: Whether the head has a sky branch -- and with it the sky fill -- (the monocular
            model) rather than a depth-confidence channel (the any-view models).
        process_res: The longest side, in pixels, an image is resized to before the network:
            upstream's ``process_res`` default.
    """

    encoder: Literal["vits", "vitb", "vitl"]
    out_layers: tuple[int, int, int, int]
    alt_start: int
    qknorm_start: int
    rope_start: int
    cat_token: bool
    head_dim_in: int
    features: int
    out_channels: tuple[int, int, int, int]
    sky: bool
    process_res: int = 504


def _any_view(encoder: str, width: int, features: int, out_channels: tuple) -> ModelSpec:
    # configs/da3-small.yaml, da3-base.yaml: the token width doubles because cat_token is on.
    return ModelSpec(encoder=encoder, out_layers=(5, 7, 9, 11), alt_start=4, qknorm_start=4,
                     rope_start=4, cat_token=True, head_dim_in=2 * width, features=features,
                     out_channels=out_channels, sky=False)


#: mozo variant -> specification. Upstream's names: DA3-SMALL, DA3-BASE, DA3MONO-LARGE.
MODEL_SPECS: dict[str, ModelSpec] = {
    "small": _any_view("vits", 384, 64, (48, 96, 192, 384)),
    "base": _any_view("vitb", 768, 128, (96, 192, 384, 768)),
    # configs/da3mono-large.yaml
    "mono-large": ModelSpec(encoder="vitl", out_layers=(4, 11, 17, 23), alt_start=-1,
                            qknorm_start=-1, rope_start=-1, cat_token=False, head_dim_in=1024,
                            features=256, out_channels=(256, 512, 1024, 1024), sky=True),
}


def get_spec(name: str) -> ModelSpec:
    """Look up a variant's specification.

    Raises:
        KeyError: If *name* is not a variant this package builds.
    """
    try:
        return MODEL_SPECS[name]
    except KeyError:
        raise KeyError(
            f"unknown variant {name!r}; variants are {', '.join(MODEL_SPECS)}"
        ) from None

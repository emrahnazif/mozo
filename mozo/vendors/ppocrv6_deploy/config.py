# SPDX-License-Identifier: Apache-2.0
"""Which variant is which: the geometry of each published network, written down.

Every number here is copied from the ``config.json`` PaddlePaddle publishes beside each
``model.safetensors`` (``PaddlePaddle/PP-OCRv6_*_safetensors`` on the Hugging Face Hub). None of
it is derived, and the strict checkpoint load is what holds it against the weights: a wrong width
is a shape mismatch, a missing block is a missing key.
"""

from __future__ import annotations

from dataclasses import dataclass

# (kernel, in_channels, out_channels, stride, squeeze_excitation) per block, per stage.
Blocks = tuple[tuple[tuple[int, int, int, object, bool], ...], ...]


@dataclass(frozen=True)
class Backbone:
    stem_type: str                       # "large" or "small"
    stem_channels: tuple[int, int, int]
    blocks: Blocks
    stem_strides: tuple[int, ...] = (2, 1, 1, 2, 1)
    reduction: int = 4

    @property
    def stage_channels(self) -> tuple[int, ...]:
        return tuple(stage[-1][2] for stage in self.blocks)


@dataclass(frozen=True)
class Detector:
    backbone: Backbone
    neck: str                            # "large_kernel" (medium) or "residual_se" (small, tiny)
    neck_channels: int
    dilated_kernel_size: int = 0         # residual_se only
    kernel_list: tuple[int, int, int] = (3, 2, 2)


@dataclass(frozen=True)
class Recogniser:
    backbone: Backbone
    head: str                            # "svtr" (medium, small) or "ctc_conv" (tiny)
    hidden_size: int
    classes: int                         # the CTC head's width: blank + charset + space
    charset: str                         # file under assets/
    mlp_ratio: float = 0.0               # svtr only
    depth: int = 0                       # svtr only
    heads: int = 8                       # svtr only
    conv_kernel: tuple[int, int] = (1, 7)  # svtr only


@dataclass(frozen=True)
class Spec:
    detector: Detector
    recogniser: Recogniser


def _b(*stages):
    return tuple(tuple(tuple(block) for block in stage) for stage in stages)


_T, _F = True, False

_MEDIUM_DET = Backbone("large", (3, 64, 128), _b(
    [(3, 128, 128, 1, _T), (3, 128, 128, 1, _F)],
    [(3, 128, 256, 2, _F), (3, 256, 256, 1, _T), (3, 256, 256, 1, _F)],
    [(3, 256, 512, 2, _F), (3, 512, 512, 1, _T), (3, 512, 512, 1, _F), (3, 512, 512, 1, _T),
     (3, 512, 512, 1, _F)],
    [(3, 512, 896, 2, _F), (3, 896, 896, 1, _T), (3, 896, 896, 1, _F)],
))

_SMALL_DET = Backbone("large", (3, 24, 48), _b(
    [(3, 48, 48, 1, _T), (3, 48, 48, 1, _F)],
    [(3, 48, 96, 2, _F), (3, 96, 96, 1, _T), (3, 96, 96, 1, _F)],
    [(3, 96, 192, 2, _F), (3, 192, 192, 1, _T), (3, 192, 192, 1, _F), (3, 192, 192, 1, _T),
     (3, 192, 192, 1, _F)],
    [(3, 192, 384, 2, _F), (3, 384, 384, 1, _T), (3, 384, 384, 1, _F)],
))

_TINY_DET = Backbone("large", (3, 16, 32), _b(
    [(3, 32, 32, 1, _T), (3, 32, 32, 1, _F)],
    [(3, 32, 48, 2, _F), (3, 48, 48, 1, _T), (3, 48, 48, 1, _F)],
    [(3, 48, 64, 2, _F), (3, 64, 64, 1, _T), (3, 64, 64, 1, _F), (3, 64, 64, 1, _T),
     (3, 64, 64, 1, _F)],
    [(3, 64, 160, 2, _F), (3, 160, 160, 1, _T), (3, 160, 160, 1, _F)],
))

_H = (2, 1)  # recognisers halve the height and keep the width

_MEDIUM_REC = Backbone("large", (3, 64, 128), _b(
    [(3, 128, 128, 1, _T)],
    [(3, 128, 256, 1, _F), (3, 256, 256, 1, _F), (3, 256, 256, 1, _T)],
    [(3, 256, 512, _H, _F), (3, 512, 512, 1, _T), (3, 512, 512, 1, _F), (3, 512, 512, 1, _T),
     (3, 512, 512, 1, _F), (3, 512, 512, 1, _T), (3, 512, 512, 1, _F)],
    [(3, 512, 768, _H, _F), (3, 768, 768, 1, _T), (3, 768, 768, 1, _F)],
))

_SMALL_REC = Backbone("large", (3, 48, 96), _b(
    [(3, 96, 96, 1, _T)],
    [(3, 96, 96, 1, _F), (3, 96, 96, 1, _F)],
    [(3, 96, 192, _H, _F), (3, 192, 192, 1, _T), (3, 192, 192, 1, _F), (3, 192, 192, 1, _T),
     (3, 192, 192, 1, _F), (3, 192, 192, 1, _T), (3, 192, 192, 1, _F)],
    [(3, 192, 384, _H, _F), (3, 384, 384, 1, _T), (3, 384, 384, 1, _F)],
))

_TINY_REC = Backbone("small", (3, 24, 48), _b(
    [(3, 48, 48, 1, _T)],
    [(3, 48, 48, 1, _F)],
    [(3, 48, 96, _H, _F), (3, 96, 96, 1, _T), (3, 96, 96, 1, _F)],
    [(3, 96, 160, _H, _F), (3, 160, 160, 1, _T), (3, 160, 160, 1, _F), (3, 160, 160, 1, _F)],
))

SPECS: dict[str, Spec] = {
    "medium": Spec(
        Detector(_MEDIUM_DET, "large_kernel", 256),
        Recogniser(_MEDIUM_REC, "svtr", 192, 18710, "ppocrv6.json", mlp_ratio=4.0, depth=2),
    ),
    "small": Spec(
        Detector(_SMALL_DET, "residual_se", 96, dilated_kernel_size=7),
        Recogniser(_SMALL_REC, "svtr", 120, 18710, "ppocrv6.json", mlp_ratio=2.0, depth=2),
    ),
    "tiny": Spec(
        Detector(_TINY_DET, "residual_se", 64, dilated_kernel_size=5),
        Recogniser(_TINY_REC, "ctc_conv", 80, 6906, "ppocrv6-tiny.json"),
    ),
}

VARIANTS: tuple[str, ...] = tuple(SPECS)

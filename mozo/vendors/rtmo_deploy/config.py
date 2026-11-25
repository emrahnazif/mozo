# SPDX-License-Identifier: Apache-2.0
"""RTMO's geometry, as frozen dataclasses.

Upstream spreads these over an MMEngine config per checkpoint (``rtmo-*_body7-*.py``) that
inherits a runtime file and names its widths through two multipliers. mozo publishes the checkpoint
without its config, so every number that decides a tensor shape is written out here, read off the
four body7 configs at MMPose v1.3.2, and ``tests/families/test_rtmo.py`` holds them against a strict
load of each published checkpoint.

What stays derived is derived by upstream's own code -- ``int(channels * widen)`` and
``max(round(blocks * deepen), 1)`` -- reproduced in :mod:`~.network`, because those casts *are* the
architecture: ``int(36 * 0.375)`` is 13, not 13.5 rounded.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["SPECS", "Spec", "get_spec", "KEYPOINTS"]

#: COCO's seventeen joints. Every published head predicts these, in COCO's order.
KEYPOINTS = 17


@dataclass(frozen=True)
class Spec:
    """One published geometry.

    Args:
        input_size: The square side the frame is letterboxed into. 640, or 416 for ``tiny``.
        widen: Upstream's ``widen_factor``, applied to the backbone and the head.
        deepen: Upstream's ``deepen_factor``, applied to the CSP blocks of backbone and neck.
        neck_in: Channels of the three backbone outputs the encoder projects (strides 8/16/32).
        pose_channels: Width of the pose vector each grid cell emits, which is also the
            projector's output width and the DCC's input width.
    """

    input_size: int
    widen: float
    deepen: float
    neck_in: tuple[int, int, int]
    pose_channels: int

    # Shared by every published variant, named so a reader does not have to find them.
    hidden: int = 256            # encoder / FPN width
    heads: int = 8               # encoder attention heads
    ffn: int = 1024              # encoder feed-forward width
    strides: tuple[int, int] = (16, 32)
    groups: int = 8              # group convolutions in the pose branch
    channels_per_group: int = 36
    dcc_channels: int = 128      # DCC keypoint-token width
    bins: tuple[int, int] = (192, 256)  # 1-D heatmap bins, x then y
    spe_channels: int = 128      # sine encoding width for the bins
    spe_temperature: float = 300.0
    gau_s: int = 128
    gau_expansion: int = 2
    bbox_padding: float = 1.25   # box -> DCC window
    nms_threshold: float = 0.65
    pad_value: int = 114


SPECS: dict[str, Spec] = {
    "tiny": Spec(input_size=416, widen=0.375, deepen=0.33, neck_in=(96, 192, 384),
                 pose_channels=192),
    "small": Spec(input_size=640, widen=0.5, deepen=0.33, neck_in=(128, 256, 512),
                  pose_channels=256),
    "medium": Spec(input_size=640, widen=0.75, deepen=0.67, neck_in=(192, 384, 768),
                   pose_channels=384),
    "large": Spec(input_size=640, widen=1.0, deepen=1.0, neck_in=(256, 512, 1024),
                  pose_channels=512),
}


def get_spec(variant: str) -> Spec:
    """Return the geometry for *variant*, naming the known ones when it is not one."""
    try:
        return SPECS[variant]
    except KeyError:
        raise ValueError(f"unknown RTMO variant {variant!r}; known: {sorted(SPECS)}") from None

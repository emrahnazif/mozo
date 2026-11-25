# SPDX-License-Identifier: Apache-2.0
# Derived from libfacedetection.train, Copyright (c) 2022-2026, Shiqi Yu (BSD-3-Clause: see LICENSE).
"""YuNet's two published geometries, as frozen data.

Upstream keeps these as ``YuNetModelConfig`` instances in ``yunet_train/models/config.py``; they
are copied here field for field from commit ``dca340aa`` and held against a strict load of each
published checkpoint by ``tests/families/test_yunet.py``. Only the fields that decide a tensor
shape or a decode are kept: ``num_classes`` is 1 and ``stacked_convs`` is 0 for both, so their
branches are not carried.

Upstream's names are ``yunet_s`` and ``yunet_n``. The ``n`` is "normal" (its training run is
``normal_0.5_1.5``), not "nano": it is the *larger* of the two, which is why mozo calls it
``base`` beside ``small``.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["KEYPOINTS", "SPECS", "Spec", "get_spec"]

#: Five facial landmarks per face, in upstream's channel order: right eye, left eye, nose tip,
#: right mouth corner, left mouth corner (the subject's right: image-left on a frontal face).
KEYPOINTS = 5


@dataclass(frozen=True)
class Spec:
    """One published geometry.

    Args:
        stage_channels: Per backbone stage, its block's channels: ``(in, mid, out)`` for the stem
            (``Conv_head``), ``(in, out)`` for each ``Conv4layerBlock`` after it.
        shared_stacked_convs: ``ConvDPUnit`` layers each head level runs before its four output
            branches. 1 for ``base``, 0 for ``small``.
        upstream: The name upstream gives this geometry.
    """

    stage_channels: tuple[tuple[int, ...], ...]
    shared_stacked_convs: int
    upstream: str

    # Shared by both published variants, named so a reader does not have to find them.
    downsample_idx: tuple[int, ...] = (0, 2, 3, 4)   # stages followed by a 2x2 max pool
    out_idx: tuple[int, ...] = (3, 4, 5)             # stages the neck reads (strides 8/16/32)
    channels: int = 64                               # neck and head width
    strides: tuple[int, ...] = (8, 16, 32)
    size_divisor: int = 32                           # the input is padded to a multiple of this
    nms_threshold: float = 0.45                      # upstream's evaluation setting


SPECS: dict[str, Spec] = {
    "small": Spec(
        stage_channels=((3, 16, 16), (16, 32), (32, 64), (64, 64), (64, 64), (64, 64)),
        shared_stacked_convs=0,
        upstream="yunet_s",
    ),
    "base": Spec(
        stage_channels=((3, 16, 16), (16, 64), (64, 64), (64, 64), (64, 64), (64, 64)),
        shared_stacked_convs=1,
        upstream="yunet_n",
    ),
}


def get_spec(variant: str) -> Spec:
    """The geometry of one published variant.

    Raises:
        ValueError: For a name that is not published, naming the ones that are.
    """
    try:
        return SPECS[variant]
    except KeyError:
        raise ValueError(f"Unknown YuNet variant {variant!r}. Known: {list(SPECS)}") from None

# SPDX-License-Identifier: Apache-2.0
# Derived from libfacedetection.train, Copyright (c) 2022-2026, Shiqi Yu (BSD-3-Clause: see LICENSE).
"""The picture the network sees: the image at its own size, BGR, padded to a multiple of 32.

Upstream's evaluation in "origin" mode (``build_eval_transforms(image_size=None,
size_divisor=32)``): ``Pad`` with 0 at the bottom and right, ``Normalize(mean=0, std=1,
to_rgb=False)`` and ``ToTensor``. Nothing is resized, so a face is seen at the size it has in the
picture -- which is also why faces much larger than the ~300 px the model was trained on are
missed.

Every value is an integer 0..255 in float32 either way, so the order of the three steps does not
change a pixel: here the uint8 image goes to the device first and is flipped, cast and padded
there, which moves a quarter of the bytes a float copy would.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["to_batch"]


def to_batch(image: np.ndarray, size_divisor: int, device: torch.device) -> torch.Tensor:
    """RGB ``HxWx3`` ``uint8`` -> ``(1, 3, H', W')`` BGR float32, ``H'`` and ``W'`` the next
    multiples of *size_divisor*, the padding zero.

    Upstream reads with ``cv2.imread``, so its channels are BGR and the network was trained on
    them; mozo's images are RGB, so the channels are reversed here.
    """
    height, width = image.shape[:2]
    pad_h = -height % size_divisor
    pad_w = -width % size_divisor
    pixels = torch.from_numpy(np.ascontiguousarray(image)).to(device)
    batch = pixels.permute(2, 0, 1).flip(0).unsqueeze(0).float()
    return F.pad(batch, (0, pad_w, 0, pad_h)).contiguous()

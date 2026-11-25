# SPDX-License-Identifier: Apache-2.0
"""Image in, the network's ``(1, 3, 640, 640)`` tensor out.

The authors' pipeline and the reference's PIL processor agree: squash the whole image to 640x640
with **PIL's bilinear filter on the uint8 image** -- aspect not kept, nothing padded -- then
``(x - mean) / std`` in float32 with ImageNet statistics on the 0-255 scale.

PIL's bilinear filter widens with the downscaling factor (an antialiasing filter); OpenCV's and
``F.interpolate``'s do not, and neither does the torchvision backend transformers 5 hands out by
default. The torchvision path is a different picture; see ``PROVENANCE.md``.

Named ``preprocess.py``, not ``image.py``: ``tests/test_vendor_agreement.py`` holds every
vendor's ``image.py`` to the YOLO letterbox interface, which a squash is not.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

__all__ = ["MEAN", "STD", "preprocess"]

MEAN = (123.675, 116.28, 103.53)
STD = (58.395, 57.12, 57.375)


def preprocess(image: np.ndarray, size: int = 640) -> torch.Tensor:
    """``HxWx3`` RGB uint8 -> ``(1, 3, size, size)`` float32, normalised."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 uint8 RGB array, got {image.dtype} {image.shape}")
    resized = np.array(Image.fromarray(image).resize((size, size), Image.Resampling.BILINEAR))
    pixels = resized.astype(np.float32)
    pixels = (pixels - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)))[None]

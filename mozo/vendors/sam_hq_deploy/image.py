# SPDX-License-Identifier: Apache-2.0
"""Resizing, prompt scaling, and the mapping of masks back to the source image.

Images arrive already decoded, as RGB ``uint8`` arrays. Nothing here reads a file: mozo decodes in
one place (:mod:`mozo.image`) so exactly one piece of code decides channel order.

Three things in here differ from SAM 2's preprocessing, and each would run without complaint if
SAM 2's were used instead:

**HQ-SAM letterboxes; SAM 2 squashes.** The long side is scaled to 1024, the short side keeps the
aspect ratio, and the encoder input is zero-padded on the bottom and right to a square. The mask
resize has to crop that padding off before it scales back.

**The resize is PIL's, on ``uint8``.** Upstream's ``ResizeLongestSide.apply_image`` hands the
array to ``torchvision.transforms.functional.resize`` *as a PIL image*, which calls
``Image.resize(..., BILINEAR)``. PIL's bilinear widens its filter with the downscale factor; no
``F.interpolate`` setting reproduces it, and the result is rounded back to eight bits before the
normalise. This module calls Pillow directly, which is the same call without torchvision.

**Normalisation is on the 0..255 scale, after the resize.** ``(x - 123.675) / 58.395``, not
``(x / 255 - 0.485) / 0.229``: the two differ in the last bits of a float.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

__all__ = ["MASK_THRESHOLD", "preprocess_shape", "resize_longest_side", "to_model_coords",
           "to_original"]

#: Above this a mask logit is foreground. Upstream's ``Sam.mask_threshold``.
MASK_THRESHOLD = 0.0


def preprocess_shape(height: int, width: int, long_side: int) -> tuple[int, int]:
    """The ``(h, w)`` an image is resized to: long side ``long_side``, aspect kept.

    Upstream's ``ResizeLongestSide.get_preprocess_shape``, rounding included: ``int(x + 0.5)``
    rounds half up, where Python's ``round`` would round half to even.
    """
    scale = long_side * 1.0 / max(height, width)
    newh, neww = height * scale, width * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)


def resize_longest_side(image: np.ndarray, long_side: int) -> torch.Tensor:
    """Resize an RGB image so its long side is ``long_side``, as the encoder expects it.

    Args:
        image: ``HxWx3`` RGB ``uint8`` array, as :func:`mozo.image.load_image` returns.
        long_side: The encoder's square side.

    Returns:
        The resized image as a ``(1, 3, h, w)`` ``uint8`` tensor, unpadded and unnormalised;
        :meth:`..network.SamHq.encode` does both, where upstream's ``Sam.preprocess`` does.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 RGB array, got shape {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"expected a uint8 array, got {image.dtype}")
    h, w = preprocess_shape(image.shape[0], image.shape[1], long_side)
    resized = np.array(Image.fromarray(np.ascontiguousarray(image)).resize((w, h), Image.BILINEAR))
    return torch.as_tensor(resized).permute(2, 0, 1).contiguous()[None, :, :, :]


def to_model_coords(coords: np.ndarray, shape: tuple[int, int], long_side: int) -> torch.Tensor:
    """Scale x, y prompt coordinates from source pixels into the resized image's.

    Upstream's ``ResizeLongestSide.apply_coords``: in float64, each axis by its own ratio of
    rounded sizes, then cast to float32. Doing it in float32, or with one shared scale factor,
    moves a corner by a last bit -- enough to put a pixel on the other side of the threshold.

    Args:
        coords: ``(..., 2)`` x, y in the source image's pixels.
        shape: The source image's ``(height, width)``.
        long_side: The encoder's square side.

    Returns:
        The same shape as a float32 tensor.
    """
    old_h, old_w = shape
    new_h, new_w = preprocess_shape(old_h, old_w, long_side)
    scaled = np.array(coords, dtype=float)
    scaled[..., 0] = scaled[..., 0] * (new_w / old_w)
    scaled[..., 1] = scaled[..., 1] * (new_h / old_h)
    return torch.as_tensor(scaled, dtype=torch.float)


def to_original(masks: torch.Tensor, input_size: tuple[int, int], shape: tuple[int, int],
                long_side: int) -> torch.Tensor:
    """Resize low-resolution mask logits back to the source image.

    Upstream's ``Sam.postprocess_masks``: up to the padded square, crop the padding, then to the
    source size. Two resizes rather than one, because that is what the weights are evaluated
    under and the two are not the same interpolation.

    Args:
        masks: ``(B, C, 256, 256)`` logits.
        input_size: The resized image's ``(h, w)`` before padding.
        shape: The source image's ``(height, width)``.
        long_side: The encoder's square side.

    Returns:
        ``(B, C, height, width)`` logits; thresholding is the caller's.
    """
    masks = F.interpolate(masks, (long_side, long_side), mode="bilinear", align_corners=False)
    masks = masks[..., : input_size[0], : input_size[1]]
    return F.interpolate(masks, shape, mode="bilinear", align_corners=False)

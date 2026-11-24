# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 the mayaku authors. Extracted from mayaku 3.1.1 (mayaku/inference/preprocess.py,
# mayaku/data/geometry.py).
"""The letterbox onto the square canvas, the network's input tensor, and the inverse for boxes.

Upstream is BGR end to end and reverses the channels while laying the canvas out planar. mozo hands
over RGB, so this letterboxes the RGB image and lays it out without the reversal. The resize is
``cv2.INTER_LINEAR`` applied to each channel alone and the padding is the same grey in all three,
so the planes come out the same as upstream's -- a claim the gate checks rather than assumes.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

#: Mid-grey padding, upstream's ``PAD``.
PAD = 114


def fitted(hw: tuple[int, int], r: float) -> tuple[int, int]:
    """The (h, w) an image of shape *hw* has after scaling by *r*. Python's ``round``: half to
    even, as upstream."""
    return round(hw[0] * r), round(hw[1] * r)


def letterbox(image: np.ndarray, canvas: tuple[int, int]) -> tuple[np.ndarray, dict]:
    """Fit an ``HxWx3`` uint8 image into the (H, W) canvas, centred, grey around it.

    Returns the ``(H, W, 3)`` canvas and the meta ``{"ratio", "pad": (left, top), "shape": (h, w)}``
    every inverse below takes. Written the way upstream's ``_letterbox_into`` writes it -- the
    resize lands straight in its slot of a preallocated canvas -- rather than resize then pad.
    """
    (ch, cw), (h, w) = canvas, image.shape[:2]
    out = np.empty((ch, cw, 3), np.uint8)
    r = min(ch / h, cw / w)
    nh, nw = fitted((h, w), r)
    left, top = (cw - nw) // 2, (ch - nh) // 2
    out[:top] = PAD
    out[top + nh:] = PAD
    out[top:top + nh, :left] = PAD
    out[top:top + nh, left + nw:] = PAD
    place = out[top:top + nh, left:left + nw]
    if (nw, nh) != (w, h):
        cv2.resize(image, (nw, nh), dst=place, interpolation=cv2.INTER_LINEAR)
    else:
        place[...] = image
    return out, {"ratio": r, "pad": (left, top), "shape": (h, w)}


def planar(canvas: np.ndarray) -> torch.Tensor:
    """``(H, W, 3)`` uint8 RGB canvas -> the network's ``(1, 3, H, W)`` float32 in [0, 1].

    Divided by 255, not multiplied by its reciprocal: upstream's ``planar`` divides, and the two
    differ in the last bit.
    """
    h, w, _ = canvas.shape
    out = np.empty((1, 3, h, w), np.float32)
    np.divide(canvas[None].transpose(0, 3, 1, 2), np.float32(255), out=out)
    return torch.from_numpy(out)


def unletterbox(boxes: torch.Tensor, meta: dict) -> torch.Tensor:
    """xyxy boxes in canvas pixels -> the original image's pixels, clipped to it."""
    boxes = boxes.clone()
    ratio, pad, (h, w) = meta["ratio"], meta["pad"], meta["shape"]
    boxes[..., 0::2] = (boxes[..., 0::2] - pad[0]) / ratio
    boxes[..., 1::2] = (boxes[..., 1::2] - pad[1]) / ratio
    boxes[..., 0::2] = boxes[..., 0::2].clip(0, w)
    boxes[..., 1::2] = boxes[..., 1::2].clip(0, h)
    return boxes

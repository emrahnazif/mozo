# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------
"""Pixels in, the tensors the network wants out: upstream's evaluation pre-processing, exactly.

From ``training/lightning_module.py`` at ``aa017c3``: ``scale_img_size_instance_panoptic`` and
``resize_and_pad_imgs_instance_panoptic`` for panoptic checkpoints, ``scale_img_size_semantic``
and ``window_imgs_semantic`` for semantic ones. Both resize the ``uint8`` picture with PIL's
bilinear filter -- on eight bits, before any division -- and both ``round()`` the new size.

**Two different scales.** Panoptic fits the *long* side to the network's square and pads the
bottom and right with zeros; semantic fits the *short* side and slides square windows along the
long one, spreading their overlap evenly.

The tensors stay ``uint8`` here; ``/ 255`` happens where upstream does it, just before the network.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

__all__ = ["pad_size", "resize_and_pad", "window_size", "windows"]


def pad_size(size: tuple[int, int], side: int) -> list[int]:
    """``scale_img_size_instance_panoptic``: (h, w) scaled so the long side fits *side*."""
    factor = min(side / size[0], side / size[1])
    return [round(s * factor) for s in size]


def window_size(size: tuple[int, int], side: int) -> list[int]:
    """``scale_img_size_semantic``: (h, w) scaled so the short side fits *side*."""
    factor = max(side / size[0], side / size[1])
    return [round(s * factor) for s in size]


def _resize(image: np.ndarray, height: int, width: int) -> torch.Tensor:
    """PIL bilinear on the ``uint8`` image, as ``(3, height, width)`` ``uint8``."""
    resized = Image.fromarray(image).resize((width, height), Image.BILINEAR)
    return torch.from_numpy(np.array(resized)).permute(2, 0, 1)


def resize_and_pad(image: np.ndarray, side: int) -> torch.Tensor:
    """``resize_and_pad_imgs_instance_panoptic`` for one ``HxWx3`` RGB ``uint8`` image.

    Returns:
        ``(1, 3, side, side)`` ``uint8``: the resized picture top-left, zeros below and right.
    """
    new_h, new_w = pad_size(image.shape[:2], side)
    resized = _resize(image, new_h, new_w)
    # torchvision's ``pad(img, [0, 0, pad_w, pad_h])`` upstream: zeros, right and bottom.
    padded = F.pad(resized, (0, max(0, side - new_w), 0, max(0, side - new_h)))
    return padded[None]


def windows(image: np.ndarray, side: int) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """``window_imgs_semantic`` for one ``HxWx3`` RGB ``uint8`` image.

    Returns:
        ``(N, 3, side, side)`` ``uint8`` windows along the long side, and each one's
        ``(start, end)`` on that side of the resized picture.
    """
    new_h, new_w = window_size(image.shape[:2], side)
    resized = _resize(image, new_h, new_w)
    long_side = max(resized.shape[-2:])
    num_crops = math.ceil(long_side / side)
    overlap = num_crops * side - long_side
    overlap_per_crop = (overlap / (num_crops - 1)) if overlap > 0 else 0

    crops, origins = [], []
    for j in range(num_crops):
        start = int(j * (side - overlap_per_crop))
        end = start + side
        if resized.shape[-2] > resized.shape[-1]:
            crops.append(resized[:, start:end, :])
        else:
            crops.append(resized[:, :, start:end])
        origins.append((start, end))
    return torch.stack(crops), origins

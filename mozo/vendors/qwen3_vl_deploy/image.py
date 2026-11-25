# SPDX-License-Identifier: Apache-2.0
"""An RGB image to the patches the vision tower reads.

Upstream is ``Qwen2VLImageProcessor`` -- the torchvision backend, which ``AutoProcessor`` picks by
default in ``transformers`` 5. Its PIL backend is a different function (one 8-bit level apart after
a resize, 5.9e-08 apart even without one), so the torchvision calls are reproduced here, not
re-derived:

1. **Resize** to the nearest size whose sides are multiples of 32 (patch 16 x merge 2), kept
   between a floor and a ceiling on the pixel count (:func:`smart_resize`), with torchvision's
   antialiased bicubic on the ``uint8`` tensor -- not PIL's bicubic, and not on floats.
2. **Normalise** with the rescale folded into the statistics: ``(x - 127.5) / 127.5`` computed as
   ``tvF.normalize(x, mean=0.5 * 255, std=0.5 * 255)``, which is not bit-identical to
   ``(x / 255 - 0.5) / 0.5``.
3. **Patchify**: 16x16 patches, ordered by 2x2 merge window rather than raster, each duplicated
   along a time axis of 2 (an image is a one-frame video), flattened to ``3 * 2 * 16 * 16 = 1536``
   values.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torchvision.transforms.v2 import functional as tvF

from .config import VisionSpec

__all__ = ["MAX_PIXELS", "MIN_PIXELS", "PIXEL_CEILING", "preprocess", "smart_resize"]

#: The repository's ``size.shortest_edge``: an image smaller than this many pixels is scaled *up*.
MIN_PIXELS = 65_536

#: The repository's ``size.longest_edge``, 4096 x 4096: the most pixels upstream ever keeps.
PIXEL_CEILING = 16_777_216

#: mozo's default pixel budget: 2,560 merged tokens of 32 x 32 pixels. Upstream defaults to
#: :data:`PIXEL_CEILING`, which lets a phone photograph through at up to 16,384 image tokens --
#: more than a 24 GB GPU holds in fp32. See ``PROVENANCE.md``.
MAX_PIXELS = 2_621_440

# ``Qwen2VLImageProcessor``'s defaults, as ``preprocessor_config.json`` sets them.
_MEAN = (0.5, 0.5, 0.5)
_STD = (0.5, 0.5, 0.5)
_RESCALE = 1 / 255


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int
                 ) -> tuple[int, int]:
    """The size upstream resizes to: sides multiples of *factor*, area within the bounds.

    Copied from ``transformers/models/qwen2_vl/image_processing_qwen2_vl.py``, including the
    ``round`` (not floor) for the common case and the refusal of extreme aspect ratios.

    Raises:
        ValueError: If one side is more than 200 times the other, as upstream does.
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _patchify(images: torch.Tensor, spec: VisionSpec) -> tuple[torch.Tensor, int, int]:
    """``(1, 3, H, W)`` normalised pixels to ``(grid_h * grid_w, 1536)`` patches, merge-window order."""
    patch, merge, temporal = spec.patch, spec.merge, spec.temporal_patch
    batch, channel, height, width = images.shape
    grid_h, grid_w = height // patch, width // patch
    patches = images.reshape(batch, channel, grid_h // merge, merge, patch, grid_w // merge, merge, patch)
    patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
    flat = (patches.unsqueeze(6)
            .expand(-1, -1, -1, -1, -1, -1, temporal, -1, -1)
            .reshape(batch, grid_h * grid_w, channel * temporal * patch * patch))
    return flat, grid_h, grid_w


def preprocess(rgb: np.ndarray, spec: VisionSpec, max_pixels: int = MAX_PIXELS
               ) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """An ``HxWx3`` RGB ``uint8`` array to ``(pixel_values, (t, grid_h, grid_w))``.

    ``pixel_values`` is ``(grid_h * grid_w, 1536)`` float32 on the CPU; the grid counts 16-pixel
    patches, so the image becomes ``grid_h * grid_w / 4`` tokens.
    """
    image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).contiguous()[None]
    height, width = image.shape[-2:]
    size = smart_resize(height, width, spec.factor, MIN_PIXELS, max_pixels)
    image = tvF.resize(image, list(size), interpolation=tvF.InterpolationMode.BICUBIC, antialias=True)
    # The fold upstream makes (``_fuse_mean_std_and_rescale_factor``), operation for operation.
    mean = torch.tensor(_MEAN) * (1.0 / _RESCALE)
    std = torch.tensor(_STD) * (1.0 / _RESCALE)
    image = tvF.normalize(image.to(torch.float32), mean, std, inplace=False)
    flat, grid_h, grid_w = _patchify(image, spec)
    return flat[0], (1, grid_h, grid_w)

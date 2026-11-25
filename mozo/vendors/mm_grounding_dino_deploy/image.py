# SPDX-License-Identifier: Apache-2.0
"""Preprocessing: resize the short side to 800, cap the long side at 1333, normalise.

This is ``transformers``' ``GroundingDinoImageProcessor`` -- the slow, NumPy one, which is what
``AutoProcessor`` returns for these checkpoints -- reduced to the single-image path. Nothing is
letterboxed: the aspect ratio is kept and nothing is padded, so the tensor a photograph becomes
depends on the photograph.

Three details are where a restatement goes wrong, and each moves pixels:

- **The resize goes through PIL**, bilinear, on the ``uint8`` image. PIL's filter has support
  that scales with the downsampling factor; neither ``cv2.resize`` nor
  ``torch.nn.functional.interpolate`` reproduces it.
- **The rescale is ``float64`` and then cast down**: ``(x.astype(float64) * (1/255))`` becomes
  ``float32`` before the mean is subtracted in ``float32``. Folding the ``1/255`` into the
  statistics, or doing it in ``float32``, is a different rounding.
- **The size rounds differently from Grounding DINO's.** When the 1333 cap binds, this processor
  scales the long side from the *unrounded* short side, so a 500x4000 image becomes 167x1333.
  The authors' Grounding DINO code (and mozo's ``grounding_dino``) rounds first and returns
  167x1336. Same idea, different pixels; this module follows the processor the checkpoint ships.

:func:`fit_square` is mozo's rather than upstream's: scaled to fit a fixed square and padded, with
a mask saying which pixels are padding, because a compiled graph runs at one shape. The model
reads the mask as it reads the padding of a training batch, so it attends only to the picture --
but the picture is smaller than the resize would make it, so the answers differ.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch import Tensor

__all__ = ["IMAGENET_MEAN", "IMAGENET_STD", "fit_square", "fitted_size", "preprocess",
           "resized_size"]

#: ImageNet statistics, in RGB order, as the checkpoints' ``preprocessor_config.json`` states.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: ``rescale_factor`` from the same file: 1/255 as a Python float, applied in float64.
_RESCALE = 0.00392156862745098


def resized_size(height: int, width: int, short_side: int, max_side: int) -> tuple[int, int]:
    """Return the ``(height, width)`` the image is resized to.

    Reproduces ``get_size_with_aspect_ratio`` from ``transformers``' Grounding DINO processor,
    including its early return when the image is already the right size, and its use of the
    unrounded short side for the long side when the cap binds. See the module docstring.
    """
    raw = None
    size = short_side
    smallest = float(min(height, width))
    largest = float(max(height, width))
    if largest / smallest * size > max_side:
        raw = max_side * smallest / largest
        size = int(round(raw))

    if (height <= width and height == size) or (width <= height and width == size):
        return height, width
    if width < height:
        return int((raw if raw is not None else size) * height / width), size
    return size, int((raw if raw is not None else size) * width / height)


def fitted_size(height: int, width: int, size: int) -> tuple[int, int]:
    """Return the ``(height, width)`` an image is scaled to so it fits a *size* square.

    The long side becomes exactly *size* and the short side keeps the aspect ratio, rounded.
    """
    if height >= width:
        return size, max(1, round(width * size / height))
    return max(1, round(height * size / width)), size


def _normalised(image: np.ndarray, height: int, width: int) -> Tensor:
    """Resize *image* to ``(height, width)`` through PIL and normalise it, as upstream does.

    Returns ``(3, height, width)`` float32.
    """
    resized = np.array(Image.fromarray(image).resize((width, height), Image.BILINEAR))
    scaled = (resized.astype(np.float64) * _RESCALE).astype(np.float32)
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(((scaled - mean) / std).transpose(2, 0, 1)))


def preprocess(image: np.ndarray, short_side: int, max_side: int) -> Tensor:
    """Turn an ``HxWx3`` RGB ``uint8`` array into the ``(3, h, w)`` tensor the model wants.

    Args:
        image: RGB ``uint8``, mozo's contract.
        short_side: Target for the shorter side. From :class:`~.config.Spec`, which owns it.
        max_side: Ceiling for the longer side. Likewise.

    Returns:
        A normalised float32 tensor, no batch dimension.
    """
    return _normalised(image, *resized_size(*image.shape[:2], short_side, max_side))


def fit_square(image: np.ndarray, size: int) -> tuple[Tensor, Tensor]:
    """Scale an image to fit a *size* square, pad the rest, and say which pixels are padding.

    Args:
        image: RGB ``uint8``, mozo's contract.
        size: Side of the square.

    Returns:
        ``(3, size, size)`` normalised float32, the picture at the top left and zeros after it --
        zeros in the normalised space, which is what upstream's batch padding fills -- and a
        ``(size, size)`` boolean mask, True where a pixel is padding.
    """
    height, width = fitted_size(*image.shape[:2], size)
    canvas = torch.zeros((3, size, size), dtype=torch.float32)
    canvas[:, :height, :width] = _normalised(image, height, width)
    padding = torch.ones((size, size), dtype=torch.bool)
    padding[:height, :width] = False
    return canvas, padding

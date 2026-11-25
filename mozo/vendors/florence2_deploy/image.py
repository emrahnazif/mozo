# SPDX-License-Identifier: MIT
"""Florence-2's preprocessing: upstream's *slow* ``CLIPImageProcessor``, reproduced.

Microsoft's ``preprocessor_config.json`` names ``CLIPImageProcessor`` and the remote code runs the
PIL implementation (``transformers`` 4.x): resize to 768 x 768 with PIL's bicubic filter, the aspect
ratio *not* kept and nothing cropped; multiply by 1/255 in float64 and round to float32; subtract
ImageNet's mean and divide by its standard deviation in float32; channels first.

``transformers`` 5's default ``CLIPImageProcessor`` is a torchvision implementation whose pixels
differ by up to one 8-bit level (Phase 0: 67% of values, 0.0175 after normalising) -- enough to move
a box by one bin. This is the PIL one.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from .config import IMAGE_SIZE

__all__ = ["MEAN", "STD", "preprocess"]

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def preprocess(image: np.ndarray) -> torch.Tensor:
    """An ``HxWx3`` RGB ``uint8`` array to ``(1, 3, 768, 768)`` float32 pixels."""
    resized = Image.fromarray(image).resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.BICUBIC,
                                            reducing_gap=None)
    x = (np.array(resized).astype(np.float64) * (1 / 255)).astype(np.float32)
    x = (x - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]

# SPDX-License-Identifier: Apache-2.0
"""Pixels in, the tensor the backbone wants out: upstream's evaluation transform, exactly.

``make_classification_eval_transform`` in ``dinov2/data/transforms.py``: ``Resize(256, BICUBIC)``
on a PIL image, ``CenterCrop(224)``, ``ToTensor``, ``Normalize`` with ImageNet's statistics. The
same torchvision functions those transforms call, on the same PIL image, so nothing is
re-derived.

**Use torchvision's crop, never a slice.** It places the crop at ``int(round((w - 224) / 2))``;
``transformers``' image processor uses ``(w - 224) // 2``, which is a pixel to the left whenever
the margin is odd -- 2.8 apart on a normalised tensor for a 1920x1281 photograph. The resize
scales the short side, so only the middle of a long side is ever seen: 49% of a 16:9 frame's
width.
"""

from __future__ import annotations

import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from torch import Tensor

from .config import CROP, MEAN, RESIZE, STD

__all__ = ["preprocess"]


def preprocess(image: np.ndarray) -> Tensor:
    """Turn an ``HxWx3`` RGB ``uint8`` array into the ``(3, 224, 224)`` float32 the backbone wants."""
    resized = TF.resize(Image.fromarray(image), RESIZE, interpolation=TF.InterpolationMode.BICUBIC)
    cropped = TF.center_crop(resized, CROP)
    return TF.normalize(TF.to_tensor(cropped), mean=list(MEAN), std=list(STD))

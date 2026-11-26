# ------------------------------------------------------------------------
# BiRefNet -- Bilateral Reference for High-Resolution Dichotomous Image Segmentation
# Copyright (c) 2024 ZhengPeng. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
#
# refine_foreground() is Photoroom's blur-fusion foreground estimator, which
# upstream's handler.py credits to https://github.com/Photoroom/fast-foreground-estimation
# ------------------------------------------------------------------------
"""Pixels in, matte out: everything either side of the network.

Reproduces the inference snippet on every BiRefNet model card and the repository's ``handler.py``,
which agree step for step::

    transforms.Compose([transforms.Resize((S, S)), transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    preds = birefnet(input_images)[-1].sigmoid()
    mask = transforms.ToPILImage()(preds[0].squeeze()).resize(image.size)

Written out here so this package needs neither torchvision's transforms nor its PIL conversion:

* **``Resize((S, S))`` on a PIL image is PIL's BILINEAR**, which torchvision hands straight to
  ``Image.resize``. PIL's filter support widens with the downscale factor -- it antialiases -- and
  neither ``cv2.resize`` nor ``F.interpolate`` reproduces it, so Pillow stays.
* **The resize squashes.** Aspect ratio is not kept; the checkpoints were trained square.
* **``ToPILImage`` truncates.** A float tensor becomes ``mul(255).byte()``: 0.999 -> 254.
* **The resize back is Pillow's default filter for a mode-``L`` image, BICUBIC**, applied to
  the 8-bit matte -- not a bilinear resize of the float sigmoid. Named explicitly here, because
  Pillow's default is a version-dependent fact (it was NEAREST before Pillow 7).
"""

from __future__ import annotations

__all__ = ["postprocess", "preprocess", "refine_foreground"]

import cv2
import numpy as np
import torch
from PIL import Image

from .config import MEAN, STD


def preprocess(rgb: np.ndarray, resolution: int, device: str | torch.device = "cpu",
               dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """``(H, W, 3)`` uint8 RGB -> ``(1, 3, S, S)`` normalised.

    Args:
        rgb: ``(H, W, 3)`` uint8, already RGB. ``handler.py`` converts with ``.convert('RGB')``
            first; the model card's snippet does not, and fails on a four-channel PNG.
        resolution: The checkpoint's square side, from :data:`~.config.MODEL_SPECS`.
        device: Where to put the tensor.
        dtype: Compute dtype. Parity is claimed for ``float32`` only.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"expected an (H, W, 3) uint8 RGB array, got {rgb.shape} {rgb.dtype}")

    square = Image.fromarray(rgb).resize((resolution, resolution), resample=Image.BILINEAR)
    # torchvision's ToTensor, then Normalize, written out. np.array, not np.asarray: PIL hands
    # back a read-only buffer and torch.from_numpy warns on one.
    tensor = torch.from_numpy(np.array(square)).permute(2, 0, 1).contiguous().to(torch.float32).div(255)
    mean = torch.tensor(MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(STD, dtype=torch.float32).view(3, 1, 1)
    tensor = tensor.sub(mean).div(std)
    return tensor.unsqueeze(0).to(device=device, dtype=dtype)


def postprocess(logits: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    """``(1, 1, S, S)`` logits -> ``(H, W)`` uint8 alpha at *size* ``(height, width)``.

    The sigmoid is a per-pixel probability of foreground; it is not stretched or renormalised, so
    255 means "certainly foreground" in every image (unlike BEN2's default). For the segmentation
    checkpoints it is close to binary; for ``matting`` it is a soft alpha.
    """
    # Upstream's order and dtype: the sigmoid on the logits' own device and in their own dtype
    # (float16 under autocast or ``.half()``), then ToPILImage's ``mul(255).byte()`` on the host.
    # A float32 sigmoid of float16 logits moves pixels by a grey level.
    small = logits.sigmoid().cpu()[0].squeeze()
    small = small.mul(255).byte().numpy()
    height, width = size
    return np.asarray(Image.fromarray(small, mode="L").resize((width, height), resample=Image.BICUBIC))


def _blur_fusion(image: np.ndarray, foreground: np.ndarray, background: np.ndarray,
                 alpha: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray]:
    """One pass of Photoroom's blur-fusion foreground estimator. Float arrays in ``[0, 1]``."""
    blurred_alpha = cv2.blur(alpha, (r, r))[:, :, None]
    blurred_fa = cv2.blur(foreground * alpha, (r, r))
    blurred_f = blurred_fa / (blurred_alpha + 1e-5)
    blurred_b1a = cv2.blur(background * (1 - alpha), (r, r))
    blurred_b = blurred_b1a / ((1 - blurred_alpha) + 1e-5)
    estimated = blurred_f + alpha * (image - alpha * blurred_f - (1 - alpha) * blurred_b)
    return np.clip(estimated, 0, 1), blurred_b


def refine_foreground(rgb: np.ndarray, alpha: np.ndarray, r: int = 90) -> np.ndarray:
    """Estimate unmixed foreground colours. ``(H, W, 3)`` uint8 and ``(H, W)`` uint8 in.

    ``handler.py``'s ``refine_foreground``: two blur-fusion passes, ``r=90`` then ``r=6``, in
    float64, truncated back to uint8. **It changes RGB, not alpha** -- along a soft edge it removes
    the fringe of the old background a composite would otherwise carry.
    """
    image = rgb / 255.0
    a = (alpha / 255.0)[:, :, None]
    foreground, blurred_b = _blur_fusion(image, image, image, a, r)
    foreground, _ = _blur_fusion(image, foreground, blurred_b, a, 6)
    return (foreground * 255.0).astype(np.uint8)

# SPDX-License-Identifier: Apache-2.0
"""Preprocessing for SAM 3.

Images arrive already decoded, as RGB ``uint8`` arrays -- mozo decodes in one place
(:mod:`mozo.image`) so exactly one piece of code decides channel order.

Five things here are easy to get wrong quietly.

**SAM 3 squashes, it does not letterbox.** The image goes straight to 1008x1008, distorting the
aspect ratio, with no padding to undo afterwards. Boxes and masks scale back by independent x and
y factors.

**The resize happens in uint8, not in float.** Upstream's transform is ``ToDtype(uint8) ->
Resize -> ToDtype(float32, scale=True) -> Normalize``, so the interpolation result is rounded
back to 8 bits *before* being divided by 255. Resizing in float instead -- the obvious way, and
what SAM 2 does -- shifts roughly every second pixel by 1/255. Reproduced here exactly, verified
bit-for-bit against ``torchvision.transforms.v2.Resize``.

**The resize antialiases.** ``v2.Resize`` defaults to ``antialias=True`` on tensors;
``cv2.INTER_LINEAR`` does not. On this image the difference reaches 72 grey levels.

**Normalisation is to [-1, 1], not ImageNet.** Mean and standard deviation are 0.5 on every
channel. Reaching for the ImageNet constants every other family uses is wrong here.

**Both heads read this one preprocessing.** Meta's own image path -- ``Sam3Processor.set_image``
then ``predict_inst`` -- runs the trunk once, on this transform, and feeds the concept head and
the click head from that single encode. ``transformers``' ``sam3_tracker``, which the click head
was first matched against, prepares the image the way SAM 2 does instead (resize in float, divide
by 255), and so needs a second encode of its own. The two disagree by half a grey level, which is
enough to move a click mask by hundreds of pixels; measured against Meta's code on 15 photographs
and 225 masks, this transform is bit-identical and the other is not.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .config import SPEC, Spec

__all__ = ["fill_holes", "preprocess", "to_model_coords", "to_original"]


def preprocess(image: np.ndarray, spec: Spec = SPEC) -> torch.Tensor:
    """Scale an RGB image to the encoder's square and normalise it.

    Args:
        image: ``HxWx3`` RGB ``uint8`` array, as :func:`mozo.image.load_image` returns.
        spec: Vision geometry, for the resolution and the normalisation constants.

    Returns:
        The ``(1, 3, size, size)`` normalised float batch.

    Raises:
        ValueError: If the array is not ``HxWx3``.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 RGB array, got shape {image.shape}")
    size = spec.trunk.image_size

    chw = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
    # Interpolate in float, then round back to uint8 -- see the module docstring. ``round`` and
    # not truncation: truncating moves 1.5 million pixels on a single 1920x1281 photograph.
    resized = F.interpolate(
        chw[None].float(), size=(size, size), mode="bilinear", align_corners=False, antialias=True
    )
    resized = resized.round_().clamp_(0, 255).to(torch.uint8)

    # Multiply by the reciprocal, do not divide by 255. ``torchvision``'s uint8-to-float
    # conversion is ``image.to(dtype).mul_(1.0 / 255)``, and 1/255 is not exactly representable,
    # so the two disagree in the last bit on 40 percent of pixels.
    batch = resized.float().mul_(1.0 / 255.0)
    mean = batch.new_tensor(spec.mean).view(1, 3, 1, 1)
    std = batch.new_tensor(spec.std).view(1, 3, 1, 1)
    return batch.sub_(mean).div_(std)



def to_model_coords(
    coords: np.ndarray, shape: tuple[int, int], spec: Spec = SPEC
) -> torch.Tensor:
    """Scale x, y prompt coordinates from source pixels into the encoder's square.

    The squash makes this simpler than it is for a letterboxing model, not harder: x and y each
    scale by their own factor and there is no padding offset to subtract. It is the same mapping
    :func:`preprocess` applies to the pixels, so a click lands on the feature it was
    pointing at.

    Args:
        coords: ``(..., 2)`` x, y in the source image's pixels.
        shape: The source image's ``(height, width)``.
        spec: Vision geometry, for the square's side.

    Returns:
        The same shape, as a float tensor in ``[0, size]``.
    """
    height, width = shape
    # ``torch.tensor`` copies, which is what keeps the in-place divides below off the caller's
    # array -- ``as_tensor`` would share memory with a float32 input and scale it under them.
    scaled = torch.tensor(np.asarray(coords, dtype=np.float32))
    # Normalise then scale, rather than multiplying by the combined ratio. The two differ in the
    # last bits of a float, which is enough to move a click across a pixel boundary.
    scaled[..., 0] /= width
    scaled[..., 1] /= height
    return scaled * spec.trunk.image_size


def to_original(masks: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Resize mask logits back to the source image.

    Args:
        masks: ``(B, C, h, w)`` logits.
        shape: The source image's ``(height, width)``.

    Returns:
        ``(B, C, height, width)`` logits. Thresholding stays the caller's decision.
    """
    return F.interpolate(masks.float(), shape, mode="bilinear", align_corners=False)


def fill_holes(logits: torch.Tensor, threshold: float, max_area: int) -> torch.Tensor:
    """Turn small background regions of low-resolution mask logits into foreground.

    What Meta's interactive predictor does to every click mask before resizing it
    (``SAM2Transforms.postprocess_masks`` with ``max_hole_area=256``): each connected background
    region -- logits at or below ``threshold``, joined through all eight neighbours -- of at most
    ``max_area`` pixels is set just above the threshold, ``threshold + 10``. That includes a small
    region touching the border, not only a hole enclosed by the mask; it is Meta's rule, kept as it
    is. Upstream labels with a CUDA kernel or scikit-image; OpenCV gives the same labels.

    Args:
        logits: ``(B, C, h, w)`` mask logits, on any device.
        threshold: At or below this a logit is background.
        max_area: The largest region, in low-resolution pixels, that counts as a hole.

    Returns:
        A new tensor with the holes filled. ``logits`` is not changed: the unfilled logits are what
        a caller feeds back as the next prompt's mask, as upstream does.
    """
    background = (logits <= threshold).to(torch.uint8).flatten(0, 1).cpu().numpy()
    holes = np.zeros(background.shape, dtype=bool)
    for index, plane in enumerate(background):
        _, labels, stats, _ = cv2.connectedComponentsWithStats(plane, connectivity=8)
        small = stats[:, cv2.CC_STAT_AREA] <= max_area
        small[0] = False  # label 0 is the foreground, which OpenCV counts as a region of its own
        holes[index] = small[labels]
    holes = torch.from_numpy(holes).reshape(logits.shape).to(logits.device)
    return logits.masked_fill(holes, threshold + 10.0)

# ------------------------------------------------------------------------
# Depth Anything 3
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Pre-processing, forward pass and post-processing for one image.

Upstream spreads these over ``api.DepthAnything3.inference``, ``utils/io/input_processor.py``
(run through a process pool) and ``utils/io/output_processor.py``. For one image with upstream's
defaults (``process_res=504``, ``upper_bound_resize``) they come to:

1. The longest side resized to 504 with ``cv2.resize`` -- ``INTER_CUBIC`` when enlarging,
   ``INTER_AREA`` when shrinking, skipped when it is already 504.
2. Each side resized again to its nearest multiple of 14 (ties go up), with the same rule.
3. ``ToTensor`` and ImageNet ``Normalize``.
4. The network, then -- for a model with a sky head -- the sky fill.

The depth comes back at that processed resolution, as upstream returns it. Resizing it to the
input's size is the caller's business (:func:`resize`).
"""

from __future__ import annotations

__all__ = ["Predictor", "postprocess", "preprocess", "processed_size", "resize"]

import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision.transforms.functional as TF  # noqa: N812

from .config import ModelSpec, get_spec
from .net import DepthAnything3Net, fill_sky

# ImageNet statistics, as upstream's InputProcessor.NORMALIZE applies them.
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]
PATCH_SIZE = 14

#: The seed of the generator the sky fill samples from, fresh on every call so the same image
#: always gives the same depth. Upstream samples from torch's global RNG instead.
SKY_SEED = 0


def _resize_longest_side(img: np.ndarray, target_size: int) -> np.ndarray:
    # InputProcessor._resize_longest_side, on the array its PIL image wraps.
    h, w = img.shape[:2]
    longest = max(w, h)
    if longest == target_size:
        return img
    scale = target_size / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(img, (new_w, new_h), interpolation=interpolation)


def _nearest_multiple(x: int, p: int) -> int:
    down = (x // p) * p
    up = down + p
    return up if abs(up - x) <= abs(x - down) else down


def _make_divisible_by_resize(img: np.ndarray, patch: int) -> np.ndarray:
    # InputProcessor._make_divisible_by_resize.
    h, w = img.shape[:2]
    new_w = max(1, _nearest_multiple(w, patch))
    new_h = max(1, _nearest_multiple(h, patch))
    if new_w == w and new_h == h:
        return img
    upscale = (new_w > w) or (new_h > h)
    interpolation = cv2.INTER_CUBIC if upscale else cv2.INTER_AREA
    return cv2.resize(img, (new_w, new_h), interpolation=interpolation)


def processed_size(height: int, width: int, process_res: int = 504) -> tuple[int, int]:
    """The ``(h, w)`` an image of this size reaches the network at, without resizing anything."""
    longest = max(width, height)
    if longest != process_res:
        scale = process_res / float(longest)
        width, height = max(1, int(round(width * scale))), max(1, int(round(height * scale)))
    return (max(1, _nearest_multiple(height, PATCH_SIZE)),
            max(1, _nearest_multiple(width, PATCH_SIZE)))


def resize(depth: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Resize an ``HxW`` depth map to *size*, bilinearly with corners aligned.

    Not upstream's: Depth Anything 3 returns depth at the processed resolution. This is the
    resize Depth Anything V2 applies on the way out, so mozo's two families agree.
    """
    return F.interpolate(depth[None, None], size, mode="bilinear", align_corners=True)[0, 0]


def preprocess(image: np.ndarray, process_res: int = 504) -> torch.Tensor:
    """Turn one ``HxWx3`` uint8 RGB image into the ``1x1x3xH'xW'`` network input, on the CPU.

    Raises:
        ValueError: If the image is so elongated that its short side would round to nothing
            (an aspect ratio beyond 72:1), which upstream fails on with an assertion.
    """
    h, w = processed_size(*image.shape[:2], process_res)
    if h % PATCH_SIZE or w % PATCH_SIZE:
        raise ValueError(
            f"a {image.shape[1]}x{image.shape[0]} image is too elongated: its short side "
            f"would be {min(h, w)} px at a {process_res} px long side, and the network needs "
            f"at least {PATCH_SIZE}"
        )
    # Upstream wraps the array in a PIL image, which copies it; a view with negative strides
    # (``bgr[..., ::-1]``) would otherwise reach cv2.resize.
    image = np.ascontiguousarray(image)
    resized = _make_divisible_by_resize(_resize_longest_side(image, process_res), PATCH_SIZE)
    tensor = TF.normalize(TF.to_tensor(resized), mean=_MEAN, std=_STD)
    return tensor[None, None].float()


def postprocess(outputs: dict[str, torch.Tensor], sky: bool) -> torch.Tensor:
    """Upstream's post-process on the network's outputs: the depth, sky filled if it has one.

    Returns:
        The ``H'xW'`` depth at the processed resolution.
    """
    depth = outputs["depth"]
    if sky:
        generator = torch.Generator(device=depth.device).manual_seed(SKY_SEED)
        depth = fill_sky(depth, outputs["sky"], generator)
    return depth[0, 0]


class Predictor:
    """One loaded Depth Anything 3 checkpoint, ready to run on one image at a time.

    Attributes:
        spec: The variant's architecture specification.
        model: The :class:`~.net.DepthAnything3Net`, in eval mode.
    """

    def __init__(self, model: DepthAnything3Net, spec: ModelSpec) -> None:
        self.model = model.eval()
        self.spec = spec

    @classmethod
    def from_pretrained(
        cls,
        name: str,
        *,
        weights: str | os.PathLike[str],
        device: str | torch.device = "cpu",
    ) -> Predictor:
        """Build variant *name*, load its checkpoint strictly, and return a ready predictor.

        Raises:
            KeyError: If *name* is not a variant this package builds.
            RuntimeError: If the checkpoint does not match the architecture.
        """
        spec = get_spec(name)
        model = DepthAnything3Net(spec)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        # strict=True on purpose: upstream loads through PyTorchModelHubMixin, which is not
        # strict, and a tensor left at its initialisation is a silent bug.
        model.load_state_dict(state, strict=True)
        return cls(model.to(device), spec)

    @property
    def device(self) -> torch.device:
        """Where the model's parameters live."""
        return next(self.model.parameters()).device

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """:func:`preprocess` at this variant's processing size."""
        return preprocess(image, self.spec.process_res)

    def postprocess(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """:func:`postprocess` for this variant."""
        return postprocess(outputs, self.spec.sky)

    @torch.inference_mode()
    def predict(self, image: np.ndarray) -> torch.Tensor:
        """Estimate depth for one ``HxWx3`` uint8 RGB image.

        Returns:
            The ``H'xW'`` float32 depth at the processed resolution, on this predictor's device:
            larger is farther, on a scale of the model's choosing.
        """
        tensor = self.preprocess(image).to(self.device)
        return self.postprocess(self.model(tensor))

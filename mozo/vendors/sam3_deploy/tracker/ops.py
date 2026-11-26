# SPDX-License-Identifier: Apache-2.0
"""The small operations around the networks: frame preprocessing, mask IoU, NMS, hole filling.

None of this is in ``transformers``' video port as Meta runs it: there, NMS and hole filling come
from a compiled kernel downloaded from the Hub at run time, and are silently skipped without it.
These are this package's own code, written to the rule upstream states and checked against Meta's
CPU path (``PROVENANCE.md``).
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image

from ..config import SPEC, Spec

__all__ = ["component_areas", "fill_holes", "load_click_frame", "load_frame", "mask_iou", "nms"]


def load_frame(image: np.ndarray, spec: Spec = SPEC) -> torch.Tensor:
    """Preprocess one RGB frame exactly as Meta's *video* loader does.

    Not the image path's transform (:func:`..image.preprocess`): the video loader resizes the 8-bit
    picture with PIL's bilinear filter (which antialiases), scales to [0, 1], **stores the frame in
    float16**, and normalises it in float16. The detector reads it back as float32.

    Args:
        image: ``HxWx3`` RGB ``uint8``.

    Returns:
        ``(1, 3, size, size)`` float32.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"expected an HxWx3 RGB uint8 array, got {image.dtype} {image.shape}")
    size = spec.trunk.image_size
    resized = Image.fromarray(image).convert("RGB").resize((size, size), Image.BILINEAR)
    frame = torch.from_numpy(np.array(resized)).permute(2, 0, 1).contiguous()
    frame = frame.to(torch.float32).div(255).to(torch.float16)
    frame -= torch.tensor(spec.mean, dtype=torch.float16)[:, None, None]
    frame /= torch.tensor(spec.std, dtype=torch.float16)[:, None, None]
    return frame.float().unsqueeze(0)


def load_click_frame(image: np.ndarray, spec: Spec = SPEC) -> torch.Tensor:
    """Preprocess one RGB frame as Meta's tracker loads it for clicks (no detector).

    A third preprocessing, SAM 2's: PIL's *default* resize filter on the 8-bit picture, divided by
    255 in float64, stored as float32, normalised in float32. Meta's tracker-only mode reads frames
    this way, not as its concept tracker does (:func:`load_frame`).
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"expected an HxWx3 RGB uint8 array, got {image.dtype} {image.shape}")
    size = spec.trunk.image_size
    resized = np.array(Image.fromarray(image).convert("RGB").resize((size, size)))
    frame = torch.from_numpy(resized / 255.0).permute(2, 0, 1).to(torch.float32).contiguous()
    frame -= torch.tensor(spec.mean, dtype=torch.float32)[:, None, None]
    frame /= torch.tensor(spec.std, dtype=torch.float32)[:, None, None]
    return frame.unsqueeze(0)


def mask_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU of ``(N, H, W)`` and ``(M, H, W)`` boolean masks, ``(N, M)``."""
    n, h, w = a.shape
    m = b.shape[0]
    a = a.view(n, 1, h * w)
    b = b.view(1, m, h * w)
    inter = (a & b).sum(dim=2).float()
    union = (a | b).sum(dim=2).float()
    return inter / union.clamp(min=1)


def nms(probs: torch.Tensor, masks: torch.Tensor, threshold: float, iou: float) -> torch.Tensor:
    """Greedy mask NMS over the detections above ``threshold``; returns a keep mask.

    Highest score first (NumPy's ``argsort``, reversed, which fixes the order among equal
    scores), dropping every later detection whose mask IoU with a kept one exceeds ``iou``.
    """
    valid = probs > threshold
    kept_probs = probs[valid]
    if kept_probs.numel() == 0:
        return valid
    binary = masks[valid] > 0
    ious = mask_iou(binary, binary).float().cpu().numpy()
    order = kept_probs.float().cpu().numpy().argsort()[::-1]
    kept = []
    while order.size > 0:
        i = order.item(0)
        kept.append(i)
        order = order[np.where(ious[i, order[1:]] <= iou)[0] + 1]
    index = torch.where(valid, valid.cumsum(dim=0) - 1, -1)
    return torch.isin(index, torch.tensor(kept, dtype=torch.int64, device=probs.device))


def component_areas(masks: torch.Tensor) -> torch.Tensor:
    """For each pixel of ``(B, 1, H, W)`` boolean masks, the area of its 8-connected component.

    Zero outside the mask.
    """
    planes = masks.to(torch.uint8).flatten(0, 1).cpu().numpy()
    areas = np.zeros(planes.shape, dtype=np.int64)
    for index, plane in enumerate(planes):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(plane, connectivity=8)
        sizes = stats[:, cv2.CC_STAT_AREA].astype(np.int64)
        sizes[0] = 0
        areas[index] = sizes[labels]
    return torch.from_numpy(areas).reshape(masks.shape).to(masks.device)


def fill_holes(logits: torch.Tensor, max_area: int) -> torch.Tensor:
    """Fill small holes and remove small specks in ``(B, 1, H, W)`` mask logits.

    The video tracker's rule (``fill_holes_in_mask_scores``): every background region of at most
    ``max_area`` pixels becomes foreground at 0.1; then every foreground region of at most
    ``min(max_area, area // 2)`` pixels (the whole mask's area, per object) becomes background at
    -0.1. Regions are 8-connected.
    """
    if max_area <= 0:
        return logits
    background = logits <= 0
    small = background & (component_areas(background) <= max_area)
    logits = torch.where(small, 0.1, logits)
    foreground = logits > 0
    limit = torch.sum(foreground, dim=(2, 3), keepdim=True, dtype=torch.int32)
    limit.floor_divide_(2).clamp_(max=max_area)
    small = foreground & (component_areas(foreground) <= limit)
    return torch.where(small, -0.1, logits)

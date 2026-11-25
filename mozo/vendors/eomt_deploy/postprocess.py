# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------
"""The network's queries out, a partition of the picture into classed segments in.

Upstream's evaluation post-processing (``training/lightning_module.py`` at ``aa017c3``,
``mask_classification_{panoptic,semantic}.py::eval_step``), last layer only, with two changes
that change no number and keep the memory bounded. ``tools/verify/eomt.py`` holds both to
upstream's own functions with ``torch.equal`` on every fixture.

**Panoptic: the kept queries are chosen before the upsample.** Upstream brings all 200 queries'
mask logits to the full picture (``F.interpolate`` to the square, crop, ``F.interpolate`` to
``(H, W)``) and only then keeps those whose class probability clears ``mask_thresh``. On a
1920x1281 photograph that is 200 float maps, ~2 GB. Bilinear interpolation treats each channel on
its own, so interpolating only the kept channels gives the same numbers for them; the rest were
never read.

**Semantic: the per-class map is upsampled in slices and argmax'd as it goes.** Upstream
upsamples all 150 class maps to ``(H, W)`` and takes the argmax: ~1.5 GB on the same photograph.
Here the classes are upsampled a slice at a time and a running maximum kept; a later class wins
only when *strictly* greater, which is ``torch.argmax``'s first-index rule for ties.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .image import pad_size, window_size

__all__ = ["CLASS_SLICE", "MASK_THRESHOLD", "OVERLAP_THRESHOLD", "panoptic", "semantic"]

#: ``mask_thresh`` and ``overlap_thresh``: the panoptic Lightning module's defaults, which no
#: published config overrides.
MASK_THRESHOLD = 0.8
OVERLAP_THRESHOLD = 0.8

#: Class maps upsampled at once in :func:`semantic`. Any value gives the same answer.
CLASS_SLICE = 16


def panoptic(mask_logits: torch.Tensor, class_logits: torch.Tensor, size: tuple[int, int],
             side: int, stuff: tuple[int, ...], threshold: float = MASK_THRESHOLD,
             overlap: float = OVERLAP_THRESHOLD) -> tuple[torch.Tensor, list[tuple[int, int, float]]]:
    """``to_per_pixel_preds_panoptic`` for one image's network output.

    Args:
        mask_logits: ``(queries, h, w)`` at the network's output resolution.
        class_logits: ``(queries, classes + 1)``.
        size: The original picture's ``(H, W)``.
        side: The square the network ran at.
        stuff: Classes merged into one segment each.
        threshold: ``mask_thresh``: the class probability a query needs to be considered.
        overlap: ``overlap_thresh``: the share of a query's own mask it must keep after the
            argmax to become a segment.

    Returns:
        ``(H, W)`` int64 segment ids, -1 where no segment is (void), and one
        ``(segment_id, class_id, score)`` per segment, in id order. ``score`` is the class
        probability of the query that founded the segment.
    """
    num_classes = class_logits.shape[-1] - 1
    scores, classes = class_logits.softmax(dim=-1).max(-1)
    segments = -torch.ones(size, dtype=torch.long, device=mask_logits.device)

    keep = classes.ne(num_classes) & (scores > threshold)
    if not keep.any():
        return segments, []

    # Upstream's revert, on the kept queries only (see the module docstring).
    kept = F.interpolate(mask_logits[keep][None], (side, side), mode="bilinear")
    scaled = pad_size(size, side)
    kept = F.interpolate(kept[:, :, : scaled[0], : scaled[1]], size, mode="bilinear")[0]

    masks = kept.sigmoid()
    kept_scores = scores[keep]
    mask_ids = (kept_scores[..., None, None] * masks).argmax(0)
    stuff_segment_ids: dict[int, int] = {}
    found: list[tuple[int, int, float]] = []
    segment_id = 0

    for k, class_id in enumerate(classes[keep].tolist()):
        orig_mask = masks[k] >= 0.5
        new_mask = mask_ids == k
        final_mask = orig_mask & new_mask

        orig_area = orig_mask.sum().item()
        new_area = new_mask.sum().item()
        final_area = final_mask.sum().item()
        if orig_area == 0 or new_area == 0 or final_area == 0 or new_area / orig_area < overlap:
            continue

        if class_id in stuff:
            if class_id in stuff_segment_ids:
                segments[final_mask] = stuff_segment_ids[class_id]
                continue
            stuff_segment_ids[class_id] = segment_id

        segments[final_mask] = segment_id
        found.append((segment_id, class_id, kept_scores[k].item()))
        segment_id += 1

    return segments, found


def semantic(mask_logits: torch.Tensor, class_logits: torch.Tensor, origins: list[tuple[int, int]],
             size: tuple[int, int], side: int) -> torch.Tensor:
    """``to_per_pixel_logits_semantic`` + ``revert_window_logits_semantic`` + argmax, one image.

    Args:
        mask_logits: ``(windows, queries, h, w)`` at the network's output resolution.
        class_logits: ``(windows, queries, classes + 1)``.
        origins: Each window's ``(start, end)`` along the long side, from
            :func:`~mozo.vendors.eomt_deploy.image.windows`.
        size: The original picture's ``(H, W)``.
        side: The window's side.

    Returns:
        ``(H, W)`` int64 class ids.
    """
    mask_logits = F.interpolate(mask_logits, (side, side), mode="bilinear")
    crop_logits = torch.einsum("bqhw, bqc -> bchw", mask_logits.sigmoid(),
                               class_logits.softmax(dim=-1)[..., :-1])
    del mask_logits

    h, w = window_size(size, side)
    sums = torch.zeros((crop_logits.shape[1], h, w), device=crop_logits.device)
    counts = torch.zeros((crop_logits.shape[1], h, w), device=crop_logits.device)
    for crop, (start, end) in zip(crop_logits, origins):
        if size[0] > size[1]:
            sums[:, start:end, :] += crop
            counts[:, start:end, :] += 1
        else:
            sums[:, :, start:end] += crop
            counts[:, :, start:end] += 1
    del crop_logits
    averaged = (sums / counts)[None]
    del sums, counts

    best = labels = None
    for first in range(0, averaged.shape[1], CLASS_SLICE):
        logits = F.interpolate(averaged[:, first:first + CLASS_SLICE], size, mode="bilinear")[0]
        # argmax, not max(dim): argmax documents the first index on ties.
        index = logits.argmax(0)
        value = logits.gather(0, index[None])[0]
        if best is None:
            best, labels = value, index
        else:
            better = value > best
            best = torch.where(better, value, best)
            labels = torch.where(better, index + first, labels)
    return labels

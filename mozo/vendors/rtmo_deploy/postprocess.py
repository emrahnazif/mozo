# SPDX-License-Identifier: Apache-2.0
"""Everything around RTMO's network that is not a layer: grid cells, box decode, filter, NMS, and
the map back to the frame.

From MMPose v1.3.2: ``MlvlPointGenerator`` (``centralize_points=True``), ``YOLOXPoseHead``'s
``_flatten_predictions`` and ``decode_bbox``, ``RTMOHead.predict``, ``filter_scores_and_topk``,
``nms_torch`` with ``bbox_overlaps``, ``bbox_xyxy2cs`` and ``BottomupPoseEstimator
.add_pred_to_datasample``. Each keeps upstream's order of operations, because each is a place a
"tidier" rewrite moves a float.

Two things upstream does that look like mistakes and are reproduced:

- **The boxes are clamped to the input square, and the joints are not.** The DCC is run on the
  unclamped boxes first; only then are the boxes clamped. A joint can land outside the frame.
- **The map back to the frame is done in numpy, and in float64.** Upstream divides a float32 array
  by a tuple of ints, which promotes. The frame coordinates come back as float64 because of it.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["clamp_boxes", "decode_boxes", "flatten", "grid_priors", "nms", "select", "to_frame",
           "window"]


def grid_priors(sizes: list, strides: tuple[int, ...], dtype: torch.dtype,
                device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Every cell's centre in input pixels ``(N, 2)``, and its stride ``(N,)``, level by level.

    ``centralize_points`` sets the offset to 0 and adds ``(stride - 1) / 2`` instead: the centre of
    a cell's pixels, not the half-pixel grid. 7.5 rather than 8 at stride 16.
    """
    priors, per_cell = [], []
    for (h, w), stride in zip(sizes, strides):
        shift_x = ((torch.arange(0, w, device=device) + 0.0) * stride).to(dtype)
        shift_y = ((torch.arange(0, h, device=device) + 0.0) * stride).to(dtype)
        shift_x = shift_x + float(stride - 1) / 2.0
        shift_y = shift_y + float(stride - 1) / 2.0
        yy, xx = torch.meshgrid(shift_y, shift_x, indexing="ij")
        level = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        priors.append(level)
        per_cell.append(level.new_full((int(h) * int(w),), stride))
    return torch.cat(priors), torch.cat(per_cell)


def flatten(maps: list[torch.Tensor]) -> torch.Tensor:
    """``[(B, C, H, W)]`` per level -> ``(B, sum(H*W), C)``, row-major within a level."""
    return torch.cat([x.permute(0, 2, 3, 1).flatten(1, 2) for x in maps], dim=1)


def decode_boxes(pred: torch.Tensor, priors: torch.Tensor, strides: torch.Tensor) -> torch.Tensor:
    """YOLOX's decode: centre offset and log-size, in strides, to xyxy in input pixels."""
    strides = strides.view(1, strides.size(0), 1)
    priors = priors.view(1, priors.size(0), 2)
    xys = (pred[..., :2] * strides) + priors
    whs = pred[..., 2:].exp() * strides
    return torch.stack([xys[..., 0] - whs[..., 0] / 2, xys[..., 1] - whs[..., 1] / 2,
                        xys[..., 0] + whs[..., 0] / 2, xys[..., 1] + whs[..., 1] / 2], -1)


def _overlaps(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU ``(N, M)``, ``bbox_overlaps(mode='iou')``'s arithmetic, element for element."""
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    lt = torch.max(boxes1[..., :, None, :2], boxes2[..., None, :, :2])
    rb = torch.min(boxes1[..., :, None, 2:], boxes2[..., None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[..., 0] * wh[..., 1]
    union = area1[..., None] + area2[..., None, :] - overlap
    union = torch.max(union, union.new_tensor([1e-6]))
    return overlap / union


def nms(boxes: torch.Tensor, scores: torch.Tensor, threshold: float) -> torch.Tensor:
    """Greedy NMS with ``nms_torch``'s result: indices of the boxes kept, best first.

    Upstream loops on the device -- one IoU row and one host sync per kept box. Here every IoU is
    computed in one call and the greedy pass runs over the boolean matrix on the host. Same
    answer, not an approximation of it: each IoU is the same elementwise arithmetic on the same
    two boxes whichever shape it is computed in, and a box can only be suppressed by one ranked
    above it, which is the order the pass walks.

    Not ``torchvision.ops.nms``: mozo does not need torchvision for this, and ``nms_torch``
    suppresses strictly above the threshold with its own IoU epsilon -- both of which decide which
    of two near-duplicates survives.
    """
    _, order = scores.sort(descending=True)
    ranked = boxes[order]
    suppresses = (_overlaps(ranked, ranked) > threshold).cpu().numpy()
    removed = np.zeros(len(order), dtype=bool)
    kept = []
    for i in range(len(order)):
        if removed[i]:
            continue
        kept.append(i)
        removed[i + 1:] |= suppresses[i, i + 1:]
    return order[torch.tensor(kept, dtype=torch.long, device=order.device)]


def select(scores: torch.Tensor, threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    """``filter_scores_and_topk`` for one class: the cells above *threshold*, best first.

    Returns the sorted scores and the cell indices they came from. ``nms_pre`` is upstream's
    100000, which no input reaches (640x640 has 2000 cells), so there is no cap.
    """
    valid = scores > threshold
    kept = scores[valid]
    cells = torch.nonzero(valid)[:, 0]
    kept, order = kept.sort(descending=True)
    return kept, cells[order]


def window(boxes: torch.Tensor, padding: float) -> torch.Tensor:
    """``bbox_xyxy2cs`` concatenated: ``(N, 4)`` xyxy -> ``(N, 4)`` centre and padded size."""
    scale = (boxes[..., 2:] - boxes[..., :2]) * padding
    center = (boxes[..., 2:] + boxes[..., :2]) * 0.5
    return torch.cat((center, scale), dim=1)


def clamp_boxes(boxes: torch.Tensor, side: int) -> torch.Tensor:
    """Clamp xyxy boxes to the input square, in place -- after the DCC has used them unclamped."""
    boxes[:, 0::2].clamp_(0, side)
    boxes[:, 1::2].clamp_(0, side)
    return boxes


def to_frame(points: np.ndarray, size: tuple[int, int], center: np.ndarray,
             scale: np.ndarray) -> np.ndarray:
    """Input pixels -> frame pixels, as ``add_pred_to_datasample`` writes it (float64 out).

    *points* is ``(..., 2)`` float32, *size* the input square as a tuple of ints, *center* and
    *scale* float32: those dtypes are what make the result float64, so they are part of the
    contract rather than incidental.
    """
    return points / size * scale + center - 0.5 * scale

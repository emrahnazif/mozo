# SPDX-License-Identifier: Apache-2.0
# Derived from libfacedetection.train, Copyright (c) 2022-2026, Shiqi Yu (BSD-3-Clause: see LICENSE).
"""From the network's raw maps to faces: priors, decode, score, threshold, NMS.

``YuNetPostprocessor`` (``tasks/face/postprocess.py``), ``MlvlPointGenerator``
(``engine/priors.py``), ``bbox_decode``/``kps_decode`` (``tasks/face/codec.py``) and ``nms``
(``engine/nms.py``) at commit ``dca340aa``, for one image and one class.

Three things here are upstream's arithmetic and are kept as written, because each is a different
rounding from the obvious rewrite and none of them raises:

* A cell's prior is its corner, ``(col * stride, row * stride)``, not its centre, and the decode is
  ``offset * stride + corner`` -- OpenCV's FaceDetectorYN computes ``(col + offset) * stride``
  instead, which rounds differently.
* The score is ``sigmoid(cls) * sigmoid(obj)``. OpenCV reports the square root of it.
* NMS keeps a box whose IoU with a kept one is ``<= 0.45``, IoU dividing by the union clamped to
  float32 epsilon, and the survivors are re-sorted by score at the end, as ``batched_nms`` does.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["decode", "grid_priors", "nms"]


def grid_priors(featmap_sizes: list[tuple[int, int]], strides: tuple[int, ...],
                dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """``(sum H*W, 4)`` rows of ``(x, y, stride, stride)``, row-major per level, levels in order.

    ``MlvlPointGenerator(strides, offset=0).grid_priors(..., with_stride=True)``, concatenated.
    """
    levels = []
    for (feat_h, feat_w), stride in zip(featmap_sizes, strides):
        # Integer arithmetic, cast after, as upstream does with its offset of 0.
        shift_x = torch.arange(0, feat_w, device=device) * stride
        shift_y = torch.arange(0, feat_h, device=device) * stride
        yy, xx = torch.meshgrid(shift_y.to(dtype), shift_x.to(dtype), indexing="ij")
        xx, yy = xx.reshape(-1), yy.reshape(-1)
        stride_w = xx.new_full((xx.shape[0],), stride).to(dtype)
        stride_h = yy.new_full((yy.shape[0],), stride).to(dtype)
        levels.append(torch.stack([xx, yy, stride_w, stride_h], dim=-1))
    return torch.cat(levels, dim=0)


def _flatten(preds: list[torch.Tensor]) -> torch.Tensor:
    """``[(1, C, H, W)]`` per level -> ``(sum H*W, C)``, cells row-major, as the priors are."""
    return torch.cat([p.permute(0, 2, 3, 1).reshape(1, -1, p.shape[1]) for p in preds], dim=1)[0]


#: Above this many candidates :func:`nms` walks upstream's loop instead of building the IoU
#: matrix, whose size grows with the square: at a threshold near 0 an 8K image makes every one of
#: ~170k cells a candidate. Below it the matrix is at most 4 MB per intermediate.
MATRIX_LIMIT = 1024


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """Greedy NMS, best first; indices of the kept boxes in the order they were kept.

    Upstream's ``nms`` loops on the device, one IoU row per kept box -- a dozen small kernels and
    a sync per face, 1.2 ms on CUDA for 32 candidates. For up to :data:`MATRIX_LIMIT` candidates
    every IoU is computed at once instead, with the same operations in the same order (so the
    same float32 values: ``max``, ``min``, ``clamp``, ``+``, ``-``, ``*`` and ``/`` round
    identically one element at a time or many), compared with upstream's ``<=``, and only the
    greedy walk over the resulting booleans runs on the host. Past it, upstream's loop. The gate
    holds both to upstream's indices.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)
    eps = torch.finfo(boxes.dtype).eps
    if len(order) > MATRIX_LIMIT:
        keep = []
        while order.numel() > 0:
            # A plain int, where upstream keeps ``order[0]`` -- a view that holds all of ``order``
            # alive, one copy per kept box: at a threshold of 0 on an 8K image that exhausted a
            # 24 GB GPU. The values indexed are the same.
            i = int(order[0])
            keep.append(i)
            if order.numel() == 1:
                break
            rest = order[1:]
            xx1 = torch.maximum(x1[i], x1[rest])
            yy1 = torch.maximum(y1[i], y1[rest])
            xx2 = torch.minimum(x2[i], x2[rest])
            yy2 = torch.minimum(y2[i], y2[rest])
            inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
            union = areas[i] + areas[rest] - inter
            order = rest[inter / union.clamp(min=eps) <= iou_threshold]
        return torch.tensor(keep, device=order.device)

    x1, y1, x2, y2, areas = x1[order], y1[order], x2[order], y2[order], areas[order]
    xx1 = torch.maximum(x1[:, None], x1[None, :])
    yy1 = torch.maximum(y1[:, None], y1[None, :])
    xx2 = torch.minimum(x2[:, None], x2[None, :])
    yy2 = torch.minimum(y2[:, None], y2[None, :])
    inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
    union = areas[:, None] + areas[None, :] - inter
    suppresses = (~(inter / union.clamp(min=eps) <= iou_threshold)).cpu().numpy()

    keep = []
    removed = np.zeros(len(order), dtype=bool)
    for i in range(len(order)):
        if not removed[i]:
            keep.append(i)
            removed |= suppresses[i]
    return order[torch.tensor(keep, device=order.device)]


def decode(outputs, strides: tuple[int, ...], threshold: float, iou_threshold: float
           ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The faces in one image, best first, in the padded input's pixels.

    Args:
        outputs: ``(cls, bbox, obj, kps)``, each three ``(1, C, H, W)`` maps, as
            :class:`~.network.YuNet` returns them.
        strides: One per level.
        threshold: Keep cells whose score is ``>=`` this, before NMS -- upstream's
            ``score_threshold``.
        iou_threshold: NMS overlap above which the lower-scored box goes.

    Returns:
        ``(boxes (N, 4) xyxy, scores (N,), keypoints (N, 10))``, float32, keypoints as
        ``x0, y0, ..., x4, y4``.
    """
    cls_preds, bbox_preds, obj_preds, kps_preds = outputs
    sizes = [tuple(c.shape[2:]) for c in cls_preds]
    priors = grid_priors(sizes, strides, cls_preds[0].dtype, cls_preds[0].device)

    cls = _flatten(cls_preds).sigmoid()
    bbox = _flatten(bbox_preds)
    obj = _flatten(obj_preds).squeeze(-1).sigmoid()
    kps = _flatten(kps_preds)

    # bbox_decode / kps_decode, as upstream writes them.
    xys = (bbox[..., :2] * priors[..., 2:]) + priors[..., :2]
    whs = bbox[..., 2:].exp() * priors[..., 2:]
    boxes = torch.stack([xys[..., 0] - whs[..., 0] / 2, xys[..., 1] - whs[..., 1] / 2,
                         xys[..., 0] + whs[..., 0] / 2, xys[..., 1] + whs[..., 1] / 2], dim=-1)
    points = torch.cat([(kps[..., [2 * i, 2 * i + 1]] * priors[..., 2:]) + priors[..., :2]
                        for i in range(kps.shape[-1] // 2)], dim=-1)

    max_scores, _ = torch.max(cls, dim=1)
    scores = max_scores * obj
    keep = scores >= threshold
    boxes, scores, points = boxes[keep], scores[keep], points[keep]

    kept = nms(boxes, scores, iou_threshold)
    if kept.numel():
        # ``batched_nms`` re-sorts what one class kept by score; with one class this is the order
        # NMS kept them in, re-derived the way upstream derives it so ties fall the same way.
        _, order = scores[kept].sort(descending=True)
        kept = kept[order]
    return boxes[kept], scores[kept], points[kept]

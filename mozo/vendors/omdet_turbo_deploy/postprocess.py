# SPDX-License-Identifier: Apache-2.0
"""Raw boxes and logits -> scored, suppressed, clipped detections in pixels.

``post_process_grounded_object_detection`` from transformers' ``OmDetTurboProcessor``, for one
image, op for op -- including the parts that look like they could be simplified:

- every (query, class) pair is a candidate, sigmoid-scored; the best 900 of the 900xC are kept
  with ``topk(sorted=False)``, whose order is the kernel's;
- boxes are (cx, cy, w, h) in [0, 1], scaled by the *original* image's width and height;
- ``> threshold``, then class-wise NMS at 0.5 with torchvision's ``batched_nms``, then clipping.

The result is in NMS order (best first). ``nms`` defaults to upstream's 0.5.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torchvision.ops.boxes import batched_nms

__all__ = ["NMS", "postprocess"]

#: Upstream's class-wise NMS IoU threshold.
NMS = 0.5


def postprocess(boxes: Tensor, logits: Tensor, size: tuple[int, int], threshold: float,
                nms: float = NMS) -> tuple[Tensor, Tensor, Tensor]:
    """One image's detections.

    Args:
        boxes: ``(900, 4)`` sigmoid (cx, cy, w, h).
        logits: ``(900, C)`` class logits.
        size: The original image's ``(height, width)``.
        threshold: Keep scores strictly above this.

    Returns:
        ``(N, 4)`` xyxy boxes in pixels, ``(N,)`` scores, ``(N,)`` class indices.
    """
    queries, classes = logits.shape
    scores = torch.sigmoid(logits)
    labels = torch.arange(classes, device=logits.device).unsqueeze(0).repeat(queries, 1).flatten(0, 1)
    scores, top = scores.flatten(0, 1).topk(queries, sorted=False)
    labels = labels[top]
    boxes = boxes.view(-1, 1, 4).repeat(1, classes, 1).view(-1, 4)[top]

    cx, cy, w, h = boxes.unbind(-1)
    boxes = torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)
    boxes = boxes * torch.tensor(size[::-1]).repeat(2).to(boxes.device)

    keep = (scores > threshold).nonzero(as_tuple=False).view(-1)
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    keep = batched_nms(boxes, scores, labels, nms)
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    height, width = size
    boxes = torch.stack((boxes[:, 0].clamp(min=0, max=width), boxes[:, 1].clamp(min=0, max=height),
                         boxes[:, 2].clamp(min=0, max=width), boxes[:, 3].clamp(min=0, max=height)),
                        dim=-1)
    return boxes, scores, labels

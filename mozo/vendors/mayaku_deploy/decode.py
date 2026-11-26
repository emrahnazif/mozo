# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 the mayaku authors. Extracted from mayaku 3.1.1 (mayaku/inference/decode.py,
# mayaku/model/box.py), reduced to the prediction decode: one class per location, a score
# threshold, top-k, class-wise NMS. Upstream's evaluation decode (every class above 0.001) is not
# here; mozo serves predictions.
"""Raw head maps -> boxes, scores and class ids in canvas pixels, plus the anchor each came from.

Upstream thresholds on the class logits before decoding any box, so the DFL expectation and the box
arithmetic touch the few anchors that survive rather than all 13,125 of an 800 canvas. Kept that
way: decoding every anchor and then selecting gives the same numbers only up to the order of
operations, and the gate is exact.
"""

from __future__ import annotations

import functools
import math

import torch
from torchvision.ops import batched_nms

from .blocks import STRIDES


def anchor_grid(shapes, strides, device="cpu", dtype=torch.float32):
    """Cell centres ``(N, 2)`` in pixels and strides ``(N, 1)`` for (H, W) feature shapes,
    concatenated fine to coarse. Cached and shared: read-only."""
    return _grid(tuple(tuple(s) for s in shapes), tuple(strides), torch.device(device), dtype)


@functools.lru_cache(maxsize=32)
def _grid(shapes, strides, device, dtype, offset=0.5):
    points, strides_out = [], []
    for (h, w), s in zip(shapes, strides):
        sx = (torch.arange(w, device=device, dtype=dtype) + offset) * s
        sy = (torch.arange(h, device=device, dtype=dtype) + offset) * s
        y, x = torch.meshgrid(sy, sx, indexing="ij")
        points.append(torch.stack((x.reshape(-1), y.reshape(-1)), -1))
        strides_out.append(torch.full((h * w, 1), float(s), device=device, dtype=dtype))
    return torch.cat(points), torch.cat(strides_out)


def flatten_levels(maps, c):
    """Per-level maps ``[(B, c, h, w), ...]`` -> ``(B, N, c)``, N fine to coarse."""
    b = maps[0].shape[0]
    return torch.cat([m.reshape(b, c, -1) for m in maps], 2).permute(0, 2, 1)


def decode_boxes(dist, points, stride, reg_max):
    """DFL bin logits ``(n, 4 * reg_max)`` -> xyxy boxes ``(n, 4)`` in pixels: the softmax mean
    over bins 0..reg_max-1 per side (GFL, arXiv 2006.04388), as ltrb distances in strides."""
    p = dist.unflatten(-1, (4, reg_max)).softmax(-1)
    bins = torch.arange(reg_max, device=dist.device, dtype=dist.dtype)
    ltrb = (p * bins).sum(-1) * stride
    lt, rb = ltrb.chunk(2, -1)
    return torch.cat((points - lt, points + rb), -1)


def postprocess(preds, nc, reg_max, conf, iou, topk, max_det):
    """The head's six maps of one image -> ``(dets, anchors, points, stride)``.

    ``dets`` is ``(n, 6)`` ``[x1, y1, x2, y2, score, class]`` in canvas pixels, highest score
    first; ``anchors`` the flat anchor index each came from, which is how the mask branch finds its
    kernels; ``points`` and ``stride`` the anchor grid, on the device *preds* are on.
    """
    head = preds[:2 * len(STRIDES)]
    shapes = [tuple(c.shape[2:]) for c in head[0::2]]
    # Upstream's own flattening (one concatenation per kind), kept as it is for exactness.
    cls, dist = flatten_levels(head[0::2], nc)[0], flatten_levels(head[1::2], 4 * reg_max)[0]
    points, stride = anchor_grid(shapes, STRIDES, device=cls.device, dtype=cls.dtype)
    # The threshold is applied to the logit, never to a sigmoid: comparing logits is what upstream
    # does, and the two disagree at the last bit for scores near the threshold.
    floor = math.log(conf / (1 - conf)) if conf > 0 else -math.inf
    best, lab = cls.max(1)
    anchor = (best > floor).nonzero().squeeze(1)
    score, lab = best[anchor].sigmoid(), lab[anchor]
    if len(score) > topk:
        keep = score.topk(topk).indices
        score, lab, anchor = score[keep], lab[keep], anchor[keep]
    box = decode_boxes(dist[anchor], points[anchor], stride[anchor], reg_max)
    keep = batched_nms(box.float(), score.float(), lab, iou)[:max_det]
    dets = torch.cat((box[keep], score[keep, None], lab[keep, None].float()), 1)
    return dets, anchor[keep], points, stride

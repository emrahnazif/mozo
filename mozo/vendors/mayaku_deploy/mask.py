# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 the mayaku authors. Extracted from mayaku 3.1.1 (mayaku/model/mask.py,
# mayaku/inference/decode.py), reduced to the decode of one image's masks.
"""The per-instance mask head, run on the host over the graph's stride-4 mask feature.

Dynamic kernels in the manner of CondInst (arXiv 2003.05664) and RTMDet-Ins (arXiv 2212.07784):
each detection's anchor emitted 169 numbers, the weights and biases of a three-layer 1x1 network
(8 feature + 2 relative-coordinate channels -> 8 -> 8 -> 1, ReLU between), and that network runs
over the mask feature inside the detection's box.

The full definition is: logits over the whole map, -inf outside the box's cells, bilinear upsample
to the canvas, cut the letterbox, bilinear resize to the source image, threshold at 0. Upstream
computes the same thing without full-frame maps -- both resizes folded into two small weight tables
per image size (:func:`_axis`), the network run on the box's cells only, detections batched in
chunks of similar size -- and this copies that operation for operation, chunking included, because
a different chunking is a different matrix shape and can be different bits.
"""

from __future__ import annotations

import functools
import itertools
import math

import torch
import torch.nn.functional as F

from .blocks import STRIDES
from .decode import anchor_grid, flatten_levels
from .image import fitted
from .network import KERNEL_LAYOUT, KERNEL_PARAMS

#: Relative coordinates are in units of this many strides of the anchor's level (CondInst's size
#: of interest): a stride-8 anchor sees +-1 at 64 px.
COORD_SCALE = 8

#: Most mask elements one chunk computes at once (a float each, 512 MB).
MASK_BUDGET = 1 << 27


def stride_of(mask_feat, head0):
    """The mask feature's stride, from its size and the finest head map's."""
    return head0.shape[-2] * STRIDES[0] // mask_feat.shape[-2]


def split_kernels(k):
    """(P, 169) -> [(W1 (P,8,10), b1 (P,8)), (W2 (P,8,8), b2), (W3 (P,1,8), b3)]."""
    out, o = [], 0
    for cin, cout in KERNEL_LAYOUT:
        w = k[:, o:o + cin * cout].reshape(-1, cout, cin)
        o += cin * cout
        b = k[:, o:o + cout]
        o += cout
        out.append((w, b))
    return out


def rel_coords(points, stride, h, w, mask_stride, dtype, cells):
    """Relative coordinates of the cells *cells* (P, Q) of the stride-*mask_stride* map, seen from
    each detection's anchor, in units of ``COORD_SCALE * stride`` -> (P, 2, Q)."""
    grid = anchor_grid([(h, w)], [mask_stride], device=points.device, dtype=dtype)[0].T  # (2, h*w)
    grid = grid[:, cells].transpose(0, 1)
    scale = (COORD_SCALE * stride).to(dtype)                       # (P, 1)
    return (grid - points.to(dtype)[:, :, None]) / scale[:, :, None]


def dyn_conv(feat, coords, kernels):
    """Each instance's network over its own cells: feat (P, 8, Q), coords (P, 2, Q), kernels
    (P, 169) -> logits (P, Q)."""
    x = torch.cat((coords.to(feat.dtype), feat), 1)                # (P, 10, Q)
    layers = split_kernels(kernels.to(feat.dtype))
    for i, (w, b) in enumerate(layers):
        x = torch.baddbmm(b[:, :, None], w, x)                     # (P, cout, Q)
        if i + 1 < len(layers):
            x = F.relu(x)
    return x[:, 0]


def crop_cells(boxes, h, w, mask_stride):
    """xyxy canvas boxes -> inclusive-exclusive cell rectangles (x0, y0, x1, y1) on the mask map."""
    s = mask_stride
    x0 = (boxes[:, 0] / s).floor().long().clamp(0, w - 1)
    y0 = (boxes[:, 1] / s).floor().long().clamp(0, h - 1)
    x1 = torch.maximum((boxes[:, 2] / s).ceil().long(), x0 + 1).clamp(max=w)
    y1 = torch.maximum((boxes[:, 3] / s).ceil().long(), y0 + 1).clamp(max=h)
    return x0, y0, x1, y1


@functools.lru_cache(maxsize=64)
def _axis(cells, mask_stride, offset, fit, out, device):
    """One axis of the two resizes -- cells to canvas (x *mask_stride*), the letterbox cut
    ``[offset, offset + fit)``, canvas to *out* source pixels -- run over an identity and over -inf
    masks, so every number is those interpolations' own.

    Returns weights (out, cells), each output pixel's weight on each cell, and lo, hi (out,), the
    first and last cell each output pixel reads: a pixel can be set only when every cell it reads
    is inside the box, which gives the rectangle a box's mask can occupy."""
    def interp(x):  # x (cells, n): each column resampled alone -> (out, n)
        x = F.interpolate(x.T[:, None, :, None], size=(cells * mask_stride, 1), mode="bilinear",
                          align_corners=False)
        return F.interpolate(x[:, :, offset:offset + fit], size=(out, 1), mode="bilinear",
                             align_corners=False)[:, 0, :, 0].T
    ar = torch.arange(cells, device=device)
    weights = interp(torch.eye(cells, device=device))
    inf = torch.tensor(float("-inf"), device=device)
    below = torch.where(ar[:, None] < ar[None, :], inf, 0.0)
    above = torch.where(ar[:, None] > ar[None, :], inf, 0.0)
    lo = interp(below).isfinite().sum(1) - 1
    hi = cells - interp(above).isfinite().sum(1)
    return weights, lo, hi


class _Axis:
    """One axis (rows or columns) of an image's box masks: the weight table, each box's cell range
    ``[c0, c1)`` and the pixel range ``[r0, r1)`` its mask can occupy."""

    def __init__(self, meta, dim, cells, mask_stride, c0, c1):
        dev = c0.device
        n = meta["shape"][dim]
        offset = meta["pad"][1 - dim]                             # pad is (left, top)
        fit = fitted(meta["shape"], meta["ratio"])[dim]
        self.cells, self.n, self.c0, self.c1 = cells, n, c0, c1
        self.weights, lo, hi = _axis(cells, mask_stride, offset, fit, n, dev)
        ok = (lo[None] >= c0[:, None]) & (hi[None] < c1[:, None])
        self.r0 = ok.to(torch.uint8).argmax(1)
        self.r1 = self.r0 + ok.sum(1)

    def chunk(self, k, n_cells, n_px):
        """For detections *k* padded to *n_cells* cells and *n_px* pixels: their cell indices
        (g, n_cells) clamped to the map, which are inside the box, which pixels are inside the
        rectangle (g, n_px), and the weights (g, n_px, n_cells), zero outside both."""
        c = self.c0[k, None] + torch.arange(n_cells, device=k.device)
        inc = c < self.c1[k, None]
        c = c.clamp(max=self.cells - 1)
        r = self.r0[k, None] + torch.arange(n_px, device=k.device)
        okr = r < self.r1[k, None]
        w = self.weights[r.clamp(max=self.n - 1)[:, :, None], c[:, None, :]]
        return c, inc, okr, w * (okr[:, :, None] & inc[:, None, :])


def _chunks(rh, rw, budget):
    """Lists of detection indices with similar rectangles: grouped by height and width band (powers
    of 4), each group split so its count times its padded area stays within *budget*."""
    def band(i):
        return math.ceil(math.log2(rh[i]) / 2), math.ceil(math.log2(rw[i]) / 2)

    for _, grp in itertools.groupby(sorted(range(len(rh)), key=band), key=band):
        chunk, mh, mw = [], 0, 0
        for i in grp:
            nh, nw = max(mh, rh[i]), max(mw, rw[i])
            if chunk and (len(chunk) + 1) * nh * nw > budget:
                yield chunk
                chunk, nh, nw = [], rh[i], rw[i]
            chunk.append(i)
            mh, mw = nh, nw
        yield chunk


@torch.no_grad()
def decode_masks(preds, dets, anchors, points, stride, meta, budget=MASK_BUDGET):
    """One image's masks in its source frame: ``(n, h, w)`` bool, on the CPU.

    *preds* is the graph's whole output list (the mask feature and the three kernel maps follow the
    six head maps), *dets* the decoded detections in canvas pixels and *anchors* their anchor
    indices, all from :func:`~.decode.postprocess`; *meta* the letterbox meta.
    """
    head0, mask_feat, kers = preds[0], preds[6], preds[7:10]
    mask_stride = stride_of(mask_feat, head0)
    kflat = flatten_levels(kers, KERNEL_PARAMS)[0]
    mask_feat = mask_feat.float()
    kernels, pts, strd = kflat[anchors].float(), points[anchors], stride[anchors]
    boxes = dets[:, :4].float()

    out = torch.zeros(len(dets), *meta["shape"], dtype=torch.bool)
    _, _, hs, ws = mask_feat.shape
    x0, y0, x1, y1 = crop_cells(boxes, hs, ws, mask_stride)
    ys = _Axis(meta, 0, hs, mask_stride, y0, y1)
    xs = _Axis(meta, 1, ws, mask_stride, x0, x1)
    feat = mask_feat[0].flatten(1).transpose(0, 1)               # (Hs*Ws, 8)
    rh, rw, ch, cw = torch.stack(((ys.r1 - ys.r0).clamp(min=1), (xs.r1 - xs.r0).clamp(min=1),
                                  y1 - y0, x1 - x0)).tolist()
    for idx in _chunks(rh, rw, budget):
        k = torch.tensor(idx, device=mask_feat.device)
        cy, iny, oky, a = ys.chunk(k, max(ch[i] for i in idx), max(rh[i] for i in idx))
        cx, inx, okx, b = xs.chunk(k, max(cw[i] for i in idx), max(rw[i] for i in idx))
        cell = (cy[:, :, None] * ws + cx[:, None, :]).flatten(1)   # (g, cells)
        coords = rel_coords(pts[k], strd[k], hs, ws, mask_stride, mask_feat.dtype, cell)
        lg = dyn_conv(feat[cell].transpose(1, 2), coords, kernels[k])
        lg = lg.view(len(idx), cy.shape[1], cx.shape[1]).masked_fill(
            ~(iny[:, :, None] & inx[:, None, :]), 0)
        m = (a @ lg @ b.transpose(1, 2) > 0) & oky[:, :, None] & okx[:, None, :]
        rects = torch.stack((ys.r0[k], ys.r1[k], xs.r0[k], xs.r1[k]), 1)
        for i, mi, (r0, r1, c0, c1) in zip(idx, m.cpu(), rects.tolist()):
            out[i, r0:r1, c0:c1] = mi[:r1 - r0, :c1 - c0]
    return out

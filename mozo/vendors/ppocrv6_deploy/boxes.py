# SPDX-License-Identifier: Apache-2.0
# Extracted from PaddleX 3.7.2 (the engine under PaddleOCR 3.7.0):
#   paddlex/inference/models/text_detection/processors.py         DBPostProcess (quad)
#   paddlex/inference/pipelines/components/common/sort_boxes.py   SortQuadBoxes
# Copyright (c) 2024-2026 PaddlePaddle Authors. Apache-2.0.
#
# ``_offset`` and ``_union`` reimplement, for the one case DBPostProcess uses, the round-join
# polygon offset of Angus Johnson's Clipper 6.4.2 (the C++ library pyclipper wraps), Copyright
# Angus Johnson 2010-2017, Boost Software License 1.0 -- see NOTICE.
"""From the detector's probability map to sorted quadrilaterals in page coordinates.

Upstream expands each box with ``pyclipper``. mozo does not depend on it: the expansion here
reproduces Clipper's integer output exactly for the convex quadrilaterals DB hands it, which
tools/verify/ppocrv6.py checks against pyclipper on real maps and on random boxes.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

#: ``OCR.yaml``'s TextDetection settings in PaddleOCR 3.7.0. ``box_thresh`` is mozo's
#: ``threshold``; the others are fixed properties of the published pipeline.
THRESH = 0.3
BOX_THRESH = 0.6
UNCLIP_RATIO = 1.5
MAX_CANDIDATES = 3000   # inference.yml PostProcess.max_candidates, every PP-OCRv6 detector
MIN_SIZE = 3


def quads(prob: np.ndarray, shape: np.ndarray, box_thresh: float = BOX_THRESH) -> tuple[np.ndarray, list]:
    """``DBPostProcess.boxes_from_bitmap``: ``(H, W)`` probabilities to int16 quads and scores."""
    bitmap = prob > THRESH
    src_h, src_w, _ratio_h, _ratio_w = shape
    dest_width, dest_height = src_w, src_h
    height, width = bitmap.shape
    width_scale = dest_width / width
    height_scale = dest_height / height

    contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)
    boxes, scores = [], []
    for contour in contours[:MAX_CANDIDATES]:
        points, sside = mini_box(contour)
        if sside < MIN_SIZE:
            continue
        points = np.array(points)
        score = _box_score_fast(prob, points.reshape(-1, 2))
        if box_thresh > score:
            continue
        box, sside = mini_box(unclip(points, UNCLIP_RATIO).reshape(-1, 1, 2))
        if sside < MIN_SIZE + 2:
            continue
        box = np.array(box)
        for i in range(box.shape[0]):
            box[i, 0] = max(0, min(round(box[i, 0] * width_scale), dest_width))
            box[i, 1] = max(0, min(round(box[i, 1] * height_scale), dest_height))
        boxes.append(box.astype(np.int16))
        scores.append(score)
    return np.array(boxes, dtype=np.int16), scores


def mini_box(contour) -> tuple[list, float]:
    """``get_mini_boxes``: the minimum-area rectangle, corners ordered from the top-left."""
    rect = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(rect)), key=lambda x: x[0])
    a, d = (0, 1) if points[1][1] > points[0][1] else (1, 0)
    b, c = (2, 3) if points[3][1] > points[2][1] else (3, 2)
    return [points[a], points[b], points[c], points[d]], min(rect[1])


def _box_score_fast(bitmap: np.ndarray, box: np.ndarray) -> float:
    h, w = bitmap.shape[:2]
    box = box.copy()
    xmin = max(0, min(math.floor(box[:, 0].min()), w - 1))
    xmax = max(0, min(math.ceil(box[:, 0].max()), w - 1))
    ymin = max(0, min(math.floor(box[:, 1].min()), h - 1))
    ymax = max(0, min(math.ceil(box[:, 1].max()), h - 1))
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def unclip(box: np.ndarray, ratio: float) -> np.ndarray:
    """``DBPostProcess.unclip``: grow the box by ``area * ratio / perimeter``, round corners.

    Returns what ``np.array(pyclipper.PyclipperOffset(...).Execute(distance))`` returns: one
    int64 polygon, shape ``(1, M, 2)``, or an empty array.
    """
    distance = cv2.contourArea(box) * ratio / cv2.arcLength(box, True)
    polygon = _union(_offset(box, distance))
    return np.array([polygon]) if polygon else np.array([])


# -- Clipper 6.4.2, ClipperOffset with jtRound / etClosedPolygon, one path, delta > 0 -------------

_ARC_TOLERANCE = 0.25   # ClipperOffset's default ArcTolerance


def _round(value: float) -> int:
    """Clipper's ``Round``: half away from zero."""
    return int(value - 0.5) if value < 0 else int(value + 0.5)


def _unit_normal(a, b) -> tuple[float, float]:
    if a == b:
        return (0.0, 0.0)
    dx = float(b[0] - a[0])
    dy = float(b[1] - a[1])
    f = 1 * 1.0 / math.sqrt(dx * dx + dy * dy)
    dx *= f
    dy *= f
    return (dy, -dx)


def _offset(points, delta: float) -> list[tuple[int, int]]:
    """``AddPath`` + ``FixOrientations`` + ``DoOffset``: the raw offset polygon, before the union."""
    # pyclipper turns each float coordinate into Clipper's integer by C truncation, toward zero.
    path = [(int(x), int(y)) for x, y in points]
    high = len(path) - 1
    while high > 0 and path[0] == path[high]:
        high -= 1
    contour = [path[0]]
    for i in range(1, high + 1):
        if contour[-1] != path[i]:
            contour.append(path[i])
    if len(contour) < 3:
        return []

    # Orientation is Area >= 0, with Clipper's own sign convention; a single path is reversed
    # when its orientation is wrong.
    area = 0.0
    j = len(contour) - 1
    for i in range(len(contour)):
        area += (float(contour[j][0]) + contour[i][0]) * (float(contour[j][1]) - contour[i][1])
        j = i
    if not -area * 0.5 >= 0:
        contour.reverse()

    tolerance = _ARC_TOLERANCE
    if _ARC_TOLERANCE > abs(delta) * _ARC_TOLERANCE:
        tolerance = abs(delta) * _ARC_TOLERANCE
    steps = math.pi / math.acos(1 - tolerance / abs(delta))
    if steps > abs(delta) * math.pi:
        steps = abs(delta) * math.pi
    sin = math.sin(2 * math.pi / steps)
    cos = math.cos(2 * math.pi / steps)
    steps_per_rad = steps / (2 * math.pi)

    n = len(contour)
    normals = [_unit_normal(contour[i], contour[i + 1]) for i in range(n - 1)]
    normals.append(_unit_normal(contour[n - 1], contour[0]))

    out: list[tuple[int, int]] = []
    k = n - 1
    for j in range(n):                                   # OffsetPoint(j, k, jtRound)
        nk, nj = normals[k], normals[j]
        px, py = contour[j]
        sin_a = nk[0] * nj[1] - nj[0] * nk[1]
        if abs(sin_a * delta) < 1.0:
            if nk[0] * nj[0] + nj[1] * nk[1] > 0:
                out.append((_round(px + nk[0] * delta), _round(py + nk[1] * delta)))
                continue                                 # Clipper returns here, k unchanged
        elif sin_a > 1.0:
            sin_a = 1.0
        elif sin_a < -1.0:
            sin_a = -1.0
        if sin_a * delta < 0:
            out.append((_round(px + nk[0] * delta), _round(py + nk[1] * delta)))
            out.append((px, py))
            out.append((_round(px + nj[0] * delta), _round(py + nj[1] * delta)))
        else:                                            # DoRound(j, k)
            a = math.atan2(sin_a, nk[0] * nj[0] + nk[1] * nj[1])
            count = max(_round(steps_per_rad * abs(a)), 1)
            x, y = nk
            for _ in range(count):
                out.append((_round(px + x * delta), _round(py + y * delta)))
                x2 = x
                x = x * cos - sin * y
                y = x2 * sin + y * cos
            out.append((_round(px + nj[0] * delta), _round(py + nj[1] * delta)))
        k = j
    return out


def _union(polygon: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """What Clipper's positive-fill union hands back for the offset of a convex quad.

    The union re-traces the polygon: repeated and collinear vertices go, the orientation stays,
    and the output ends at the lowest-y vertex with the largest x. The vertex *order* matters
    downstream -- ``cv2.minAreaRect`` can return the same rectangle described from another
    corner, 90 degrees apart, depending on where the polygon starts.
    """
    points = [p for i, p in enumerate(polygon) if p != polygon[i - 1]]
    changed = True
    while changed and len(points) >= 3:
        changed = False
        for i in range(len(points)):
            a, b, c = points[i - 1], points[i], points[(i + 1) % len(points)]
            if (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) == 0:
                points.pop(i)
                changed = True
                break
    if len(points) < 3:
        return []
    lowest = min(p[1] for p in points)
    end = points.index(max(p for p in points if p[1] == lowest))
    return points[end + 1:] + points[:end + 1]


def sort_quads(quads: np.ndarray) -> list[np.ndarray]:
    """``SortQuadBoxes``: top to bottom by first corner, then left to right within 10 pixels."""
    boxes = list(sorted(np.array(quads), key=lambda x: (x[0][1], x[0][0])))
    for i in range(len(boxes) - 1):
        for j in range(i, -1, -1):
            if abs(boxes[j + 1][0][1] - boxes[j][0][1]) < 10 and boxes[j + 1][0][0] < boxes[j][0][0]:
                boxes[j], boxes[j + 1] = boxes[j + 1], boxes[j]
            else:
                break
    return boxes

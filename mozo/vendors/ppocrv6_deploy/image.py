# SPDX-License-Identifier: Apache-2.0
# Extracted from PaddleX 3.7.2 (the engine under PaddleOCR 3.7.0):
#   paddlex/inference/models/text_detection/processors.py      DetResizeForTest, NormalizeImage
#   paddlex/inference/models/text_recognition/processors.py    OCRReisizeNormImg, ToBatch
#   paddlex/inference/pipelines/components/common/crop_image_regions.py   CropByPolys (quad)
# Copyright (c) 2024-2026 PaddlePaddle Authors. Apache-2.0.
"""Pixels into both networks: the page for the detector, and each line for the recogniser.

Everything here works on BGR, as upstream does from its decoder onwards; the caller converts
once. The arithmetic is upstream's line for line, including the parts that look odd -- see
PROVENANCE.md, under "Measured traps".
"""

from __future__ import annotations

import math

import cv2
import numpy as np

#: ``OCR.yaml``'s TextDetection settings in PaddleOCR 3.7.0: scale the page up until its short
#: side is at least 64, never let the long side pass 4000, round both to multiples of 32.
LIMIT_SIDE_LEN = 64
LIMIT_TYPE = "min"
MAX_SIDE_LIMIT = 4000

#: ``RecResizeImg.image_shape`` in every PP-OCRv6 recogniser's inference.yml.
REC_HEIGHT = 48
REC_WIDTH = 320
REC_MAX_WIDTH = 3200

_MEAN = (0.485, 0.456, 0.406)   # applied to B, G, R in that order: upstream's convention
_STD = (0.229, 0.224, 0.225)


def resize_for_detector(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``DetResizeForTest`` type 0. Returns the resized page and ``[src_h, src_w, ratio_h, ratio_w]``."""
    src_h, src_w, _ = bgr.shape
    img = bgr
    if sum([src_h, src_w]) < 64:
        padded = np.zeros((max(32, src_h), max(32, src_w), img.shape[2]), np.uint8)
        padded[:src_h, :src_w, :] = img
        img = padded

    h, w, _ = img.shape
    if LIMIT_TYPE == "min":
        if min(h, w) < LIMIT_SIDE_LEN:
            ratio = float(LIMIT_SIDE_LEN) / h if h < w else float(LIMIT_SIDE_LEN) / w
        else:
            ratio = 1.0
    resize_h = int(h * ratio)
    resize_w = int(w * ratio)
    if max(resize_h, resize_w) > MAX_SIDE_LIMIT:
        ratio = float(MAX_SIDE_LIMIT) / max(resize_h, resize_w)
        resize_h, resize_w = int(resize_h * ratio), int(resize_w * ratio)
    resize_h = max(int(round(resize_h / 32) * 32), 32)
    resize_w = max(int(round(resize_w / 32) * 32), 32)

    if resize_h == h and resize_w == w:
        ratio_h, ratio_w = 1.0, 1.0
    else:
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
    return img, np.array([src_h, src_w, ratio_h, ratio_w])


def normalize_for_detector(img: np.ndarray) -> np.ndarray:
    """``NormalizeImage`` (order ``hwc``) then ``ToCHWImage``: ``(3, H, W)`` float32.

    Per channel, ``x * (1/255 / std) + (-mean / std)`` in float32 -- a multiply-add, not
    ``(x / 255 - mean) / std``, which rounds differently.
    """
    alpha = [(1.0 / 255.0) / _STD[i] for i in range(3)]
    beta = [-_MEAN[i] / _STD[i] for i in range(3)]
    planes = list(cv2.split(img))
    for c in range(3):
        planes[c] = planes[c].astype(np.float32)
        planes[c] *= alpha[c]
        planes[c] += beta[c]
    return cv2.merge(planes).transpose((2, 0, 1))


def _ordered_box(rect) -> np.ndarray:
    points = sorted(list(cv2.boxPoints(rect)), key=lambda x: x[0])
    a, d = (0, 1) if points[1][1] > points[0][1] else (1, 0)
    b, c = (2, 3) if points[3][1] > points[2][1] else (3, 2)
    return np.array([points[a], points[b], points[c], points[d]])


def crop_line(bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """``CropByPolys.get_minarea_rect_crop``: rectify one detected quad to an upright line.

    A crop at least 1.5 times taller than wide is turned a quarter, so vertical text reads
    left to right. Pixels outside the page are replicated from its edge, bicubic in between.
    """
    points = _ordered_box(cv2.minAreaRect(np.array(quad).astype(np.int32)))
    width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
    target = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    matrix = cv2.getPerspectiveTransform(points, target)
    crop = cv2.warpPerspective(bgr, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE,
                               flags=cv2.INTER_CUBIC)
    crop_h, crop_w = crop.shape[0:2]
    if crop_h * 1.0 / crop_w >= 1.5:
        crop = np.rot90(crop)
    return crop


def resize_for_recogniser(line: np.ndarray) -> np.ndarray:
    """``OCRReisizeNormImg.resize``: height 48, width kept to the line's own aspect (at least
    320), bilinear, scaled to [-1, 1], zero-padded on the right to its own target width."""
    h, w = line.shape[:2]
    max_wh_ratio = max(REC_WIDTH / REC_HEIGHT, w * 1.0 / h)
    target_w = int(REC_HEIGHT * max_wh_ratio)
    if target_w > REC_MAX_WIDTH:
        resized = cv2.resize(line, (REC_MAX_WIDTH, REC_HEIGHT))
        resized_w = target_w = REC_MAX_WIDTH
    else:
        ratio = w / float(h)
        resized_w = target_w if math.ceil(REC_HEIGHT * ratio) > target_w else int(
            math.ceil(REC_HEIGHT * ratio))
        resized = cv2.resize(line, (resized_w, REC_HEIGHT))
    resized = resized.astype("float32").transpose((2, 0, 1)) / 255
    resized -= 0.5
    resized /= 0.5
    padded = np.zeros((3, REC_HEIGHT, target_w), dtype=np.float32)
    padded[:, :, 0:resized_w] = resized
    return padded


def batch_for_recogniser(lines: list[np.ndarray]) -> np.ndarray:
    """``ToBatch``: zero-pad every line to the widest in the batch and stack them."""
    widest = max(line.shape[2] for line in lines)
    padded = [np.pad(line, ((0, 0), (0, 0), (0, widest - line.shape[2])), mode="constant",
                     constant_values=0) for line in lines]
    return np.stack(padded, axis=0).astype(dtype=np.float32, copy=False)

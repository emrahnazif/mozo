# SPDX-License-Identifier: Apache-2.0
"""RTMO's input: the frame letterboxed, centred, into a square, by an affine warp.

From MMPose v1.3.2 ``BottomupResize`` (``resize_mode='fit'``, ``use_udp=False``) and
``get_warp_matrix``. Not a resize-and-pad: upstream builds a 2x3 affine from a centre and a scale
with ``cv2.getAffineTransform`` and resamples with ``cv2.warpAffine`` (bilinear, border 114). The
picture is centred, so the grey bars are split between both sides -- unlike YOLOX's top-left
letterbox, which this model was *not* trained with at test time.

The warp is float32 throughout, as upstream's: the three anchor points are float32 arrays, and
OpenCV's fixed-point resampler takes the matrix from there. Rebuilding the same picture with
``cv2.resize`` and a pad would move pixels by the sub-pixel offset the affine carries.

Named ``preprocess.py``, not ``image.py``: ``tests/test_vendor_agreement.py`` holds every vendor's
``image.py`` to the YOLO letterbox interface, which this is not.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

__all__ = ["letterbox", "to_batch"]


def _third_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The point that makes a right angle at *b* (``_get_3rd_point``)."""
    direction = a - b
    return b + np.r_[-direction[1], direction[0]]


def _warp_matrix(center: np.ndarray, scale: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """``get_warp_matrix(center, scale, rot=0, output_size=size)``, the unrotated case.

    ``_rotate_point`` at angle 0 is a float64 matrix product with ``cos 0 = 1`` and ``sin 0 = 0``,
    which returns its input exactly, so it is not reproduced.
    """
    src_w = scale[0]
    dst_w, dst_h = size
    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * np.array((0., 0.))
    src[1, :] = center + np.array([src_w * -0.5, 0.]) + scale * np.array((0., 0.))
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + np.array([dst_w * -0.5, 0.])
    src[2, :] = _third_point(src[0, :], src[1, :])
    dst[2, :] = _third_point(dst[0, :], dst[1, :])
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def letterbox(image: np.ndarray, side: int, pad: int = 114
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Warp an ``HxWx3`` ``uint8`` frame into a ``side x side`` square, centred.

    Channel order is passed through untouched; RTMO wants BGR, and :mod:`~.predictor` flips.

    Returns:
        The square, and the float32 ``center`` and ``scale`` (in frame pixels) that map it back:
        ``frame = input / side * scale + center - scale / 2``.
    """
    img_h, img_w = image.shape[:2]
    ratio = img_w / img_h
    actual_w = min(side, side * ratio)
    actual_h = min(side, side / ratio)
    center = np.array([img_w / 2, img_h / 2], dtype=np.float32)
    scale = np.array([img_w * side / actual_w, img_h * side / actual_h], dtype=np.float32)
    matrix = _warp_matrix(center, scale, (side, side))
    canvas = cv2.warpAffine(image, matrix, (side, side), flags=cv2.INTER_LINEAR,
                            borderValue=(pad, pad, pad))
    return canvas, center, scale


def to_batch(canvas: np.ndarray) -> torch.Tensor:
    """``HxWx3`` ``uint8`` -> ``(1, 3, H, W)`` float32, unnormalised (RTMO's mean is 0, std 1).

    ``PackPoseInputs`` then ``PoseDataPreprocessor``: a contiguous CHW tensor, cast to float. The
    normalisation, ``(x - 0) / 1``, is exact and skipped; 640 and 416 are multiples of the 32-pixel
    pad divisor, so nothing is padded.
    """
    tensor = torch.from_numpy(np.ascontiguousarray(canvas)).permute(2, 0, 1).contiguous()
    return tensor.float()[None]

# SPDX-License-Identifier: BSD-3-Clause
# Derived from libfacedetection.train, Copyright (c) 2022-2026, Shiqi Yu. See LICENSE.
"""YuNet, extracted for deployment: an image in, faces with five landmarks out.

The tiny face detector behind OpenCV's ``FaceDetectorYN``, from the authors' PyTorch training
repository (libfacedetection.train), in plain torch. See ``PROVENANCE.md`` for what this derives
from and what it leaves behind, and ``README.md`` for how to drive it.

    >>> from mozo.vendors.yunet_deploy import Predictor        # doctest: +SKIP
    >>> faces = Predictor("torch-fp32.pth", "small").predict(image)  # doctest: +SKIP
    >>> faces.keypoints.shape                                  # doctest: +SKIP
    (4, 5, 2)
"""

from .config import KEYPOINTS, SPECS, Spec, get_spec
from .predictor import Faces, Predictor

__all__ = ["KEYPOINTS", "SPECS", "Faces", "Predictor", "Spec", "get_spec"]

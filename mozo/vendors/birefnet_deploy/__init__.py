# ------------------------------------------------------------------------
# BiRefNet -- Bilateral Reference for High-Resolution Dichotomous Image Segmentation
# Copyright (c) 2024 ZhengPeng. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
"""Deployment-only BiRefNet: load a published checkpoint, run an image, get an alpha matte.

Depends on ``torch``, ``torchvision`` (for ``ops.deform_conv2d`` alone), ``numpy``, ``cv2`` and
``Pillow`` -- nothing else. Upstream's remote code additionally imports ``transformers``,
``timm``, ``einops`` and ``kornia``; none survives to inference. ``tools/verify/birefnet.py``
holds every stage equal to upstream's.

Every import inside this package is relative, so the directory can be renamed or moved into
another project without edits.

Examples:
    >>> from birefnet_deploy import Predictor  # doctest: +SKIP
    >>> predictor = Predictor.from_pretrained("general", path)  # doctest: +SKIP
    >>> alpha = predictor.matte(rgb)  # doctest: +SKIP
"""

from .config import BACKBONES, MEAN, MODEL_SPECS, STD
from .image import postprocess, preprocess, refine_foreground
from .network import BiRefNet
from .predictor import Predictor

__all__ = ["BACKBONES", "BiRefNet", "MEAN", "MODEL_SPECS", "Predictor", "STD", "postprocess",
           "preprocess", "refine_foreground"]

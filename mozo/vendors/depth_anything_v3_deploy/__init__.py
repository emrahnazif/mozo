# ------------------------------------------------------------------------
# Depth Anything 3
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Deployment-only Depth Anything 3: single-image depth from three Apache-2.0 checkpoints.

Depends on ``torch``, ``torchvision``, ``numpy`` and ``cv2`` -- nothing else. ``cv2`` is not
incidental: upstream resizes with ``cv2.INTER_AREA``/``INTER_CUBIC``, which no torch resize
reproduces.

Every import inside this package is relative, so the directory can be renamed or moved into
another project without edits.

Examples:
    >>> from depth_anything_v3_deploy import Predictor  # doctest: +SKIP
    >>> predictor = Predictor.from_pretrained("small", weights=path)  # doctest: +SKIP
    >>> depth = predictor.predict(rgb)  # H'xW' depth at the processed resolution  # doctest: +SKIP
"""

from .config import MODEL_SPECS, ModelSpec, get_spec
from .net import DepthAnything3Net, fill_sky, shape_inputs
from .predictor import Predictor, postprocess, preprocess, processed_size, resize

__all__ = ["MODEL_SPECS", "DepthAnything3Net", "ModelSpec", "Predictor", "fill_sky", "get_spec",
           "postprocess", "preprocess", "processed_size", "resize", "shape_inputs"]

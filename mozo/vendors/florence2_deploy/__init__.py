# SPDX-License-Identifier: MIT
"""Florence-2, extracted for deployment: one checkpoint, several box-returning tasks.

Microsoft's DaViT + BART encoder-decoder, its default 3-beam decoding, and its post-processor for
``<OD>`` (served by mozo as ``florence2_detect``) and ``<OCR_WITH_REGION>`` (gated, not yet served).
See ``PROVENANCE.md`` for what this derives from and ``README.md`` for how to drive it.

    >>> from mozo.vendors.florence2_deploy import Florence2Predictor   # doctest: +SKIP
    >>> model = Florence2Predictor("torch-fp16.pth", "base")           # doctest: +SKIP
    >>> boxes = model.detect(image)                                    # doctest: +SKIP
"""

from .config import SPECS, VARIANTS
from .predictor import Florence2Predictor

__all__ = ["Florence2Predictor", "SPECS", "VARIANTS"]

# SPDX-License-Identifier: Apache-2.0
"""OmDet-Turbo, extracted for deployment: open-vocabulary detection.

Prompts and an image in, boxes out. See ``PROVENANCE.md`` for what this derives from and what it
leaves behind, and ``README.md`` for how to drive it.

    >>> from mozo.vendors.omdet_turbo_deploy import SPECS, Detector   # doctest: +SKIP
    >>> model = Detector("torch-fp32.pth", SPECS["tiny"])            # doctest: +SKIP
    >>> boxes, scores, labels = model(image, ["cat", "remote"])      # doctest: +SKIP
"""

from .config import SPECS, VARIANTS, Spec
from .predictor import THRESHOLD, Detector

__all__ = ["Detector", "SPECS", "Spec", "THRESHOLD", "VARIANTS"]

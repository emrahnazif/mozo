# SPDX-License-Identifier: Apache-2.0
"""Deployment-only HQ-SAM image segmentation, extracted from ``SysCV/sam-hq``.

    >>> from mozo.vendors.sam_hq_deploy import Segmenter   # doctest: +SKIP
    >>> segmenter = Segmenter("sam_hq_vit_b.pth")
    >>> found = segmenter.predict(image, boxes=[40, 60, 300, 480])

See ``README.md`` for the prompt conventions and ``PROVENANCE.md`` for what was extracted.
"""

from .predictor import Segmentation, Segmenter

__all__ = ["Segmentation", "Segmenter"]

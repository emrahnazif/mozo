# SPDX-License-Identifier: Apache-2.0
"""EoMT's inference path, extracted from ``tue-mps/eomt``.

A picture becomes a partition into classed segments: semantic segmentation (every pixel a class)
or panoptic (and each countable thing on its own), from a plain DINOv2 ViT with learned queries
in its last blocks.

    >>> from mozo.vendors.eomt_deploy import Segmenter, SPECS
    >>> segmenter = Segmenter(weights, SPECS["coco-panoptic-small"])      # doctest: +SKIP
    >>> segmenter.segment(image).class_ids                                 # doctest: +SKIP

See ``PROVENANCE.md`` for what was taken, what was left, and where this diverges.
"""

from .config import SPECS, VARIANTS, Spec
from .predictor import Segmenter, Segments

__all__ = ["SPECS", "VARIANTS", "Segmenter", "Segments", "Spec"]

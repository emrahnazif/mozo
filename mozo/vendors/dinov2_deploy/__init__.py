# SPDX-License-Identifier: Apache-2.0
"""DINOv2's inference path, extracted from ``facebookresearch/dinov2``.

A photograph becomes one vector, and photographs that look alike land close together: search,
deduplicate or cluster images by what they show. Image only: there is no text tower, so these
vectors do not share a space with CLIP's or SigLIP 2's.

    >>> from mozo.vendors.dinov2_deploy import Encoder, SPECS
    >>> encoder = Encoder(weights, SPECS["base"])                      # doctest: +SKIP
    >>> encoder.encode_image(image)                                    # doctest: +SKIP

See ``PROVENANCE.md`` for what was taken, what was left, and where this diverges.
"""

from .config import SPECS, VARIANTS, Spec
from .predictor import Encoder

__all__ = ["SPECS", "VARIANTS", "Encoder", "Spec"]

# SPDX-License-Identifier: Apache-2.0
"""Granite Embedding R2's inference path, extracted from ``transformers/models/modernbert``.

An English text becomes one vector, and texts that mean the same thing land close together. Text
only: there is no image tower, so these vectors do not share a space with CLIP's or SigLIP 2's.

    >>> from mozo.vendors.granite_embedding_r2_deploy import Encoder, SPECS
    >>> encoder = Encoder(weights, SPECS["small"])                     # doctest: +SKIP
    >>> encoder.encode_text(["how do I run a workflow?"])              # doctest: +SKIP

See ``PROVENANCE.md`` for what was taken, what was left, and where this diverges.
"""

from .config import CONTEXT, SPECS, VARIANTS, Spec
from .predictor import Encoder

__all__ = ["CONTEXT", "SPECS", "VARIANTS", "Encoder", "Spec"]

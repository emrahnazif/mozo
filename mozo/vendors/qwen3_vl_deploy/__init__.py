# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL's inference path, extracted from ``transformers/models/qwen3_vl``.

An image and a question in, text out, one token at a time.

    >>> from mozo.vendors.qwen3_vl_deploy import Generator, SPECS
    >>> model = Generator(weights, SPECS["2b"])                         # doctest: +SKIP
    >>> model.generate(rgb, "What is in this picture?").text            # doctest: +SKIP

See ``PROVENANCE.md`` for what was taken, what was left, and where this diverges.
"""

from .config import CONTEXT, SPECS, VARIANTS, Spec
from .image import MAX_PIXELS, MIN_PIXELS, PIXEL_CEILING
from .model import Sampling
from .predictor import Answer, Generator

__all__ = ["CONTEXT", "MAX_PIXELS", "MIN_PIXELS", "PIXEL_CEILING", "SPECS", "VARIANTS", "Answer",
           "Generator", "Sampling", "Spec"]

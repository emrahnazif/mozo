# SPDX-License-Identifier: Apache-2.0
"""MM-Grounding-DINO's detection path, extracted from ``transformers``' port, for LLMDet.

Open-vocabulary detection: name a thing in words and get boxes for it, with no class list and no
training. The checkpoints are LLMDet's (iSEE Laboratory): the MM-Grounding-DINO architecture,
fine-tuned with a language model's supervision. See ``PROVENANCE.md`` for what was taken, what was
left, and where this diverges.

    >>> from mozo.vendors.mm_grounding_dino_deploy import Predictor, SPECS
    >>> model = Predictor(weights, SPECS["llmdet-tiny"])   # doctest: +SKIP
    >>> model(image, ["a cat", "a laptop"])                 # doctest: +SKIP
"""

from .boxes import Detection
from .config import SPECS, VARIANTS, Spec
from .predictor import SEPARATORS, Predictor, caption_for

__all__ = [
    "SEPARATORS", "SPECS", "VARIANTS", "Detection", "Predictor", "Spec", "caption_for",
]

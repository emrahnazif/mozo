# SPDX-License-Identifier: Apache-2.0
"""mozo's HQ-SAM adapter: a click or a box in, PixelFlow detections out.

HQ-SAM is SAM with one more output token, trained on 44K finely annotated masks, whose mask is
built from an early encoder feature as well as the final one. What that buys is fine structure:
the spokes of a bicycle, the legs of an insect, the rails of a chair -- the thin parts SAM and
SAM 2 round off into a blob. The measured gain over SAM 2.1 on HQ-SAM's four fine-structure
validation sets is on the family's documentation page.

Both its code and its published weights are Apache-2.0. The weights are SAM's (Meta, Apache-2.0)
or MobileSAM's TinyViT (Apache-2.0) with HQ-SAM's added parameters on top.
"""

from __future__ import annotations

from .. import registry
from ..vendors import sam_hq_deploy
from ._promptable import PromptablePredictor

__all__ = ["SamHqPredictor"]


class SamHqPredictor(PromptablePredictor):
    """Promptable segmentation on HQ-SAM.

    Each mask is a SAM candidate plus the HQ token's correction, upstream's default. With
    ``multimask_output`` on, upstream keeps only the best of the three candidates; mozo returns
    all three, ranked, as every promptable family here does -- the first row is exactly
    upstream's answer.

    Examples:
        >>> model = SamHqPredictor("tiny")                               # doctest: +SKIP
        >>> found = model.predict("chair.jpg", boxes=[40, 60, 300, 480],
        ...                       multimask_output=False)                # doctest: +SKIP
    """

    FAMILY = "sam_hq"
    EXECUTES = registry.EXECUTES["sam_hq"]
    DISPLAY = "HQ-SAM"
    VENDOR = sam_hq_deploy
    #: Smallest first: it is the default, and it is the one that runs on a CPU.
    VARIANTS = ("tiny", "base", "large", "huge")

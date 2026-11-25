# SPDX-License-Identifier: Apache-2.0
"""RTMO, extracted for deployment: a frame in, people with boxes and 17 joints out.

One-stage multi-person pose estimation from MMPose, in plain torch -- no mmcv, mmengine or mmpose.
See ``PROVENANCE.md`` for what this derives from and what it leaves behind, and ``README.md`` for how
to drive it.

    >>> from mozo.vendors.rtmo_deploy import Predictor        # doctest: +SKIP
    >>> people = Predictor("torch-fp32.pth", "small").predict(frame)  # doctest: +SKIP
    >>> people.keypoints.shape                                 # doctest: +SKIP
    (5, 17, 2)
"""

from .config import KEYPOINTS, SPECS, Spec, get_spec
from .predictor import People, Predictor

__all__ = ["KEYPOINTS", "SPECS", "People", "Predictor", "Spec", "get_spec"]

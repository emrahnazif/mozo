"""Deployment-only DEIM-D-FINE: build a released variant, load its checkpoint, run inference.

Depends on ``torch``, ``torchvision``, ``numpy`` and ``pillow``, nothing else. Every import inside
the package is relative, so it imports nothing from mozo and no other vendor.

    >>> from mozo.vendors.deim_deploy import load, postprocess, preprocess  # doctest: +SKIP
    >>> model = load("small", "deim_dfine_hgnetv2_s_coco_120e.pth")       # doctest: +SKIP
    >>> batch, size = preprocess(rgb)                                     # doctest: +SKIP
    >>> labels, boxes, scores = postprocess(model(batch), size)           # doctest: +SKIP
"""

from .config import SPECS, Spec, get_spec
from .model import DEIM, build, load
from .postprocess import postprocess
from .preprocess import preprocess

__all__ = ["DEIM", "SPECS", "Spec", "build", "get_spec", "load", "postprocess", "preprocess"]

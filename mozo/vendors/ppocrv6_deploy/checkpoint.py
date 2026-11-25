# SPDX-License-Identifier: Apache-2.0
"""Loading mozo's fused checkpoint into the two networks.

PaddlePaddle publishes the detector and the recogniser as two repositories each. mozo publishes a
variant as one download, so ``tools/fetch/ppocrv6.py`` puts both state dictionaries in one file,
tensors untouched; this is the reading half of that agreement.
"""

from __future__ import annotations

__all__ = ["DETECTOR", "RECOGNISER", "load"]

from pathlib import Path

import torch

from .config import Spec
from .det import TextDetector
from .rec import TextRecogniser

#: The two keys ``tools/fetch/ppocrv6.py`` writes.
DETECTOR = "detector"
RECOGNISER = "recogniser"


def load(path: Path | str, spec: Spec) -> tuple[TextDetector, TextRecogniser]:
    """Build both networks for ``spec`` and fill them, strictly, from the file at ``path``.

    The keys are upstream's own, so nothing is renamed. The load is strict: a key named
    differently cannot be left at its initialisation unnoticed.

    Raises:
        ValueError: If the file is not a fused checkpoint, or its recogniser has another
            alphabet than ``spec`` -- which means another variant's weights.
    """
    blob = torch.load(path, map_location="cpu", weights_only=True)
    missing = [key for key in (DETECTOR, RECOGNISER) if key not in blob]
    if missing:
        raise ValueError(f"{path} is not a PP-OCRv6 checkpoint: no {' or '.join(missing)} in it.")

    classes = spec.recogniser.classes
    head = blob[RECOGNISER].get("head.head.weight", blob[RECOGNISER].get("head.fc2.weight"))
    if head is None or head.shape[0] != classes:
        found = "no classifier" if head is None else f"a {head.shape[0]}-class classifier"
        raise ValueError(
            f"{path} holds {found}, and this variant reads {classes} classes. "
            "This is another variant's checkpoint.")

    detector = TextDetector(spec.detector)
    detector.load_state_dict(blob[DETECTOR])
    recogniser = TextRecogniser(spec.recogniser)
    recogniser.load_state_dict(blob[RECOGNISER])
    return detector.eval(), recogniser.eval()

# SPDX-License-Identifier: Apache-2.0
"""mozo's Florence-2 detection adapter: an image in, boxes named in the model's own words out.

Florence-2 is one checkpoint that does a dozen tasks, picked by a task token in its prompt. This
family is its ``<OD>`` task: "locate the objects with category name in the image". The model
*writes* its answer -- a label, then four location tokens, then the next label -- so there is no
fixed class list: ``class_name`` is whatever it wrote ("human face", "glasses", "table") and
``class_id`` is ``None``.

**The confidence is upstream's own ``<OD>`` score**, not a classifier's probability: ``exp`` of the
mean log-probability of the box's four location tokens on the beam that wrote them. It says how
sure the decoder was of the coordinates once it had chosen the label. On the fixture it runs from
0.10 to 0.60; it is not comparable with another detector's score, and every box upstream returns is
kept unless you pass ``threshold``.

**Decoding is upstream's default and is not configurable**: 3-beam search that never repeats a
3-token sequence, up to 1,024 tokens. Deterministic: the same image gives the same boxes.

Code and weights are MIT (Microsoft).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
import torch

from .. import registry
from ..image import load_image
from ..runtimes import get_default_device, select_runtime
from ..vendors.florence2_deploy import VARIANTS, Florence2Predictor
from ..weights import artifacts, resolve, revision_of

try:
    import pixelflow as pf
except ImportError:
    raise ImportError("PixelFlow is not installed. Install it with: pip install pixelflow") from None

__all__ = ["Florence2DetectPredictor"]

#: The family this adapter serves, as the registry and the manifest name it.
FAMILY = "florence2_detect"

_log = logging.getLogger(__name__)

_DTYPES = {"torch-fp32": torch.float32, "torch-fp16": torch.float16}


class Florence2DetectPredictor:
    """Florence-2's object detection: every object it can name, boxed.

    Args:
        variant: ``base`` (0.23 B), ``large`` (0.77 B), or their fine-tuned ``base-ft`` and
            ``large-ft``, which Microsoft tuned on downstream tasks including COCO detection and
            which name COCO's classes.
        device: Where to run. Defaults to the best available.
        runtime: ``auto``, ``torch-fp32`` (exact) or ``torch-fp16`` (CUDA only).
        checkpoint_path: Your own checkpoint in the published layout, instead of the published
            weights. Run in float32.
        revision: Which published revision to use. Defaults to the newest.

    Attributes:
        variant: The variant in use.
        device: Where it runs.
        runtime: Which runtime is executing.
        revision: The published revision in use, or ``None`` for your own checkpoint.

    Examples:
        >>> model = Florence2DetectPredictor("base")             # doctest: +SKIP
        >>> found = model.predict("street.jpg")                  # doctest: +SKIP
        >>> found[0].class_name, found[0].confidence             # doctest: +SKIP
        ('car', 0.41)
    """

    #: The four checkpoints Microsoft publishes. ``config.py`` in the vendored package has them.
    VARIANTS = list(VARIANTS)

    #: Which runtimes this adapter can execute (torch), and which it builds at load.
    EXECUTES = registry.EXECUTES[FAMILY]
    DERIVES = registry.DERIVES[FAMILY]

    def __init__(
        self,
        variant: str = "base",
        device: str | None = None,
        *,
        runtime: str = "auto",
        checkpoint_path: str | Path | None = None,
        revision: str | None = None,
    ) -> None:
        if variant not in self.VARIANTS:
            raise ValueError(f"Unsupported variant {variant!r}. Choose from: {list(self.VARIANTS)}")
        self.variant = variant
        self.device = device or get_default_device()
        if checkpoint_path is not None:
            self.runtime, weights, self.revision = "torch-fp32", Path(checkpoint_path), None
        else:
            self.runtime = select_runtime(
                self.device, artifacts(FAMILY, variant, revision=revision), runtime,
                executes=self.EXECUTES, derives=self.DERIVES)
            # One file either way: Microsoft's float16 checkpoint, which torch-fp32 upcasts.
            weights = resolve(FAMILY, variant, "torch-fp16", revision=revision)
            self.revision = revision_of(FAMILY, variant, revision=revision)
        self._model = Florence2Predictor(weights, variant, self.device, _DTYPES[self.runtime])
        _log.info("Florence-2 %s ready on %s via %s.", variant, self.device, self.runtime)

    def predict(
        self,
        image: Union[str, Path, bytes, np.ndarray],
        threshold: float | None = None,
        labels: list[str] | None = None,
    ) -> "pf.detections.Detections":
        """Find and name every object in *image*.

        Args:
            image: A file path, encoded image bytes, or an ``HxWx3`` RGB ``uint8`` array.
            threshold: Keep only boxes whose score is at least this, in [0, 1]. ``None`` (the
                default) keeps every box the model wrote, as upstream does.
            labels: Not supported: object detection's ``labels`` renames a fixed class list, and
                Florence-2 has none -- it names what it sees in its own words.

        Returns:
            A PixelFlow ``Detections`` in the order the model wrote them (grouped by label, not
            sorted by score). Each carries ``bbox`` in the image's pixels, ``class_name`` (the
            model's words, non-ASCII characters dropped as upstream drops them),
            ``class_id = None`` and ``confidence`` (upstream's score, see the module docstring).

        Raises:
            ValueError: If ``labels`` is given, or ``threshold`` is outside [0, 1].
        """
        if labels is not None:
            raise ValueError(
                "florence2_detect has no class list to rename: Florence-2 names what it finds in "
                "its own words, which arrive as class_name. Leave labels out.")
        boxes = self._model.detect(load_image(image))
        found = pf.detections.from_arrays(
            boxes=np.array([b.box for b in boxes], dtype=float).reshape(-1, 4),
            scores=np.array([b.score for b in boxes], dtype=float),
        )
        for detection, box in zip(found, boxes):
            detection.class_name = box.label
        return found if threshold is None else found.filter_by_confidence(threshold)

# SPDX-License-Identifier: Apache-2.0
"""mozo's EoMT adapter: a picture in, a partition of it into classed segments out.

mozo's first scene segmentation. Every pixel gets a class from the model's fixed vocabulary --
"sky", "road", "wall", "person" -- and the answer is a PixelFlow ``Detections`` with one detection
per segment: a mask at the picture's size, the class, and a box around the mask. **No two masks
overlap**, which is what separates this from instance segmentation, where two people's masks can.

Two kinds of variant, one shape of answer:

- **Panoptic** (``coco-panoptic-*``): each countable *thing* (person, car, cup) is its own
  detection; each kind of *stuff* (sky, road, wall) is one detection covering all of it.
  ``confidence`` is the class probability of the query that found the segment. Pixels no segment
  claimed ("void") belong to no detection.
- **Semantic** (``ade20k-semantic-large``): one detection per class present, every pixel in
  exactly one. ``confidence`` is ``None``: the per-pixel score the model ranks classes by is a sum
  over its queries, not a probability, and mozo does not invent a region score.

A dense class map is one line away when that is what you want::

    canvas = np.full(image.shape[:2], -1)
    for d in model.predict(image):
        canvas[d.masks[0]] = d.class_id
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
import torch

from .. import registry
from ..image import load_image
from ..labels import resolve as labels_for
from ..runtimes import TensorRTRunner, get_default_device, make_runner, select_runtime
from ..vendors.eomt_deploy import SPECS, VARIANTS, Segmenter
from ..vendors.eomt_deploy.postprocess import MASK_THRESHOLD
from ..weights import artifacts, framework_of, parts, resolve, revision_of
from ._promptable import bounds

try:
    import pixelflow as pf
except ImportError:  # pragma: no cover - depends on the install
    raise ImportError("PixelFlow is not installed. Install it with: pip install pixelflow") from None

__all__ = ["EomtPredictor"]

#: The family this adapter serves, as the registry and the manifest name it.
FAMILY = "eomt"

_log = logging.getLogger(__name__)


class EomtPredictor:
    """Semantic and panoptic segmentation.

    Args:
        variant: Which published model: ``coco-panoptic-small``, ``-base``, ``-large`` (640) or
            ``-large-1280`` (COCO's 133 classes, things split per instance), or
            ``ade20k-semantic-large`` (ADE20K's 150 scene classes).
        device: Where to run. Defaults to the best available.
        runtime: Which artifact to execute. ``auto`` picks the fastest published one.
        checkpoint_path: Your own checkpoint, in upstream's network layout, instead of the
            published weights. Its classes are then unnamed.
        revision: Which published revision to use. Defaults to the newest.

    Attributes:
        variant: The variant in use.
        device: Where it runs.
        runtime: Which artifact is executing.
        revision: The published revision in use, or ``None`` for your own checkpoint.
        labels: The class names, in class-id order, or ``None`` for your own checkpoint.

    Examples:
        >>> model = EomtPredictor("coco-panoptic-small")                    # doctest: +SKIP
        >>> [d.class_name for d in model.predict("street.jpg")]             # doctest: +SKIP
        ['person', 'person', 'car', 'road', 'sky', ...]
    """

    #: The variants carried. The vendor's ``config.py`` says why these five.
    VARIANTS = list(VARIANTS)

    #: Which runtimes this adapter can execute.
    EXECUTES = registry.EXECUTES[FAMILY]

    def __init__(
        self,
        variant: str = "coco-panoptic-small",
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
        self._spec = SPECS[variant]
        self.runtime = (
            "torch-fp32" if checkpoint_path is not None
            else select_runtime(self.device, artifacts(FAMILY, variant, revision=revision),
                                runtime, executes=self.EXECUTES,
                                derives=registry.DERIVES.get(FAMILY))
        )
        self.revision = (None if checkpoint_path is not None
                         else revision_of(FAMILY, variant, revision=revision))
        names = labels_for(FAMILY, variant, revision=revision,
                           published=checkpoint_path is None)
        self.labels = [entry["name"] for entry in names] if names else None

        framework = framework_of(self.runtime)
        network = weights = None
        if framework == "tensorrt":
            runner = TensorRTRunner(parts(FAMILY, variant, self.runtime, revision=revision)[""],
                                    self.device)
            network = _one_at_a_time(lambda pixels: runner(pixels))
        elif framework == "onnx":
            runner = make_runner(resolve(FAMILY, variant, self.runtime, revision=revision),
                                 self.runtime, device=self.device)
            network = _one_at_a_time(
                lambda pixels: tuple(torch.from_numpy(o).to(pixels.device)
                                     for o in runner(pixels.cpu().numpy())))
        else:
            weights = (Path(checkpoint_path) if checkpoint_path is not None
                       else resolve(FAMILY, variant, "torch-fp32", revision=revision))
        self._segmenter = Segmenter(weights, self._spec, device=self.device, network=network)
        _log.info("EoMT %s ready on %s via %s.", variant, self.device, self.runtime)

    def predict(self, image: Union[str, Path, bytes, np.ndarray],
                threshold: float | None = None) -> "pf.Detections":
        """Segment one picture.

        Args:
            image: A path, encoded bytes, or an ``HxWx3`` RGB array.
            threshold: Panoptic variants only: the class probability a candidate segment needs
                before segments are merged, 0.8 by default (upstream's). Raising it leaves the
                dropped segments' pixels void -- it is not the same as filtering the result,
                because the merge would have given those pixels to other segments.

        Returns:
            One detection per segment, in the model's own order, masks never overlapping.

        Raises:
            ValueError: If *threshold* is given to a semantic variant, which labels every pixel
                and has nothing to threshold; or if the image cannot be decoded.
        """
        if self._spec.task == "semantic" and threshold is not None:
            raise ValueError(f"{self.variant} labels every pixel; there is no threshold to set.")
        rgb = load_image(image)
        found = self._segmenter.segment(
            rgb, threshold=MASK_THRESHOLD if threshold is None else threshold)
        if found.scores is not None:
            return pf.detections.from_arrays(
                boxes=bounds(found.masks), scores=found.scores, class_ids=found.class_ids,
                masks=found.masks, labels=self.labels)
        # Semantic regions carry no score, which ``from_arrays`` requires; built with
        # PixelFlow's own constructor instead, the box derived the same way.
        detections = pf.Detections()
        for box, mask, class_id in zip(bounds(found.masks), found.masks, found.class_ids):
            class_id = int(class_id)
            detections.add_detection(pf.Detection(
                bbox=box.tolist(), masks=[mask], class_id=class_id,
                class_name=self.labels[class_id] if self.labels else None))
        return detections


def _one_at_a_time(run):
    """A graph built for one picture, applied to each of a batch (a semantic picture's windows)."""
    def network(pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = [run(pixels[i:i + 1].contiguous()) for i in range(len(pixels))]
        return torch.cat([o[0] for o in outputs]), torch.cat([o[1] for o in outputs])
    return network

# SPDX-License-Identifier: Apache-2.0
"""mozo's OmDet-Turbo adapter: phrases and an image in, PixelFlow detections out.

The third open-vocabulary family, answering the same question through the same endpoint as OWLv2
and Grounding DINO: name things in words, get boxes for them. Where it sits between them is speed.
OmDet-Turbo was built to be the real-time one -- a Swin-T, an RT-DETR encoder, and a light decoder
that mixes the words into its queries -- and its text half depends on the words alone, so it runs
once per vocabulary and every image after that pays for the image path only.

**Two readings of the prompts.** Each prompt becomes a class embedding that detections are scored
against, and all of them together become a task sentence -- ``"Detect cat, remote."`` -- that the
decoder's queries attend to. The sentence is upstream's default; this adapter does not take a
custom one.

**Its weights are Apache-2.0**, as Om AI Lab states on the model card, and so is its code.

**Boxes only, no masks.** Pair it with SAM 2 or EdgeTAM if you want a mask.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Union

import numpy as np
import torch

from .. import registry
from ..image import load_image
from ..runtimes import TensorRTRunner, get_default_device, make_runner, select_runtime
from ..vendors.omdet_turbo_deploy import SPECS, THRESHOLD, VARIANTS, Detector
from ..vendors.omdet_turbo_deploy.export import MAX_CLASSES, example
from ..weights import artifacts, framework_of, parts, resolve

try:
    import pixelflow as pf
except ImportError:  # pragma: no cover - depends on the install
    raise ImportError("PixelFlow is not installed. Install it with: pip install pixelflow") from None

__all__ = ["OmDetTurboPredictor"]

#: The family this adapter serves, as the registry and the manifest name it.
FAMILY = "omdet_turbo"

_log = logging.getLogger(__name__)


class OmDetTurboPredictor:
    """Open-vocabulary detection: name things, get boxes for them.

    Args:
        variant: Which published model. Om AI Lab publishes one, ``tiny`` (Swin-T).
        device: Where to run. Defaults to the best available.
        runtime: Which artifact to execute. ``auto`` picks the fastest published one for the
            device; see the family's documentation for what each measures.
        revision: Which published revision to use. Defaults to the newest.

    Attributes:
        variant: The variant in use.
        device: Where it runs.
        runtime: Which artifact is executing.

    Examples:
        >>> model = OmDetTurboPredictor()                                 # doctest: +SKIP
        >>> found = model.predict("street.jpg", ["taxi", "cyclist"])      # doctest: +SKIP
        >>> found[0].class_name                                           # doctest: +SKIP
        'taxi'
    """

    #: Om AI Lab publishes one checkpoint. See ``PROVENANCE.md``.
    VARIANTS = list(VARIANTS)

    #: Frameworks this adapter executes, and runtimes it builds at load time. See the registry.
    EXECUTES = registry.EXECUTES[FAMILY]
    DERIVES = registry.DERIVES[FAMILY]

    #: The vocabulary size a TensorRT engine is tuned for. It accepts 1 to ``MAX_CLASSES``.
    TENSORRT_CLASSES = 80

    def __init__(
        self,
        variant: str = "tiny",
        device: str | None = None,
        *,
        runtime: str = "auto",
        revision: str | None = None,
    ) -> None:
        if variant not in self.VARIANTS:
            raise ValueError(f"Unsupported variant {variant!r}. Choose from: {list(self.VARIANTS)}")

        self.variant = variant
        self.device = device or get_default_device()
        self.runtime = select_runtime(
            self.device, artifacts(FAMILY, variant, revision=revision), runtime,
            executes=self.EXECUTES, derives=self.DERIVES)
        spec = SPECS[variant]

        # A graph replaces the image path only; the text tower always comes from the checkpoint,
        # which is why a graph runtime fetches it too (``registry.NEEDS``).
        forward = None
        framework = framework_of(self.runtime)
        if framework == "tensorrt":
            width = spec.class_embed_dim
            engine = TensorRTRunner(
                parts(FAMILY, variant, self.runtime, revision=revision)[""], self.device,
                shapes={"classes": ((1, 1, width), (self.TENSORRT_CLASSES, 1, width),
                                    (MAX_CLASSES, 1, width)),
                        "task": ((1, 1, width), (spec.text.context_length, 1, width),
                                 (spec.text.context_length, 1, width))},
                probe=example(spec, self.device))
            forward = engine
        elif framework == "onnx":
            graph = make_runner(resolve(FAMILY, variant, self.runtime, revision=revision),
                                self.runtime, device=self.device)

            def forward(pixels, classes, task):
                boxes, logits = graph(pixels.cpu().numpy(), classes=classes.cpu().numpy(),
                                      task=task.cpu().numpy())
                return torch.from_numpy(boxes), torch.from_numpy(logits)

        weights = resolve(FAMILY, variant, "torch-fp32", revision=revision)
        self._detector = Detector(weights, spec, device=self.device, forward=forward)
        _log.info("OmDet-Turbo %s ready on %s via %s.", variant, self.device, self.runtime)

    def predict(
        self,
        image: Union[str, Path, bytes, np.ndarray],
        text: Union[str, Sequence[str]],
        threshold: float = THRESHOLD,
    ) -> "pf.detections.Detections":
        """Find every instance of each phrase in ``text``.

        Args:
            image: A file path, encoded image bytes, or an ``HxWx3`` RGB ``uint8`` array.
            text: One phrase -- ``"taxi"``, ``"yellow school bus"`` -- or several, as a sequence.
                Each is scored separately and may be up to 75 tokens. Together they also form the
                task sentence ``"Detect taxi, yellow school bus."``, which the model reads and
                which is cut at 77 tokens, as upstream cuts it -- around twenty COCO-length names.
                The text is encoded once per distinct list of phrases and reused after that.
            threshold: Confidence floor. Upstream's default is 0.3.

        Returns:
            A PixelFlow ``Detections`` carrying a box and a score per instance, with
            ``class_name`` naming the phrase that found it and ``class_id`` its index in
            ``text``, ordered best first.

            Phrases do not compete: every candidate is scored against every phrase, so one box
            can come back under two names. Overlapping boxes under the same name are suppressed
            (IoU 0.5), as upstream does.

        Raises:
            ValueError: If no phrase is given, if one is empty, or if one is longer than the
                text tower's 75 tokens.
        """
        prompts = [text] if isinstance(text, str) else list(text)
        if not prompts:
            raise ValueError("OmDet-Turbo needs a phrase to look for; no text was given.")
        if any(not phrase.strip() for phrase in prompts):
            raise ValueError("OmDet-Turbo needs a phrase to look for; text was empty.")

        boxes, scores, labels = self._detector(load_image(image), prompts, threshold=threshold)
        return pf.detections.from_arrays(boxes=boxes.numpy(), scores=scores.numpy(),
                                         class_ids=labels.numpy(), labels=prompts)

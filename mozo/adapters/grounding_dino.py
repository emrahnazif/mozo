# SPDX-License-Identifier: Apache-2.0
"""mozo's Grounding DINO adapter: phrases and an image in, PixelFlow detections out.

The second open-vocabulary family here, and it answers the same question OWLv2 does through the
same endpoint: name a thing in words, get boxes for it, with no class list and no training. The
two are substitutable, which is the whole reason ``open_vocabulary_detection`` is one task rather
than one per family.

Where it differs from OWLv2 is inside. OWLv2 embeds each phrase separately and compares them to
patch embeddings; Grounding DINO fuses text into the image features six times over and lets the
decoder attend back to the words, which is why it reads a phrase like ``"the man on the left"``
rather than treating it as a bag of words. It costs more to run and it is the one to reach for
when the prompt is a description rather than a noun.

**Its weights are Apache-2.0**, as the authors state on the HuggingFace repositories that serve
them. The GitHub project carries the licence for the code and says nothing about the checkpoints;
see the ``NOTICE`` published beside them.

**Boxes only, no masks.** Pair it with SAM 2 or EdgeTAM if you want a mask: feed a box from here
in as a prompt there.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Union

import numpy as np

from .. import registry
from ..image import load_image
from ..runtimes import TensorRTRunner, free_device_memory, get_default_device, select_runtime
from ..vendors.grounding_dino_deploy import SPECS, VARIANTS, Predictor
from ..vendors.grounding_dino_deploy.checkpoint import build
from ..vendors.grounding_dino_deploy.export import example, export, profile, require_exporter
from ..weights import _ENGINES_SUFFIX, artifacts, framework_of, resolve

try:
    import pixelflow as pf
except ImportError:
    raise ImportError("PixelFlow is not installed. Install it with: pip install pixelflow") from None

__all__ = ["GroundingDinoPredictor"]

#: The family this adapter serves, as the registry and the manifest name it.
FAMILY = "grounding_dino"


_log = logging.getLogger(__name__)


class GroundingDinoPredictor:
    """Open-vocabulary detection: describe a thing, get boxes for it.

    Args:
        variant: Which published model. ``tiny`` is Swin-T and 82% of upstream's own downloads;
            ``base`` is Swin-B and 8.3 box AP better on COCO zero-shot, for 35% more weights.
        device: Where to run. Defaults to the best available.
        runtime: Which artifact to execute. ``auto`` picks the fastest published one;
            ``tensorrt-fp16`` must be asked for, and needs TensorRT 11 or later.
        input_size: Fit every image into this square, padded, instead of upstream's resize to a
            short side of 800. ``None`` -- the default -- is upstream's resize, except on
            ``tensorrt-fp16``, which needs one fixed shape and uses 800. A smaller square is
            faster and finds less: see the measured trade-off in the family's documentation.
        checkpoint_path: Your own checkpoint, instead of the published weights.
        revision: Which published revision to use. Defaults to the newest.

    Attributes:
        variant: The variant in use.
        device: Where it runs.
        runtime: Which artifact is executing.
        input_size: The square images are fitted into, or ``None`` for upstream's resize.

    Examples:
        >>> model = GroundingDinoPredictor()                            # doctest: +SKIP
        >>> found = model.predict("street.jpg", "taxi")                 # doctest: +SKIP
        >>> found[0].class_name                                         # doctest: +SKIP
        'taxi'
        >>> many = model.predict("street.jpg", ["taxi", "a cyclist"])   # doctest: +SKIP
    """

    #: Upstream publishes exactly two checkpoints. See ``PROVENANCE.md``.
    VARIANTS = list(VARIANTS)

    #: Frameworks this adapter executes, and runtimes it builds at load time. See the registry.
    EXECUTES = registry.EXECUTES[FAMILY]
    DERIVES = registry.DERIVES[FAMILY]

    #: The square a TensorRT engine runs at when no ``input_size`` is given. Measured on 500 COCO
    #: images: 1.6 AP under upstream's resize for ``tiny`` and 1.8 for ``base``, at the smallest
    #: square of the three measured and so the fastest. See ``docs/models/grounding-dino.md``.
    TENSORRT_SIZE = 800

    #: The caption length a TensorRT engine is tuned for. It accepts any up to the model's 256.
    TENSORRT_TOKENS = 64

    def __init__(
        self,
        variant: str = "tiny",
        device: str | None = None,
        *,
        runtime: str = "auto",
        input_size: int | None = None,
        checkpoint_path: str | Path | None = None,
        revision: str | None = None,
    ) -> None:
        # Checked even with your own checkpoint: the variant selects the backbone geometry here,
        # so a wrong one is not a mislabelling but a strict load that cannot succeed.
        if variant not in self.VARIANTS:
            raise ValueError(f"Unsupported variant {variant!r}. Choose from: {list(self.VARIANTS)}")

        self.variant = variant
        self.device = device or get_default_device()
        # What is runnable is the manifest's to declare, and what is *executable* is this
        # adapter's. ``select_runtime`` reads both and raises with the available names.
        self.runtime = (
            "torch-fp32" if checkpoint_path is not None
            else select_runtime(
                self.device,
                artifacts(FAMILY, variant, revision=revision),
                runtime,
                executes=self.EXECUTES,
                derives=self.DERIVES,
            )
        )

        if input_size is not None and not 64 <= input_size <= 2048:
            raise ValueError(f"input_size must be between 64 and 2048 pixels, not {input_size}")
        tensorrt = framework_of(self.runtime) == "tensorrt"
        # A graph runs at one shape, so TensorRT always letterboxes; torch only when asked.
        self.input_size = self.TENSORRT_SIZE if tensorrt and input_size is None else input_size

        # Neither derived runtime has a file of its own: ``torch-fp16`` is the checkpoint under
        # autocast, and a TensorRT engine is exported from it.
        weights = (Path(checkpoint_path) if checkpoint_path
                   else resolve(FAMILY, variant, "torch-fp32", revision=revision))
        if tensorrt:
            engine = self._engine(weights)
            self._predictor = Predictor(None, SPECS[variant], device=self.device,
                                        input_size=self.input_size,
                                        forward=engine)
        else:
            self._predictor = Predictor(weights, SPECS[variant], device=self.device,
                                        autocast=self.runtime == "torch-fp16",
                                        input_size=self.input_size)
        _log.info("Grounding DINO %s ready on %s via %s.", variant, self.device, self.runtime)

    def _engine(self, weights: Path) -> TensorRTRunner:
        """The TensorRT engine for this variant at ``input_size``, built if this GPU has none.

        Kept beside the checkpoint in ``torch-fp32.engines/``, so ``weights.remove`` takes it
        with the model. When it has to be built, the graph is exported from the checkpoint --
        under fp16 autocast, so the engine runs ``torch-fp16``'s precision -- and deleted after.
        """
        spec, device, size = SPECS[self.variant], self.device, self.input_size

        def write(path: Path) -> None:
            # Called only when this GPU has no engine yet, so a machine that has one needs no
            # exporter; checked before the checkpoint loads, not a minute later.
            require_exporter()
            model = build(spec, weights, device=device)
            export(model, path, size, fp16=True)
            del model
            free_device_memory()

        engines = weights.with_suffix(_ENGINES_SUFFIX)
        return TensorRTRunner(engines / f"square{size}-fp16.onnx", device, engines=engines,
                              export=write, shapes=profile(self.TENSORRT_TOKENS, spec.max_text_len),
                              probe=example(size, device))

    def predict(
        self,
        image: Union[str, Path, bytes, np.ndarray],
        text: Union[str, Sequence[str]],
        threshold: float | None = None,
    ) -> "pf.detections.Detections":
        """Find every instance of each phrase in ``text``.

        Args:
            image: A file path, encoded image bytes, or an ``HxWx3`` RGB ``uint8`` array.
            text: One phrase -- ``"taxi"``, ``"a yellow school bus"`` -- or several, as a
                sequence. They are joined into the single caption the model was trained on, so
                asking for several costs one forward pass rather than several. A phrase may not
                contain ``.`` or ``?``: those separate concepts, and a prompt carrying one would
                be silently split.
            threshold: Confidence floor. Omitted, upstream's published default of 0.35 applies.
                Grounding DINO's scores run higher than OWLv2's, whose default is 0.1 -- the two
                numbers are not comparable and neither is restated at the endpoint.

        Returns:
            A PixelFlow ``Detections`` carrying a box and a score per instance, with
            ``class_name`` naming the phrase that found it and ``class_id`` its index in
            ``text``, ordered best first.

            The name is the phrase you passed, not a span decoded from the model's tokens.
            Upstream returns the latter, which can hand back ``"yellow school"`` for
            ``"a yellow school bus"``; see ``PROVENANCE.md`` for why this does not.

            Nothing is suppressed. Two prompts that describe the same thing can both find it,
            and overlapping boxes for one prompt are the model's answer rather than a failure to
            deduplicate.

        Raises:
            ValueError: If no phrase is given, if one is empty, if a phrase contains a separator,
                or if the phrases together exceed the model's 256-token budget.
        """
        prompts = [text] if isinstance(text, str) else list(text)
        if not prompts:
            raise ValueError("Grounding DINO needs a phrase to look for; no text was given.")
        if any(not phrase.strip() for phrase in prompts):
            raise ValueError("Grounding DINO needs a phrase to look for; text was empty.")

        found = self._predictor(load_image(image), prompts, box_threshold=threshold)

        # Ranked here rather than in the vendor, which returns queries in the model's own order
        # -- the right one to verify against -- while every other family in mozo hands back
        # detections best first.
        found.sort(key=lambda d: d.score, reverse=True)

        return pf.detections.from_arrays(
            boxes=np.array([d.box for d in found], dtype=np.float32).reshape(-1, 4),
            scores=np.array([d.score for d in found], dtype=np.float32),
            class_ids=np.array([d.prompt_index for d in found], dtype=np.int64),
            labels=prompts,
        )

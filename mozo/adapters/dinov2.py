# SPDX-License-Identifier: Apache-2.0
"""mozo's DINOv2 adapter: images in, one vector per image out.

mozo's first image-only embedder. It answers nothing about an image -- no boxes, no classes -- so
there is no ``predict``: ``encode_image`` is the whole of it, and over HTTP that is ``/encode``
(``/predict`` refuses it before loading anything). The vectors are a handoff, not an answer: embed
a collection once, keep the vectors in a store of your own, and find near neighbours there --
near-duplicates, similar products, clusters, a few labelled examples standing in for a class.

**Image only, and not CLIP's space.** DINOv2 learned from images alone, with no captions, so its
vectors cannot be searched by words and mean nothing against CLIP's or SigLIP 2's. What it gives
in return is likeness of the picture itself -- object, pose, texture, layout -- which a model
trained to agree with captions partly throws away.

**Only the middle of the picture is seen.** Upstream's evaluation transform scales the short side
to 256 and keeps the central 224x224, so on a 16:9 frame the outer quarter on each side never
reaches the model. That is the transform the published k-NN numbers were measured with, and the
vectors are bit-identical to it.

**An index is tied to the weights that built it.** A vector from one variant or revision means
nothing against a vector from another, so a stored index has to record :attr:`revision`.
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
from ..vendors.dinov2_deploy import SPECS, VARIANTS, Encoder
from ..weights import artifacts, framework_of, parts, resolve, revision_of

__all__ = ["Dinov2Predictor"]

#: The family this adapter serves, as the registry and the manifest name it.
FAMILY = "dinov2"

_log = logging.getLogger(__name__)


class Dinov2Predictor:
    """General-purpose image embeddings.

    Args:
        variant: Which published model: ``small``, ``base``, ``large``, ``giant``, or
            ``small-reg`` / ``base-reg``, the same backbones trained with register tokens.
        device: Where to run. Defaults to the best available.
        runtime: Which artifact to execute. ``auto`` picks the fastest published one.
        checkpoint_path: Your own checkpoint, in the authors' key layout, instead of the
            published weights.
        revision: Which published revision to use. Defaults to the newest.

    Attributes:
        variant: The variant in use.
        device: Where it runs.
        runtime: Which artifact is executing.
        revision: The published revision in use, or ``None`` for your own checkpoint.

    Examples:
        >>> model = Dinov2Predictor("base")                                 # doctest: +SKIP
        >>> model.encode_image(["a.jpg", "b.jpg"]).shape                    # doctest: +SKIP
        (2, 768)
    """

    #: The variants carried. The vendor's ``config.py`` says why these six.
    VARIANTS = list(VARIANTS)

    #: Which runtimes this adapter can execute.
    EXECUTES = registry.EXECUTES[FAMILY]

    def __init__(
        self,
        variant: str = "base",
        device: str | None = None,
        *,
        runtime: str = "auto",
        checkpoint_path: str | Path | None = None,
        revision: str | None = None,
    ) -> None:
        # Checked even with your own checkpoint: the variant selects the geometry, so a wrong one
        # is a strict load that cannot succeed rather than a mislabelling.
        if variant not in self.VARIANTS:
            raise ValueError(f"Unsupported variant {variant!r}. Choose from: {list(self.VARIANTS)}")

        self.variant = variant
        self.device = device or get_default_device()
        self.runtime = (
            "torch-fp32" if checkpoint_path is not None
            else select_runtime(self.device, artifacts(FAMILY, variant, revision=revision),
                                runtime, executes=self.EXECUTES,
                                derives=registry.DERIVES.get(FAMILY))
        )
        #: Which published revision produced the vectors, or ``None`` for your own checkpoint. A
        #: vector is only comparable against others from the same weights, so an index built from
        #: these has to record it.
        self.revision = None if checkpoint_path else revision_of(FAMILY, variant, revision=revision)
        # Every runtime shares the encoder's preprocessing and normalisation; only the backbone
        # -- pixels to class token -- changes. The torch network is built on the first encode.
        framework = framework_of(self.runtime)
        backbone = None
        if framework == "tensorrt":
            # Built from the published graph it derives from (``registry.DERIVES``), once per GPU
            # and TensorRT version. The graph's shape is fixed, so the engine's is too.
            runner = TensorRTRunner(parts(FAMILY, variant, self.runtime, revision=revision)[""],
                                    self.device)
            backbone = lambda pixels: runner(pixels)[0]  # noqa: E731
        elif framework == "onnx":
            runner = make_runner(resolve(FAMILY, variant, self.runtime, revision=revision),
                                 self.runtime, device=self.device)
            backbone = lambda pixels: torch.from_numpy(runner(pixels.cpu().numpy())[0])  # noqa: E731
        weights = None if backbone is not None else (
            Path(checkpoint_path) if checkpoint_path
            else resolve(FAMILY, variant, "torch-fp32", revision=revision))
        self._encoder = Encoder(weights, SPECS[variant], device=self.device, backbone=backbone)
        _log.info("DINOv2 %s ready on %s via %s.", variant, self.device, self.runtime)

    def encode_image(
        self, image: Union[str, Path, bytes, np.ndarray, Sequence]
    ) -> "np.ndarray":
        """Return ``(N, D)`` L2-normalised vectors for one image or many, in the order given.

        ``D`` is 384 (small), 768 (base), 1024 (large) or 1536 (giant). Always two-dimensional,
        including for a single image. A dot product between two rows is their cosine similarity:
        compare it against other cosines, or a threshold you calibrate on your own data.

        Each image runs through the network on its own, so a vector does not depend on which
        other images it was encoded beside.

        Raises:
            ValueError: If no image is given, or one cannot be decoded.
        """
        batch = image if isinstance(image, (list, tuple)) else [image]
        return self._encoder.encode_image([load_image(item) for item in batch]).numpy()

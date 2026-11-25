# SPDX-License-Identifier: Apache-2.0
"""mozo's PP-OCRv6 adapter: an image in, located strings out.

The same contract as EasyOCR's: detections carry ``text``, never ``class_id`` or ``class_name``
(OCR reads content, it does not pick a class from a vocabulary), the four corners as read go in
``segments``, their axis-aligned hull in ``bbox``.

**A variant is a size, not a language.** PP-OCRv6's medium and small recognisers each read one
18,708-character alphabet covering about 50 languages (Chinese, Japanese, Latin scripts, Korean,
Cyrillic, Greek, Arabic and more); tiny reads 6,904 characters and no Japanese kana. No language
setting exists, and none is invented here.

**The pipeline is PaddleOCR's documented one**, with its document preprocessor and its text-line
orientation classifier off -- the configuration every PP-OCRv6 model card's Python example uses.
Those are separate networks and not part of this family. See the vendor's ``PROVENANCE.md``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
import torch

from .. import registry
from ..image import load_image
from ..runtimes import OnnxRunner, TensorRTRunner, get_default_device, select_runtime
from ..vendors.ppocrv6_deploy import SPECS, VARIANTS, Reader
from ..vendors.ppocrv6_deploy.image import REC_HEIGHT, REC_MAX_WIDTH, REC_WIDTH
from ..vendors.ppocrv6_deploy.predictor import REC_BATCH
from ..weights import artifacts, framework_of, parts, resolve

try:
    import pixelflow as pf
except ImportError:
    raise ImportError("PixelFlow is not installed. Install it with: pip install pixelflow") from None

__all__ = ["PPOCRv6Predictor"]

#: The family this adapter serves, as the registry and the manifest name it.
FAMILY = "ppocrv6"

_log = logging.getLogger(__name__)

#: The shapes a TensorRT engine accepts, ``(min, preferred, max)``: a page up to 2048 pixels a
#: side after rounding to 32, and a batch of up to six lines 48 high and up to 3200 wide -- the
#: recogniser's own cap. A larger page is refused rather than run: an engine sized for torch's
#: 4000-pixel limit reserves 4.8 GB for ``medium`` and ran slower than torch at that size.
TENSORRT_SHAPES = {
    "det": {"pixels": ((1, 3, 32, 32), (1, 3, 736, 1280), (1, 3, 2048, 2048))},
    "rec": {"pixels": ((1, 3, REC_HEIGHT, REC_WIDTH), (REC_BATCH, 3, REC_HEIGHT, 2 * REC_WIDTH),
                       (REC_BATCH, 3, REC_HEIGHT, REC_MAX_WIDTH))},
}
_TENSORRT_PAGE = TENSORRT_SHAPES["det"]["pixels"][2][2]


def _onnx_call(runner):
    return lambda batch: runner(batch)[0].astype(np.float32, copy=False)


def _tensorrt_call(engine, refuse_pages: bool):
    def run(batch: np.ndarray) -> np.ndarray:
        if refuse_pages and max(batch.shape[2:]) > _TENSORRT_PAGE:
            raise ValueError(
                f"this page reaches the detector at {batch.shape[3]}x{batch.shape[2]}, and the "
                f"TensorRT engine takes at most {_TENSORRT_PAGE} a side. Downscale it, or load "
                "the model with runtime='torch-fp32' or 'onnx-fp32', which take up to 4000.")
        return engine(torch.from_numpy(batch))[0].float().cpu().numpy()
    return run


class PPOCRv6Predictor:
    """Text detection and recognition with PaddlePaddle's PP-OCRv6: find every line, read it.

    Args:
        variant: Which size. ``medium`` is PaddleOCR's default; ``small`` and ``tiny`` trade
            accuracy for speed. See :attr:`VARIANTS`.
        device: Where to run. Defaults to the best available.
        runtime: Which artifact to execute. ``auto`` picks the fastest published one.
        revision: Which published revision to use. Defaults to the newest.

    Attributes:
        variant: The variant in use.
        device: Where it runs.
        runtime: Which artifact is executing.

    Examples:
        >>> model = PPOCRv6Predictor()                       # doctest: +SKIP
        >>> [d.text for d in model.predict("sign.jpg")]      # doctest: +SKIP
        ['EXIT 42']
    """

    #: PaddlePaddle publishes three sizes of PP-OCRv6; all three are here.
    VARIANTS = list(VARIANTS)

    #: Which runtimes this adapter can execute: whatever is published, torch and the two graphs.
    EXECUTES = registry.EXECUTES.get(FAMILY)

    def __init__(
        self,
        variant: str = "medium",
        device: str | None = None,
        *,
        runtime: str = "auto",
        revision: str | None = None,
    ) -> None:
        if variant not in self.VARIANTS:
            raise ValueError(f"Unsupported variant {variant!r}. Choose from: {self.VARIANTS}")

        self.variant = variant
        self.device = device or get_default_device()
        self.runtime = select_runtime(
            self.device,
            artifacts(FAMILY, variant, revision=revision),
            runtime,
            executes=self.EXECUTES,
            derives=registry.DERIVES.get(FAMILY),
        )
        framework = framework_of(self.runtime)
        if framework == "torch":
            weights = resolve(FAMILY, variant, self.runtime, revision=revision)
            self._reader = Reader(weights, SPECS[variant], device=self.device)
        else:
            # Two graphs, published as the parts ``det`` and ``rec``; the pipeline around them is
            # the vendor's either way, so only the two network calls change.
            graphs = parts(FAMILY, variant, self.runtime, revision=revision)
            if framework == "tensorrt":
                engines = {kind: TensorRTRunner(graphs[kind], self.device,
                                                shapes=TENSORRT_SHAPES[kind])
                           for kind in ("det", "rec")}
                calls = {kind: _tensorrt_call(engine, kind == "det")
                         for kind, engine in engines.items()}
            else:
                runners = {kind: OnnxRunner(graphs[kind], device=self.device)
                           for kind in ("det", "rec")}
                calls = {kind: _onnx_call(runner) for kind, runner in runners.items()}
            self._reader = Reader(None, SPECS[variant], device=self.device,
                                  detect=calls["det"], recognise=calls["rec"])
        _log.info("PP-OCRv6 %s ready on %s via %s.", variant, self.device, self.runtime)

    def predict(
        self,
        image: Union[str, Path, bytes, np.ndarray],
    ) -> "pf.detections.Detections":
        """Find and read every line of text in ``image``.

        Args:
            image: A file path, encoded image bytes, or an ``HxWx3`` RGB ``uint8`` array.

        Returns:
            A PixelFlow ``Detections``, one per line, in PaddleOCR's order: top to bottom, and
            left to right among lines whose top-left corners are within 10 pixels in height.
            Each carries the string in ``text``, the recogniser's confidence in ``confidence``
            (the mean probability of the characters it kept, 0 to 1), the four corners in
            ``segments`` and their hull in ``bbox``. ``class_id`` and ``class_name`` are
            ``None``. A line read as empty is kept with ``text`` set to ``""`` and confidence 0.
        """
        regions = self._reader(load_image(image))
        quads = [np.asarray(region.quad, dtype=float) for region in regions]
        return pf.detections.from_arrays(
            boxes=[[quad[:, 0].min(), quad[:, 1].min(), quad[:, 0].max(), quad[:, 1].max()]
                   for quad in quads],
            scores=[region.confidence for region in regions],
            texts=[region.text for region in regions],
            segments=quads,
        )

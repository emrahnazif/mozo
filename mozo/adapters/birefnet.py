"""BiRefNet background removal and matting -- an alpha matte, and the cut-out made from it.

The network lives in :mod:`mozo.vendors.birefnet_deploy`, extracted from the authors' Hugging Face
remote code and reduced to inference; pre- and post-processing come from the vendor too, so the
pipeline is the model card's step for step -- verified bit-identical to it on CPU in float32.

Four variants, one architecture. ``general`` is the authors' general-purpose model and the one most
people use; ``lite`` swaps the Swin-L backbone for Swin-T and runs about three times faster;
``matting`` is trained on matting data and returns soft alpha along hair and fur, where the
segmentation-trained variants return an edge that is nearly binary; ``hr`` is ``general``
retrained at 2048x2048 for large photographs.

**The alpha is the network's sigmoid**, scaled to 0-255 and truncated, then resized to the image
with a bicubic filter. Unlike BEN2's default it is not stretched per image: 255 means the model was
certain, in every image.

    >>> model = BiRefNetPredictor("general")          # doctest: +SKIP
    >>> alpha = model.predict(image)                  # doctest: +SKIP
    >>> rgba = model.cutout(image, refine=True)       # doctest: +SKIP
"""

from __future__ import annotations

import logging
from typing import Union

import numpy as np

import torch

from ..image import load_image
from ..registry import DERIVES, EXECUTES
from ..runtimes import TensorRTRunner, get_default_device, make_runner, select_runtime
from ..vendors.birefnet_deploy import (MODEL_SPECS, Predictor, postprocess, preprocess,
                                       refine_foreground)
from ..weights import artifacts, framework_of, parts, resolve

_log = logging.getLogger(__name__)


class BiRefNetPredictor:
    """One loaded BiRefNet variant, ready to matte.

    Args:
        variant: ``general``, ``lite``, ``matting`` or ``hr``.
        device: Where to run. Defaults to the best device this machine has.
        runtime: Which published runtime to execute; ``"auto"`` takes the best one for the device.
        revision: Pin a published revision instead of taking the latest.

    Attributes:
        variant: The variant in use.
        runtime: The runtime actually in use.
        device: The device actually in use.
        resolution: The square side every image is resized to before the network: 1024, or 2048
            for ``hr``.
    """

    VARIANTS = tuple(MODEL_SPECS)
    FAMILY = "birefnet"

    def __init__(
        self,
        variant: str = "general",
        device: str | None = None,
        *,
        runtime: str = "auto",
        revision: str | None = None,
    ) -> None:
        if variant not in self.VARIANTS:
            raise ValueError(f"Unsupported variant {variant!r}. Choose from: {list(self.VARIANTS)}")

        self.variant = variant
        self.device = device or get_default_device()
        self.resolution = MODEL_SPECS[variant].resolution
        self.runtime = select_runtime(
            self.device, artifacts(self.FAMILY, variant, revision=revision), runtime,
            executes=EXECUTES.get(self.FAMILY), derives=DERIVES.get(self.FAMILY))

        self._predictor = self._network = None
        framework = framework_of(self.runtime)
        if framework == "torch":
            weights = resolve(self.FAMILY, variant, "torch-fp32", revision=revision)
            # ``torch-fp16`` has no file of its own: the fp32 checkpoint under fp16 autocast.
            self._predictor = Predictor.from_pretrained(
                variant, weights, device=self.device, autocast=self.runtime == "torch-fp16")
        elif framework == "tensorrt":
            # Built once per GPU and TensorRT version from the published graph it derives from.
            engine = TensorRTRunner(parts(self.FAMILY, variant, self.runtime, revision=revision)[""],
                                    self.device)
            self._network = lambda pixels: engine(pixels)[0]
        else:
            graph = make_runner(resolve(self.FAMILY, variant, self.runtime, revision=revision),
                                self.runtime, device=self.device)
            self._network = lambda pixels: torch.from_numpy(graph(pixels.numpy())[0])
        _log.info("BiRefNet %s ready on %s via %s.", variant, self.device, self.runtime)

    def _matte(self, rgb: np.ndarray) -> np.ndarray:
        """The alpha, from whichever runtime is loaded. Only the network differs between them."""
        if self._predictor is not None:
            return self._predictor.matte(rgb)
        return postprocess(self._network(preprocess(rgb, self.resolution)), rgb.shape[:2])

    def predict(self, image: Union[str, bytes, np.ndarray]) -> np.ndarray:
        """Matte *image*.

        Args:
            image: A file path, encoded bytes, or an ``HWC`` RGB array.

        Returns:
            An ``HxW`` uint8 array at the input's resolution. 255 is foreground.
        """
        return self._matte(load_image(image))

    def cutout(self, image: Union[str, bytes, np.ndarray], *, refine: bool = False) -> np.ndarray:
        """Matte *image* and composite it, background transparent.

        Args:
            image: A file path, encoded bytes, or an ``HWC`` RGB array.
            refine: Estimate unmixed foreground colours first, as the authors' ``handler.py``
                does, so a soft edge does not carry a fringe of the background it came from. The
                alpha is the same either way. Costs two full-resolution box blurs.

        Returns:
            An ``HxWx4`` uint8 RGBA array at the input's resolution.
        """
        rgb = load_image(image)
        alpha = self._matte(rgb)
        return np.dstack([refine_foreground(rgb, alpha) if refine else rgb, alpha])

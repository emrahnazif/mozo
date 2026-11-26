"""Mayaku detection and instance segmentation, on whichever runtime the host can execute.

The architecture lives in :mod:`mozo.vendors.mayaku_deploy`, extracted from ``mayaku`` 3.1.1.
The weights are the publisher's own checkpoints, from :func:`mozo.weights.resolve`. Which of them
runs the forward pass depends on what mozo publishes for the variant and what the machine can
execute -- the torch module, or the same graph as ONNX or a TensorRT engine built from it.

Only the forward pass changes between runtimes. The letterbox, the prediction decode (threshold,
top-k, class-wise NMS) and the per-instance mask network are the vendor's on every path, run where
upstream runs them: on the GPU when the raw maps come back on CUDA, else on the CPU.

Each published Mayaku checkpoint is a variant of this one family. The street models
(``street-*``) find 22 classes of road scenes -- people, faces, helmets, bicycles, e-scooters,
cars, trucks, buses, licence plates, traffic lights and signs, animals -- from street level to drone
views; **face** and **license plate** are there so footage can be blurred before it is stored or
shared. A ``-seg-`` variant adds a mask per detection to the boxes of its detection variant.

    >>> model = MayakuPredictor("street-nano")              # doctest: +SKIP
    >>> model = MayakuPredictor("street-seg-medium")        # boxes and masks  # doctest: +SKIP
    >>> detections = model.predict(image, threshold=0.4)       # doctest: +SKIP
"""

from __future__ import annotations

import logging
from typing import Any, Union

import numpy as np
import torch

from .. import registry
from ..image import load_image
from ..labels import resolve as labels_for
from ..runtimes import TensorRTRunner, get_default_device, make_runner, select_runtime
from ..vendors.mayaku_deploy import SPECS, Detector, detect, get_spec, output_names
from ..weights import artifacts, framework_of, parts, resolve

try:
    import pixelflow as pf
except ImportError:  # pragma: no cover - depends on the install
    raise ImportError("PixelFlow is not installed. Install it with: pip install pixelflow") from None

_log = logging.getLogger(__name__)

FAMILY = "mayaku"


class MayakuPredictor:
    """One loaded Mayaku variant, ready to run.

    Args:
        variant: One of ``VARIANTS`` (the vendor's ``SPECS``), e.g. ``street-nano`` for boxes or
            ``street-seg-nano`` for boxes and masks.
        device: Where to run. Defaults to the best device this machine has.
        runtime: Which published artifact to execute, e.g. ``"torch-fp32"`` or ``"onnx-fp32"``.
            ``"auto"`` takes the best one published for the device.
        labels: Class names to attach to results, overriding the published ones.
        revision: Pin a published revision instead of taking the latest.

    Attributes:
        runtime: The artifact key actually in use.
        device: The device actually in use.
    """

    VARIANTS = tuple(SPECS)

    def __init__(
        self,
        variant: str = "street-nano",
        device: str | None = None,
        *,
        runtime: str = "auto",
        labels: list[str] | None = None,
        revision: str | None = None,
    ) -> None:
        if variant not in self.VARIANTS:
            raise ValueError(f"Unsupported variant {variant!r}. Choose from: {list(self.VARIANTS)}")

        self.variant = variant
        self.device = device or get_default_device()
        self.spec = get_spec(variant)
        self.runtime = select_runtime(
            self.device, artifacts(FAMILY, variant, revision=revision), runtime,
            derives=registry.DERIVES.get(FAMILY))

        self._detector = None
        self._runner = None
        self._tensorrt = framework_of(self.runtime) == "tensorrt"
        self._outputs = output_names(self.spec.seg)
        if self.runtime.startswith("torch"):
            self._detector = Detector(
                resolve(FAMILY, variant, self.runtime, revision=revision), variant, self.device)
        elif self._tensorrt:
            # Built once per GPU and TensorRT version from the graph it derives from
            # (``registry.DERIVES``). The graph has a fixed 1x3x800x800 input, so no profile.
            self._runner = TensorRTRunner(parts(FAMILY, variant, self.runtime, revision=revision)[""],
                                          self.device)
        else:
            self._runner = make_runner(resolve(FAMILY, variant, self.runtime, revision=revision),
                                       self.runtime, device=self.device)
        # A graph carries no vocabulary, so those runtimes read the published labels.
        names = self._detector.names if self._detector else None
        self._labels = labels_for(FAMILY, variant, caller=labels, checkpoint=names,
                                  revision=revision, published=True)
        _log.info("Mayaku %s ready on %s via %s.", variant, self.device, self.runtime)

    def _forward(self, batch: torch.Tensor) -> list[torch.Tensor]:
        """Run the letterboxed batch through whichever runtime is loaded, returning the raw maps
        in the vendor's order. Graph outputs are matched by name, not position: a runtime is free
        to declare its outputs in an order of its own."""
        if self._detector is not None:
            return self._detector.forward(batch)
        if self._tensorrt:
            found = dict(zip(self._runner.outputs, self._runner(batch)))
        else:
            found = {name: torch.from_numpy(array)
                     for name, array in zip(self._runner.outputs, self._runner(batch.numpy()))}
        return [found[name] for name in self._outputs]

    def predict(
        self,
        image: Union[str, np.ndarray],
        threshold: float = 0.4,
        labels: list[str] | None = None,
    ) -> Any:
        """Detect objects in *image*.

        Args:
            image: A file path, encoded bytes, or an ``HWC`` RGB array.
            threshold: Minimum confidence to keep, ``0 <= threshold < 1``. The default, 0.4, is the
                one the checkpoints record as their own, chosen by the publisher on the street
                validation set (best F1).
            labels: Class names for this call only, overriding the adapter's.

        Returns:
            A PixelFlow ``Detections``. The confidence is the sigmoid of the winning class's logit,
            one class per location, after class-wise NMS at IoU 0.5. A seg variant carries one
            boolean mask per detection at the image's resolution; a det variant carries none.
        """
        # The torch module gets its input made on its own device, as upstream makes it; the graph
        # runtimes are handed host memory.
        boxes, scores, class_ids, masks = detect(
            load_image(image), self._forward, self.spec, threshold,
            self._detector.device if self._detector is not None else "cpu")
        if self.device == "mps" and self._detector is not None:
            torch.mps.empty_cache()
        return pf.detections.from_arrays(
            boxes=boxes.numpy(),
            scores=scores.numpy(),
            class_ids=class_ids.numpy(),
            labels=labels if labels is not None else self._labels,
            masks=None if masks is None else masks.numpy(),
        )

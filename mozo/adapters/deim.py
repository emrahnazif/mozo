"""DEIM real-time DETR detection (COCO's 80 classes), on whichever runtime the host can execute.

The architecture lives in :mod:`mozo.vendors.deim_deploy`, extracted from the authors' repository
(DEIM-D-FINE: an HGNetv2 backbone, a hybrid encoder and D-FINE's distribution-refining decoder,
trained with DEIM's matching). The weights are the authors' own checkpoints, from
:func:`mozo.weights.resolve`. Which of them runs the forward pass depends on what mozo publishes
for the variant and what the machine can execute -- the torch module, or the same graph as ONNX or
a TensorRT engine built from it.

Only the forward pass changes between runtimes. The resize and the top-k decode are the vendor's on
every path, so the answer cannot depend on which runtime produced the raw logits and boxes. Like
every DETR there is no non-maximum suppression: each of the 300 queries proposes one box, and the
best-scoring (query, class) pairs are kept.

    >>> model = DEIMPredictor("small")                            # doctest: +SKIP
    >>> detections = model.predict(image, threshold=0.4)          # doctest: +SKIP
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
from ..vendors.deim_deploy import OUTPUTS, SPECS, get_spec, load, postprocess, preprocess
from ..weights import artifacts, framework_of, parts, resolve

try:
    import pixelflow as pf
except ImportError:  # pragma: no cover - depends on the install
    raise ImportError("PixelFlow is not installed. Install it with: pip install pixelflow") from None

_log = logging.getLogger(__name__)

FAMILY = "deim"


class DEIMPredictor:
    """One loaded DEIM variant, ready to run.

    Args:
        variant: ``nano``, ``small``, ``medium``, ``large`` or ``xlarge``, trained on COCO; or
            ``xlarge-obj365``, pretrained on Objects365 and then fine-tuned on COCO, the most
            accurate.
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
        variant: str = "medium",
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
            derives=registry.DERIVES.get(FAMILY), prefers=registry.PREFERS.get(FAMILY))

        self._model = None
        self._runner = None
        self._tensorrt = framework_of(self.runtime) == "tensorrt"
        if self.runtime.startswith("torch"):
            self._model = load(variant, resolve(FAMILY, variant, self.runtime, revision=revision))
            self._model.to(self.device)
        elif self._tensorrt:
            # Built once per GPU and TensorRT version from the graph it derives from
            # (``registry.DERIVES``). The graph has a fixed 1x3x640x640 input, so no profile.
            self._runner = TensorRTRunner(parts(FAMILY, variant, self.runtime, revision=revision)[""],
                                          self.device)
        else:
            self._runner = make_runner(resolve(FAMILY, variant, self.runtime, revision=revision),
                                       self.runtime, device=self.device)
        # The checkpoints carry no class names, so every runtime reads the published labels.
        self._labels = labels_for(FAMILY, variant, caller=labels, revision=revision, published=True)
        _log.info("DEIM %s ready on %s via %s.", variant, self.device, self.runtime)

    def _forward(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        """Raw logits and boxes, left on the device that made them: the decode runs there, as
        upstream's runs where its network ran (CUDA's ``sigmoid`` is not the CPU's in the last
        bit). Graph outputs are matched by name, not position: a runtime is free to declare its
        outputs in an order of its own."""
        if self._model is not None:
            with torch.inference_mode():
                out = self._model(batch.to(self.device))
            return {name: out[name] for name in OUTPUTS}
        if self._tensorrt:
            found = dict(zip(self._runner.outputs, self._runner(batch.to(self._runner.device))))
            # The fp16 engine returns half precision; the decode runs at full width either way.
            return {name: found[name].float() for name in OUTPUTS}
        found = dict(zip(self._runner.outputs, self._runner(batch.numpy())))
        return {name: torch.from_numpy(np.asarray(found[name], np.float32)) for name in OUTPUTS}

    def predict(
        self,
        image: Union[str, np.ndarray],
        threshold: float = 0.4,
        labels: list[str] | None = None,
    ) -> Any:
        """Detect COCO objects in *image*.

        Args:
            image: A file path, encoded bytes, or an ``HWC`` RGB array.
            threshold: Minimum confidence to keep. The default, 0.4, is the one upstream's own
                inference scripts draw at.
            labels: Class names for this call only, overriding the adapter's.

        Returns:
            A PixelFlow ``Detections``. The confidence is the sigmoid of one class's logit for one
            query (focal-loss scoring, not a softmax), so classes do not compete for it, and one
            query can appear more than once with different classes when both clear the
            threshold -- upstream's decode, kept as it is.
        """
        batch, size = preprocess(load_image(image), self.spec.size)
        outputs = self._forward(batch)
        with torch.inference_mode():
            class_ids, boxes, scores = (t.cpu() for t in postprocess(
                outputs, size.to(outputs["pred_logits"].device), self.spec.num_classes,
                self.spec.num_top_queries))
        keep = scores[0] > threshold
        if self.device == "mps" and self._model is not None:
            torch.mps.empty_cache()
        return pf.detections.from_arrays(
            boxes=boxes[0][keep].numpy(),
            scores=scores[0][keep].numpy(),
            class_ids=class_ids[0][keep].numpy(),
            labels=labels if labels is not None else self._labels,
        )

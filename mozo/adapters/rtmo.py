"""RTMO multi-person pose estimation: a frame in, people with boxes and 17 joints out.

The architecture lives in :mod:`mozo.vendors.rtmo_deploy`, extracted from MMPose and reduced to
inference in plain torch. The weights come from :func:`mozo.weights.resolve`.

**One stage, no detector.** Unlike ViTPose, which is handed the boxes of the people to pose, RTMO
finds the people itself. So it is an ``object_detection`` family whose detections carry keypoints,
the same answer ``rfdetr/keypoint-preview`` gives.

    >>> people = RTMOPredictor("small").predict(frame)          # doctest: +SKIP
    >>> people[0].keypoints[0].name                              # doctest: +SKIP
    'nose'
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import numpy as np

import torch

from .. import registry
from ..image import load_image
from ..labels import resolve as labels_for
from ..runtimes import TensorRTRunner, get_default_device, make_runner, select_runtime
from ..vendors.rtmo_deploy import Predictor
from ..weights import artifacts, framework_of, parts, resolve

try:
    import pixelflow as pf
except ImportError:  # pragma: no cover - depends on the install
    raise ImportError("PixelFlow is not installed. Install it with: pip install pixelflow") from None

_log = logging.getLogger(__name__)


class RTMOPredictor:
    """One loaded RTMO variant, ready to run.

    Args:
        variant: A published variant -- ``tiny``, ``small``, ``medium`` or ``large``.
        device: Where to run. Defaults to the best device this machine has.
        runtime: Which runtime to execute -- ``"torch-fp32"``, ``"onnx-fp32"``, or on CUDA
            ``"tensorrt-fp32"``, an engine built from the published graph. ``"auto"`` takes the
            best one for the device: TensorRT on CUDA where TensorRT 11+ is installed, else
            torch.
        checkpoint_path: A checkpoint of your own (an RTMO state dict), instead of the published
            weights. The variant then names the architecture to build, and the runtime is torch.
        labels: Names to attach, overriding the published vocabulary: a list of class names (one,
            ``person``), or the published ``labels.json`` shape, which names the joints too.
        revision: Pin a published revision instead of taking the latest.

    Attributes:
        runtime: The artifact key actually in use.
        device: The device actually in use.
    """

    VARIANTS = ("tiny", "small", "medium", "large")

    def __init__(
        self,
        variant: str = "small",
        device: str | None = None,
        *,
        runtime: str = "auto",
        checkpoint_path: str | Path | None = None,
        labels: list | None = None,
        revision: str | None = None,
    ) -> None:
        if variant not in self.VARIANTS and checkpoint_path is None:
            raise ValueError(f"Unsupported variant {variant!r}. Choose from: {list(self.VARIANTS)}")

        self.variant = variant
        self.device = device or get_default_device()

        if checkpoint_path is None:
            self.runtime = select_runtime(
                self.device, artifacts("rtmo", variant, revision=revision), runtime,
                derives=registry.DERIVES["rtmo"])
        else:
            # A checkpoint mozo did not publish: the architecture is known, the graph is not.
            self.runtime = "torch-fp32"

        # The checkpoint is loaded on every runtime: a graph replaces the dense network, and the
        # DCC -- which runs on however many people NMS kept, so has no fixed shape to export --
        # still runs in torch from it (``registry.NEEDS``).
        weights = (Path(checkpoint_path) if checkpoint_path
                   else resolve("rtmo", variant, "torch-fp32", revision=revision))
        forward = None
        if framework_of(self.runtime) == "tensorrt":
            # Built once per GPU and TensorRT version from the published graph it derives from
            # (``registry.DERIVES``). Torch tensors on the device in and out.
            forward = TensorRTRunner(
                parts("rtmo", variant, self.runtime, revision=revision)[""], self.device)
        elif not self.runtime.startswith("torch"):
            runner = make_runner(resolve("rtmo", variant, self.runtime, revision=revision),
                                 self.runtime, device=self.device)

            # By name, not position: the exporter names the outputs after what ``dense`` returns,
            # so the mapping travels with the artifact.
            order = [runner.outputs.index(name)
                     for name in ("scores", "boxes", "visibility", "pose")]

            def forward(batch: torch.Tensor) -> tuple[torch.Tensor, ...]:
                raw = runner(batch.numpy())
                return tuple(torch.from_numpy(raw[index]) for index in order)
        self._predictor = Predictor(weights, variant, device=self.device, forward=forward)

        self._labels = labels_for(
            "rtmo", variant, caller=labels, revision=revision, published=checkpoint_path is None)
        if self._labels is None:
            _log.warning(
                "[mozo] no names for this checkpoint. Detections will carry class_id 0 and "
                "class_name=None, joints an id and name=None. Pass labels=[...] to name them."
            )
        _log.info("RTMO %s ready on %s via %s.", variant, self.device, self.runtime)

    def predict(
        self,
        image: Union[str, np.ndarray],
        threshold: float = 0.5,
        labels: list | None = None,
    ) -> Any:
        """Find everyone in *image*, with their joints.

        Args:
            image: A file path, encoded bytes, or an ``HWC`` RGB array.
            threshold: The person score a candidate must exceed. Applied before NMS, as upstream
                applies its own (0.1, its evaluation setting): a candidate below it can never
                suppress one above, so the people returned at 0.5 are those returned at 0.1 that
                score above 0.5, with the same boxes and scores. Their joints can differ in the
                last float bits, because the joint decoder runs once over however many people
                survive, and that is upstream's behaviour too.
            labels: Names for this call only, overriding the adapter's.

        Returns:
            A PixelFlow ``Detections``, best first: one row per person, class 0 (``person``), the
            box, the person score, and 17 COCO keypoints as ``(x, y, visibility)``. Visibility is
            the model's own estimate that the joint can be seen; a hidden joint still gets a
            position, so filter on it before drawing.
        """
        people = self._predictor.predict(load_image(image), threshold)
        joints = np.concatenate([people.keypoints, people.visibility[..., None]], axis=-1)
        return pf.detections.from_arrays(
            boxes=people.boxes,
            scores=people.scores,
            class_ids=np.zeros(len(people.scores), dtype=np.int64),
            keypoints=joints,
            labels=labels if labels is not None else self._labels,
        )

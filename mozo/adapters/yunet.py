"""YuNet face detection: an image in, faces with five landmarks out.

The architecture lives in :mod:`mozo.vendors.yunet_deploy`, extracted from the authors' PyTorch
training repository and reduced to inference. The weights come from :func:`mozo.weights.resolve`.

**At the image's own size.** Nothing is resized: the network runs on the picture padded to a
multiple of 32, as upstream evaluates it. That is what makes it fast on small images and able to
find faces a dozen pixels wide on large ones -- and why a face much larger than about 300 px, the
range it was trained on, is missed: shrink a close-up selfie before asking.

    >>> faces = YuNetPredictor("small").predict(photo)          # doctest: +SKIP
    >>> faces[0].keypoints[2].name                               # doctest: +SKIP
    'nose_tip'
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import numpy as np

from .. import registry
from ..image import load_image
from ..labels import resolve as labels_for
from ..runtimes import TensorRTRunner, get_default_device, make_runner, select_runtime
from ..vendors.yunet_deploy import Predictor
from ..weights import artifacts, framework_of, parts, resolve

try:
    import pixelflow as pf
except ImportError:  # pragma: no cover - depends on the install
    raise ImportError("PixelFlow is not installed. Install it with: pip install pixelflow") from None

_log = logging.getLogger(__name__)


def detections(boxes: np.ndarray, scores: np.ndarray, keypoints: np.ndarray, labels) -> Any:
    """Faces as PixelFlow ``Detections``: class 0, and five landmarks that carry no score.

    YuNet predicts where each landmark is, not how sure it is, so each ``KeyPoint`` gets
    ``confidence=None`` -- PixelFlow's own "no score", which annotators always draw -- rather than
    a number the model never produced. ``from_arrays`` wants a score column, so it is handed
    zeros and they are cleared after; that way PixelFlow still resolves the landmark names from
    *labels*, in whichever format they came.
    """
    columns = np.zeros(keypoints.shape[:2] + (1,), dtype=keypoints.dtype)
    found = pf.detections.from_arrays(
        boxes=boxes, scores=scores, class_ids=np.zeros(len(scores), dtype=np.int64),
        keypoints=np.concatenate([keypoints, columns], axis=-1), labels=labels)
    for face in found:
        for point in face.keypoints:
            point.confidence = None
    return found


class YuNetPredictor:
    """One loaded YuNet variant, ready to run.

    Args:
        variant: ``small`` (upstream's ``yunet_s``, the model OpenCV's ``FaceDetectorYN`` ships)
            or ``base`` (``yunet_n``, a little larger and better on small faces).
        device: Where to run. Defaults to the best device this machine has.
        runtime: Which runtime to execute -- ``"torch-fp32"``, or a published graph.
            ``"auto"`` takes the best one for the device.
        checkpoint_path: A checkpoint of your own (a YuNet state dict), instead of the published
            weights. The variant then names the architecture to build, and the runtime is torch.
        labels: Names to attach, overriding the published vocabulary: a list of class names (one,
            ``face``), or the published ``labels.json`` shape, which names the landmarks too.
        revision: Pin a published revision instead of taking the latest.

    Attributes:
        runtime: The artifact key actually in use.
        device: The device actually in use.
    """

    VARIANTS = ("small", "base")

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
                self.device, artifacts("yunet", variant, revision=revision), runtime,
                derives=registry.DERIVES.get("yunet"))
        else:
            # A checkpoint mozo did not publish: the architecture is known, the graph is not.
            self.runtime = "torch-fp32"

        forward = None
        weights = None
        if framework_of(self.runtime) == "tensorrt":
            # Built once per GPU and TensorRT version from the published graph it derives from
            # (``registry.DERIVES``), for any image up to the size the profile allows.
            forward = TensorRTRunner(
                parts("yunet", variant, self.runtime, revision=revision)[""], self.device,
                shapes={"input": _TENSORRT_SHAPES})
        elif not self.runtime.startswith("torch"):
            runner = make_runner(resolve("yunet", variant, self.runtime, revision=revision),
                                 self.runtime, device=self.device)

            def forward(batch):
                import torch
                return [torch.from_numpy(out) for out in runner(batch.cpu().numpy())]
        else:
            weights = (Path(checkpoint_path) if checkpoint_path
                       else resolve("yunet", variant, self.runtime, revision=revision))
        self._predictor = Predictor(weights, variant, device=self.device, forward=forward)

        self._labels = labels_for(
            "yunet", variant, caller=labels, revision=revision, published=checkpoint_path is None)
        if self._labels is None:
            _log.warning(
                "[mozo] no names for this checkpoint. Detections will carry class_id 0 and "
                "class_name=None, landmarks an id and name=None. Pass labels=[...] to name them."
            )
        _log.info("YuNet %s ready on %s via %s.", variant, self.device, self.runtime)

    def predict(
        self,
        image: Union[str, np.ndarray],
        threshold: float = 0.5,
        labels: list | None = None,
    ) -> Any:
        """Find the faces in *image*.

        Args:
            image: A file path, encoded bytes, or an ``HWC`` RGB array.
            threshold: The score a face must reach to be kept, applied before NMS, as upstream's
                own ``score_threshold`` (it evaluates at 0.02). The score is
                ``sigmoid(class) * sigmoid(objectness)``; OpenCV's ``FaceDetectorYN`` reports
                the square root of the same number, so its 0.9 is 0.81 here.
            labels: Names for this call only, overriding the adapter's.

        Returns:
            A PixelFlow ``Detections``, best first: one row per face, class 0 (``face``), the box
            (not clipped to the image: a face cut by the edge can run past it), the score, and
            five landmarks -- right eye, left eye, nose tip, right and left mouth corner, the
            subject's right first -- each with ``confidence=None``: the model places them but
            does not score them.
        """
        faces = self._predictor.predict(load_image(image), threshold)
        return detections(faces.boxes, faces.scores, faces.keypoints,
                          labels if labels is not None else self._labels)


#: The image sizes a TensorRT engine accepts, as ``(min, preferred, max)`` batches, after the pad
#: to a multiple of 32. Placeholder until measured; see the family's PROVENANCE.
_TENSORRT_SHAPES = ((1, 3, 32, 32), (1, 3, 736, 1280), (1, 3, 2048, 2048))

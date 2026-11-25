# SPDX-License-Identifier: BSD-3-Clause
# Derived from libfacedetection.train, Copyright (c) 2022-2026, Shiqi Yu. See LICENSE.
"""The deployable surface: an image in, faces out -- boxes, scores and five landmarks each."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from .config import KEYPOINTS, get_spec
from .network import YuNet
from .postprocess import decode
from .preprocess import to_batch

__all__ = ["Faces", "Predictor", "load_state"]


@dataclass(frozen=True)
class Faces:
    """What one image contains, in the image's own pixels, best first.

    Attributes:
        boxes: ``(N, 4)`` float32 xyxy. Not clipped to the image, as upstream does not clip: a
            face cut by the edge can extend past it.
        scores: ``(N,)`` float32, ``sigmoid(cls) * sigmoid(obj)``.
        keypoints: ``(N, 5, 2)`` float32 ``(x, y)``: right eye, left eye, nose tip, right mouth
            corner, left mouth corner (the subject's right).
    """

    boxes: np.ndarray
    scores: np.ndarray
    keypoints: np.ndarray


def load_state(checkpoint: str | os.PathLike) -> dict[str, torch.Tensor]:
    """The published state dict: every tensor upstream's model holds, nothing dropped."""
    return torch.load(os.fspath(checkpoint), map_location="cpu", weights_only=True)


class Predictor:
    """Face detection for one YuNet checkpoint.

    Args:
        checkpoint: Path to the weights (``torch-fp32.pth``). Not read when *forward* is given.
        variant: Which published geometry. See :data:`~.config.SPECS`.
        device: Where to run.
        forward: Run the network some other way: takes the ``(1, 3, H, W)`` float batch and
            returns the twelve maps as one flat sequence, ``cls``, ``bbox``, ``obj``, ``kps``,
            each at strides 8, 16, 32. The seam an exported graph plugs into; the decode stays
            here.

    Attributes:
        spec: The geometry in use.
        model: The network, or ``None`` when *forward* replaces it.
    """

    def __init__(self, checkpoint: str | os.PathLike | None, variant: str = "small",
                 device: str | torch.device = "cpu", *,
                 forward: Callable[[torch.Tensor], tuple] | None = None):
        self.spec = get_spec(variant)
        self.device = torch.device(device)
        self._forward = forward
        self.model = None
        if forward is None:
            # On the meta device, then adopt the loaded tensors: every parameter is about to be
            # overwritten, so initialising them first is work thrown away.
            with torch.device("meta"):
                self.model = YuNet(self.spec)
            self.model.load_state_dict(load_state(checkpoint), strict=True, assign=True)
            self.model.eval().to(self.device)

    def network(self, batch: torch.Tensor) -> tuple[list[torch.Tensor], ...]:
        """Run the network on *batch*, returning ``(cls, bbox, obj, kps)`` lists of three maps."""
        if self._forward is not None:
            maps = [m.float() for m in self._forward(batch)]
            levels = len(self.spec.strides)
            return tuple(maps[i:i + levels] for i in range(0, len(maps), levels))
        with torch.inference_mode():
            return self.model(batch)

    def predict(self, image: np.ndarray, threshold: float = 0.5) -> Faces:
        """Find the faces in *image*.

        Args:
            image: ``HxWx3`` RGB ``uint8``.
            threshold: The score a face must reach (``>=``) to be kept, applied before NMS, as
                upstream's ``score_threshold``. Upstream evaluates at 0.02.

        Returns:
            :class:`Faces`, best first. An image with no face in it is an empty answer.
        """
        batch = to_batch(image, self.spec.size_divisor, self.device)
        with torch.inference_mode():
            boxes, scores, points = decode(self.network(batch), self.spec.strides, threshold,
                                           self.spec.nms_threshold)
        return Faces(boxes=boxes.cpu().numpy(), scores=scores.cpu().numpy(),
                     keypoints=points.cpu().numpy().reshape(-1, KEYPOINTS, 2))

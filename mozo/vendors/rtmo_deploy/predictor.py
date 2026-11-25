# SPDX-License-Identifier: Apache-2.0
"""The deployable surface: a frame in, people out -- boxes, scores and 17 joints each.

One stage, no detector. The network scores every grid cell as the centre of a person and predicts
that person's box, joint visibilities and a pose vector; the cells above a threshold are sorted,
NMS removes duplicates, and only then does the DCC turn each survivor's pose vector into joint
positions. That order is upstream's and it matters: the DCC sees the boxes NMS kept, so running it
on every cell and filtering afterwards would compute 2000 poses to return five -- and would batch
differently, which is not bit-identical.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from .config import KEYPOINTS, get_spec
from .network import DCC, RTMO, TRAINING_ONLY
from .postprocess import clamp_boxes, grid_priors, nms, select, to_frame, window
from .preprocess import letterbox, to_batch

__all__ = ["People", "Predictor", "load_state"]


@dataclass(frozen=True)
class People:
    """What one frame contains, in the frame's own pixels.

    Attributes:
        boxes: ``(N, 4)`` float64 xyxy.
        scores: ``(N,)`` float32, the person score (sigmoid), best first.
        keypoints: ``(N, 17, 2)`` float64 ``(x, y)``.
        visibility: ``(N, 17)`` float32, each joint's visibility (sigmoid). Upstream reports it as
            the keypoint score.
    """

    boxes: np.ndarray
    scores: np.ndarray
    keypoints: np.ndarray
    visibility: np.ndarray


def load_state(checkpoint: str | os.PathLike) -> dict[str, torch.Tensor]:
    """The published state dict, with the training-only tensors dropped by name.

    Each prefix in :data:`~.network.TRAINING_ONLY` must be present: a checkpoint laid out
    differently fails here rather than loading whatever happens to match.
    """
    state = torch.load(os.fspath(checkpoint), map_location="cpu", weights_only=True)
    for prefix in TRAINING_ONLY:
        if not any(key.startswith(prefix) for key in state):
            raise ValueError(f"checkpoint has no {prefix!r} tensors; is it an RTMO checkpoint?")
    return {k: v for k, v in state.items() if not k.startswith(TRAINING_ONLY)}


class Predictor:
    """Multi-person pose estimation for one RTMO checkpoint.

    Args:
        checkpoint: Path to the published weights (``torch-fp32.pth``).
        variant: Which published geometry. See :data:`~.config.SPECS`.
        device: Where to run.
        forward: Run the dense network some other way: takes the ``(1, 3, S, S)`` float batch and
            returns ``(scores, boxes, visibility, pose)`` as :meth:`.RTMO.forward` does. The seam
            an exported graph plugs into. The DCC still runs here, in torch, from *checkpoint*:
            it is data-dependent in size, and tiny.

    Attributes:
        spec: The geometry in use.
        model: The dense network, or ``None`` when *forward* replaces it.
        dcc: The coordinate classifier.
    """

    def __init__(self, checkpoint: str | os.PathLike, variant: str = "small",
                 device: str | torch.device = "cpu", *,
                 forward: Callable[[torch.Tensor], tuple] | None = None):
        self.spec = get_spec(variant)
        self.device = torch.device(device)
        self._forward = forward
        state = load_state(checkpoint)
        self.model = None
        if forward is None:
            # On the meta device, then adopt the loaded tensors: every parameter is about to be
            # overwritten, so initialising them first is work thrown away.
            with torch.device("meta"):
                self.model = RTMO(variant)
            self.model.load_state_dict(state, strict=True, assign=True)
            self.model.eval().to(self.device)
            self.dcc = self.model.head.dcc
        else:
            prefix = "head.dcc."
            with torch.device("meta"):
                self.dcc = DCC(self.spec)
            self.dcc.load_state_dict({k[len(prefix):]: v for k, v in state.items()
                                      if k.startswith(prefix)}, strict=True, assign=True)
            self.dcc.eval().to(self.device)
        # Every grid cell's centre: fixed by the input size and the strides, so built once.
        side = self.spec.input_size
        self._priors, _ = grid_priors([(side // s, side // s) for s in self.spec.strides],
                                      self.spec.strides, torch.float32, self.device)

    def preprocess(self, image: np.ndarray) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """RGB ``HxWx3`` ``uint8`` -> the BGR float batch, and the centre/scale that map back.

        Warped first and flipped after: the warp treats each channel alone and the border is the
        same grey in all three, so the order does not change a pixel -- and flipping the 640x640
        square instead of the frame saves copying a full-resolution image (6 MB at 1080p).
        """
        canvas, center, scale = letterbox(image, self.spec.input_size, self.spec.pad_value)
        return to_batch(canvas[..., ::-1]), center, scale

    def dense(self, batch: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Run the static half on *batch*, returning ``(scores, boxes, visibility, pose)``."""
        if self._forward is not None:
            return tuple(self._forward(batch))
        with torch.inference_mode():
            return self.model(batch.to(self.device))

    def decode(self, outputs: tuple[torch.Tensor, ...], threshold: float
               ) -> tuple[torch.Tensor, ...]:
        """``RTMOHead.predict`` for one image, in input pixels: filter, sort, NMS, DCC, clamp.

        Returns ``(boxes, scores, keypoints, visibility)`` as torch tensors.
        """
        scores, boxes, visible, pose = (t[0].to(self.device).float() for t in outputs)
        scores, cells = select(scores, threshold)
        boxes, visible = boxes[cells], visible[cells]
        if boxes.numel() > 0:
            kept = nms(boxes, scores, self.spec.nms_threshold)
            boxes, visible, scores = boxes[kept], visible[kept], scores[kept]
            survivors = cells[kept]
            with torch.inference_mode():
                keypoints = self.dcc(pose[survivors], window(boxes, self.spec.bbox_padding),
                                     self._priors[survivors])
        else:
            keypoints = boxes.new_zeros((0, KEYPOINTS, 2))
        return clamp_boxes(boxes, self.spec.input_size), scores, keypoints, visible

    def predict(self, image: np.ndarray, threshold: float = 0.5) -> People:
        """Find everyone in *image*.

        Args:
            image: ``HxWx3`` RGB ``uint8``.
            threshold: The person score a cell must exceed to be kept, before NMS. Upstream
                evaluates at 0.1.

        Returns:
            :class:`People`, best first. A frame with nobody in it is an empty answer, not an error.
        """
        batch, center, scale = self.preprocess(image)
        boxes, scores, keypoints, visible = self.decode(self.dense(batch), threshold)
        size = (self.spec.input_size, self.spec.input_size)
        boxes = boxes.cpu().numpy()
        frame_boxes = to_frame(boxes.reshape(boxes.shape[0], 2, 2), size, center, scale)
        return People(
            boxes=frame_boxes.reshape(boxes.shape[0], 4),
            scores=scores.cpu().numpy(),
            keypoints=to_frame(keypoints.cpu().numpy(), size, center, scale),
            visibility=visible.cpu().numpy(),
        )

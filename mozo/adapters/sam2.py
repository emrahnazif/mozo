# SPDX-License-Identifier: Apache-2.0
"""mozo's SAM 2 adapter: a click or a box in, PixelFlow detections out -- on a photo or a video.

SAM 2 is the reference promptable segmenter this family is measured against -- four published
sizes, from a 31.4 M-parameter tiny to a large. ``predict`` segments one image; ``tracker()``
starts a session that follows the objects you mark through the frames of a video, one frame at a
time, with SAM 2's memory of the frames before.

Unlike SAM 3, SAM 2's published checkpoints are Apache-2.0, the same as its code. Nothing in this
family carries a separate weights licence.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Union

import numpy as np
import pixelflow as pf
import torch

from .. import registry
from ..image import load_image
from ..vendors import sam2_deploy
from ..vendors.sam2_deploy.video import split_prompts
from ._promptable import PromptablePredictor, bounds

__all__ = ["Sam2Predictor", "Sam2Tracker"]


class Sam2Predictor(PromptablePredictor):
    """Promptable segmentation on SAM 2.

    Examples:
        >>> model = Sam2Predictor("tiny")                                # doctest: +SKIP
        >>> found = model.predict("street.jpg", boxes=[40, 60, 300, 480])  # doctest: +SKIP
        >>> found[0].class_name                                          # doctest: +SKIP
        None
    """

    FAMILY = "sam2"
    EXECUTES = registry.EXECUTES["sam2"]
    DISPLAY = "SAM 2"
    VENDOR = sam2_deploy
    #: Smallest first: it is the default, and it is the one that fits on an edge device.
    VARIANTS = ("tiny", "small", "base_plus", "large")

    def tracker(self) -> "Sam2Tracker":
        """Start following objects through a video, one frame at a time.

        A new, empty session on this loaded model. It holds no frames until
        :meth:`Sam2Tracker.update` gives it one, and only ever the few a later frame can still
        read, so a long video or a camera runs in constant memory. Any number of trackers can
        share one model (two cameras, two trackers); one tracker follows one stream and is not
        safe to call from two threads at once.

        Examples:
            >>> model = mozo.get_model("sam2/tiny")                       # doctest: +SKIP
            >>> tracker = model.tracker()                                 # doctest: +SKIP
            >>> for i, frame in enumerate(pf.VideoReader("street.mp4")):  # doctest: +SKIP
            ...     found = tracker.update(frame, boxes=[40, 60, 300, 480] if i == 0 else None)
        """
        return Sam2Tracker(self._segmenter.tracker(), self.device)


class Sam2Tracker:
    """One SAM 2 tracking session over one stream of frames. Made by :meth:`Sam2Predictor.tracker`.

    Each prompt passed to :meth:`update` starts one object on that frame; every call follows all
    the objects started so far onto the frame it is given. The frames are taken in order and only
    forward: SAM 2 reads the frames before, never after.
    """

    def __init__(self, session: Any, device: str) -> None:
        self._session = session
        self._device = device
        #: What to call each object, by object id; ``None`` where no name was given.
        self._names: list[str | None] = []

    def update(
        self,
        image: Union[str, Path, bytes, np.ndarray],
        points: Any = None,
        labels: Any = None,
        boxes: Any = None,
        name: Union[str, Sequence[str], None] = None,
    ) -> "pf.detections.Detections":
        """Follow every object onto the next frame, after starting one for each prompt given.

        Args:
            image: The next frame: a file path, encoded image bytes, or an ``HxWx3`` RGB ``uint8``
                array. Every frame must be the size of the first.
            points: ``(N, 2)`` or ``(B, N, 2)`` x, y clicks in the frame's own pixels.
            labels: ``(N,)`` or ``(B, N)``, 1 for a point to include and 0 for one to exclude.
            boxes: ``(4,)`` or ``(B, 4)`` x1, y1, x2, y2 in the frame's own pixels.
            name: What to call the new objects: one name, or one per prompt.

            The prompts mean what they mean in :meth:`Sam2Predictor.predict`; each of the ``B``
            prompts (a box, a set of clicks, or a box with clicks) starts one new object on this
            frame. The first call must start at least one; later calls may start more or none.

        Returns:
            A PixelFlow ``Detections`` with one row per object the model sees in this frame, in
            object order: ``tracker_id`` and ``class_id`` are the object's id (0, 1, 2… in the
            order objects were started, never reused), ``class_name`` its name or ``None``,
            ``confidence`` the model's probability that the object is visible, and the mask (at
            the frame's own size) with its tight box. An object the model says is absent -- out of
            frame, or hidden -- has no row, and comes back under the same id when it is seen
            again. An object judged present whose mask is empty has no row either: PixelFlow keeps
            no detection with an empty box.

        Raises:
            ValueError: If the first call has no prompt, a frame's size differs from the first's,
                points arrive without labels, or *name* does not give one name per prompt.
        """
        pixels = load_image(image)
        prompts = split_prompts(points, labels, boxes)
        names = [name] * len(prompts) if isinstance(name, str) else (
            list(name) if name is not None else [None] * len(prompts))
        if len(names) != len(prompts):
            raise ValueError(f"{len(names)} names for {len(prompts)} prompt(s); give one each or none")
        frame = self._session.update(pixels, prompts)
        self._names.extend(names)

        visible = (frame.object_scores > 0).cpu().numpy()
        masks = frame.masks
        confidence = torch.sigmoid(frame.object_scores).cpu().numpy()
        found = pf.Detections()
        for index in np.flatnonzero(visible):
            mask = masks[index]
            found.add_detection(pf.Detection(
                bbox=bounds(mask[None])[0].tolist(),
                masks=[mask],
                class_id=int(index),
                class_name=self._names[index],
                confidence=float(confidence[index]),
                tracker_id=int(index),
            ))
        if self._device == "mps":
            torch.mps.empty_cache()
        return found

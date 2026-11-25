# SPDX-License-Identifier: Apache-2.0
"""The public segmenter: load a checkpoint, prompt an image, get masks in its own pixels."""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock

import numpy as np
import torch

from .config import SPECS
from .image import MASK_THRESHOLD, resize_longest_side, to_model_coords, to_original
from .network import SamHq

__all__ = ["Segmentation", "Segmenter"]

#: How many images' encoder outputs to keep. Each is ~4 MB for the embedding plus ~8 MB for the
#: fused HQ features, whatever the variant, so five stays small on any target.
CACHE_SIZE = 5

#: Keys the checkpoints carry that nothing here runs: the ImageNet classifier head TinyViT was
#: pretrained with, which MobileSAM's ``vit_tiny`` checkpoint (and so HQ-SAM's) still holds.
#: Dropped by name rather than with ``strict=False``, so any other stray key is still an error.
UNUSED = ("image_encoder.norm_head.", "image_encoder.head.")


@dataclass
class Segmentation:
    """Masks for one prompt batch, in the source image's own pixels."""

    masks: np.ndarray  # (b, c, h, w) bool
    scores: np.ndarray  # (b, c) the SAM token's predicted IoU for each mask
    # Feed one of these back as ``mask_input`` to refine, choosing the candidate first:
    # ``logits[:, scores.argmax(1)][:, None]``.
    logits: np.ndarray  # (b, c, 256, 256) low-res logits

    def __len__(self) -> int:
        return len(self.masks)


def _variant_of(state: dict) -> str:
    """Name the variant a checkpoint holds, by its trunk's shape."""
    if "image_encoder.patch_embed.seq.0.c.weight" in state:
        return "tiny"
    width = state["image_encoder.patch_embed.proj.weight"].shape[0]
    for name, spec in SPECS.items():
        if getattr(spec, "embed_dim", None) == width:
            return name
    raise ValueError(f"no known variant has a ViT of width {width}")


class Segmenter:
    """Promptable segmentation for one HQ-SAM checkpoint.

    Args:
        checkpoint: Path to a ``.pth`` holding an HQ-SAM state dict, as upstream publishes it.
        variant: Which geometry to build. Inferred from the checkpoint when omitted.
        device: Where to run. mozo decides this; the default is only for direct use.

    Attributes:
        variant: Which geometry this checkpoint holds.
        image_size: Long side the encoder resizes to.
    """

    def __init__(
        self,
        checkpoint: str | os.PathLike,
        variant: str | None = None,
        device: str | torch.device = "cpu",
    ):
        state = torch.load(os.fspath(checkpoint), map_location="cpu", weights_only=True)
        variant = variant or _variant_of(state)
        if variant not in SPECS:
            raise ValueError(f"Unsupported variant {variant!r}. Choose from: {list(SPECS)}")

        self.network = SamHq(SPECS[variant])
        # Strict: every tensor the network has must be in the file, and every tensor in the file
        # must have somewhere to go, apart from the classifier head named above.
        self.network.load_state_dict({k: v for k, v in state.items() if not k.startswith(UNUSED)})
        self.network.eval().to(device)

        self.variant = variant
        self.device = device
        self.image_size = self.network.image_size
        self._cache: OrderedDict[bytes, dict] = OrderedDict()
        # One adapter instance is shared across requests -- mozo.server runs handlers in a
        # threadpool -- and check-then-act on an OrderedDict is not safe across threads.
        self._lock = Lock()

    @torch.inference_mode()
    def encode(self, image: np.ndarray) -> dict:
        """Return the encoder features for one image, computing them only if not already held.

        The expensive half, and it depends on nothing but the image, so repeated prompts on one
        photograph pay for it once. Keyed on a sha256 of the pixels, for the reasons SAM 2's
        segmenter gives: the same image arriving twice over HTTP is two arrays and one encode.
        """
        key = hashlib.sha256(np.ascontiguousarray(image)).digest()
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        batch = resize_longest_side(image, self.image_size).to(self.device)
        features = self.network.encode(batch)
        with self._lock:
            self._cache[key] = features
            while len(self._cache) > CACHE_SIZE:
                self._cache.popitem(last=False)
        return features

    @torch.inference_mode()
    def predict(
        self,
        image: np.ndarray,
        points: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        boxes: np.ndarray | None = None,
        mask_input: np.ndarray | None = None,
        multimask_output: bool = True,
    ) -> Segmentation:
        """Segment what the prompt points at.

        Args:
            image: ``HxWx3`` RGB ``uint8`` array.
            points: ``(N, 2)`` or ``(B, N, 2)`` x, y clicks in the image's own pixels.
            labels: ``(N,)`` or ``(B, N)``, 1 for a point to include and 0 for one to exclude.
                Required with *points*; guessing between the two returns a confident mask of
                the wrong thing.
            boxes: ``(4,)`` or ``(B, 4)`` x1, y1, x2, y2 in the image's own pixels.
            mask_input: ``(B, 1, 256, 256)`` logits from a previous call, to refine.
            multimask_output: Return three candidate masks instead of one.

        Returns:
            A :class:`Segmentation` whose masks are in the source image's pixels.
        """
        if points is None and boxes is None and mask_input is None:
            raise ValueError("a prompt is required: give points, boxes or mask_input")
        if (points is None) != (labels is None):
            raise ValueError("points and labels go together; got one without the other")

        shape = image.shape[:2]
        features = self.encode(image)
        clicks, corners = self._prompt(points, labels, boxes, shape)
        low_res, iou = self.network.decode(
            features,
            clicks,
            corners,
            None if mask_input is None else torch.as_tensor(mask_input, dtype=torch.float,
                                                            device=self.device),
            multimask_output,
        )
        masks = to_original(low_res, features["input_size"], shape, self.image_size)
        return Segmentation(
            masks=(masks > MASK_THRESHOLD).cpu().numpy(),
            scores=iou.cpu().numpy(),
            logits=low_res.cpu().numpy(),
        )

    def _prompt(self, points, labels, boxes, shape):
        """Scale clicks and boxes into the encoder's frame, as upstream's ``predict`` does.

        Unlike SAM 2, SAM has a box input of its own: a box goes to the prompt encoder as a box,
        not folded into the point list, and when both are given the clicks' tokens come first.
        Clicks without a box get a padding token appended; clicks with one do not.
        """
        clicks = corners = None
        if points is not None:
            # Not cast to float32 here: upstream scales in float64 from whatever the caller passed,
            # and a float64 click that float32 cannot hold would land a last bit elsewhere.
            coords = np.asarray(points)
            flags = np.asarray(labels)
            if coords.ndim == 2:
                coords, flags = coords[None], flags[None]
            if coords.shape[:2] != flags.shape[:2]:
                raise ValueError(f"{coords.shape[1]} points but {flags.shape[1]} labels")
            clicks = (to_model_coords(coords, shape, self.image_size).to(self.device),
                      torch.as_tensor(flags, dtype=torch.int, device=self.device))
        if boxes is not None:
            corners = to_model_coords(np.asarray(boxes).reshape(-1, 2, 2), shape, self.image_size)
            corners = corners.reshape(-1, 4).to(self.device)
        if clicks is not None and corners is not None and len(clicks[0]) != len(corners):
            raise ValueError(
                f"{len(corners)} boxes and {len(clicks[0])} point sets: give one point set per "
                "box, or prompt with only one of the two"
            )
        return clicks, corners

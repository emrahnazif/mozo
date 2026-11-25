# SPDX-License-Identifier: Apache-2.0
"""The public detector: load a Mayaku checkpoint, run an image, get boxes (and masks) back in the
image's own pixels."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch

from .blocks import STRIDES
from .decode import postprocess
from .image import PAD, letterbox, to_original
from .mask import COORD_SCALE, decode_masks
from .network import KERNEL_LAYOUT, MASK_CH, TIERS, Tier, build_network, read_checkpoint


@dataclass(frozen=True)
class Spec:
    """Everything a runtime needs besides the weights, written down per variant.

    The torch loader holds it against the checkpoint's own record (:func:`check_sidecar`), so a
    checkpoint that disagrees is refused rather than run with the wrong geometry. The graph
    runtimes have no checkpoint to read, which is why these are written here rather than read.
    """

    tier: str
    seg: bool
    canvas: tuple[int, int] = (800, 800)
    num_classes: int = 22
    #: The prediction decode: NMS IoU, candidates kept before NMS, detections kept after.
    iou: float = 0.5
    topk: int = 1000
    max_det: int = 300
    #: The checkpoint's own tuned score threshold (its ``predict.conf``).
    conf: float = 0.4

    @property
    def cfg(self) -> Tier:
        return TIERS[self.tier]


#: mozo's variant names -> what each publishes. Every variant is one Mayaku checkpoint; the street
#: models are the first. Another Mayaku checkpoint of a tier in :data:`~.network.TIERS` (a COCO
#: base model, say) joins as another entry here with its own class count and canvas; a new tier
#: joins ``TIERS`` first.
SPECS: dict[str, Spec] = {
    "street-nano": Spec("n", seg=False),
    "street-medium": Spec("m", seg=False),
    "street-seg-nano": Spec("n", seg=True),
    "street-seg-medium": Spec("m", seg=True),
}


def get_spec(variant: str) -> Spec:
    try:
        return SPECS[variant]
    except KeyError:
        raise ValueError(f"unknown Mayaku variant {variant!r}; known: {list(SPECS)}") from None


def check_sidecar(sidecar: dict[str, Any], spec: Spec, source: str) -> None:
    """Refuse a checkpoint whose own record disagrees with *spec* anywhere the decode depends on.

    The seg checkpoints carry no ``predict`` block, and upstream then decodes with its defaults
    (conf 0.4, IoU 0.5, one class per location), which are what the det checkpoints record; so an
    absent block is checked against those defaults rather than skipped.
    """
    model = sidecar["config"]["model"]
    predict = sidecar.get("predict") or {"conf": 0.4, "iou": 0.5, "max_det": 300, "topk": 1000,
                                          "multi_label": False}
    cfg = spec.cfg
    want = {
        "schema_version": (sidecar.get("schema_version"), 2),
        "task": (sidecar.get("task"), "detect"),
        "tier": (model.get("tier"), spec.tier),
        "seg": (bool(model.get("seg")), spec.seg),
        "keypoints": (model.get("keypoints"), None),
        "obb": (bool(model.get("obb")), False),
        "aux_arm": (model.get("aux_arm"), "box_tower"),
        "mask_stride": (model.get("mask_stride"), cfg.mask_stride),
        "canvas": (tuple(sidecar["canvas_hw"]), spec.canvas),
        "classes": (len(sidecar["class_names"]), spec.num_classes),
        "strides": (tuple(sidecar["decode"]["strides"]), STRIDES),
        "reg_max": (sidecar["decode"]["reg_max"], cfg.reg_max),
        "predict": ((predict["conf"], predict["iou"], predict["topk"], predict["max_det"],
                     predict["multi_label"]),
                    (spec.conf, spec.iou, spec.topk, spec.max_det, False)),
        "preprocess": ((sidecar["preprocess"]["resize"], sidecar["preprocess"]["pad_value"],
                        sidecar["preprocess"]["channels"]), ("letterbox", PAD, "RGB")),
    }
    mask = sidecar.get("mask")
    want["mask"] = ((mask["stride"], mask["channels"], mask["coord_scale"],
                     tuple(map(tuple, mask["kernel_layout"]))) if mask else None,
                    (cfg.mask_stride, MASK_CH, COORD_SCALE, KERNEL_LAYOUT) if spec.seg else None)
    wrong = {k: v for k, (v, expected) in want.items() if v != expected}
    if wrong:
        raise ValueError(f"{source} is not the Mayaku checkpoint this variant expects: {wrong}")


def detect(
    image: np.ndarray,
    forward: Callable[[torch.Tensor], Sequence[torch.Tensor]],
    spec: Spec,
    conf: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Detect in one image, with *forward* supplying the middle step.

    Letterbox, run whatever executes the graph, decode on the CPU, map back. A torch module, an ONNX
    session and a TensorRT engine differ only in *forward*, so all of them share one pre- and
    post-processing path.

    Args:
        image: ``HxWx3`` RGB ``uint8`` array.
        forward: Takes the ``(1, 3, 800, 800)`` float batch and returns the graph's maps in
            upstream's order (six head maps, then for a seg variant the mask feature and three
            kernel maps).
        spec: The variant's :class:`Spec`.
        conf: Minimum score, ``0 <= conf < 1``. Replaces the checkpoint's own 0.4, as upstream's
            ``conf=`` does.

    Returns:
        Boxes in the source image's pixels, scores, class ids, and for a seg variant one boolean
        mask per detection at the source resolution (``None`` for a det variant).
    """
    if not 0 <= conf < 1:
        raise ValueError(f"conf must be in [0, 1), got {conf}")
    batch, gain, pad_x, pad_y = letterbox(image, spec.canvas[0])
    meta = {"ratio": gain, "pad": (pad_x, pad_y), "shape": image.shape[:2]}
    # Decoded on the CPU whatever ran the graph, as upstream does for every device but CUDA, so
    # the answer does not depend on where the forward pass ran.
    preds = [p.float().cpu() for p in forward(batch)]
    dets, anchors, points, stride = postprocess(
        preds, spec.num_classes, spec.cfg.reg_max, conf, spec.iou, spec.topk, spec.max_det)
    masks = decode_masks(preds, dets, anchors, points, stride, meta) if spec.seg else None
    return (to_original(dets[:, :4], gain, pad_x, pad_y, meta["shape"]), dets[:, 4],
            dets[:, 5].long(), masks)


@dataclass
class Detections:
    """Detections for one image, in that image's own pixels."""

    boxes: np.ndarray       # (n, 4) x1, y1, x2, y2
    scores: np.ndarray      # (n,)
    class_ids: np.ndarray   # (n,) int64
    names: list[str]        # (n,) the class name of each detection
    masks: np.ndarray | None = None   # (n, h, w) bool for a seg variant, else None

    def __len__(self) -> int:
        return len(self.scores)


class Detector:
    """A Mayaku checkpoint loaded, fused and ready.

    Args:
        checkpoint: A Mayaku ``.pth``.
        variant: Which variant it is, a key of :data:`SPECS` (``street-nano``, ...); the
            checkpoint must agree.
        device: Where the graph runs.

    Attributes:
        names: The class names the checkpoint records, in class-id order.
        spec: The variant's :class:`Spec`.
    """

    def __init__(self, checkpoint: str | os.PathLike, variant: str = "street-nano",
                 device: str | torch.device = "cpu") -> None:
        self.spec = get_spec(variant)
        state, sidecar = read_checkpoint(checkpoint)
        check_sidecar(sidecar, self.spec, os.fspath(checkpoint))
        self.names = list(sidecar["class_names"])
        self.network = build_network(state, self.spec.cfg, self.spec.num_classes, self.spec.seg)
        self.network.eval().to(device)
        self.device = torch.device(device)

    @torch.no_grad()
    def forward(self, batch: torch.Tensor) -> list[torch.Tensor]:
        """Run one batch through the fused graph. What :func:`detect` plugs a runtime into."""
        return self.network(batch.to(self.device))

    def predict(self, image: np.ndarray, conf: float | None = None) -> Detections:
        """Detect in one ``HxWx3`` RGB ``uint8`` image; *conf* defaults to the checkpoint's 0.4."""
        conf = self.spec.conf if conf is None else conf
        boxes, scores, ids, masks = detect(image, self.forward, self.spec, conf)
        ids_np = ids.numpy()
        return Detections(boxes.numpy(), scores.numpy(), ids_np,
                          [self.names[int(i)] for i in ids_np],
                          None if masks is None else masks.numpy())


__all__ = ["Detections", "Detector", "SPECS", "Spec", "check_sidecar", "detect", "get_spec"]

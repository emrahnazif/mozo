# SPDX-License-Identifier: Apache-2.0
"""Video tracking, one frame at a time: upstream's ``SAM2VideoPredictor`` as a session.

Upstream's predictor is built around a whole video: ``init_state`` loads every frame into memory
at 1024x1024 float32 (12 MiB a frame), prompts are added to frames, and ``propagate_in_video``
walks them. This session takes the frames one by one instead and holds only what a later frame can
still read. That is not an approximation of upstream, and it is worth saying why.

**What a frame reads.** Tracking frame *t* reads, per object, its prompted frame's output, the
outputs of frames *t-1* … *t-6* (``num_maskmem - 1`` memories) and the object pointers of frames
*t-1* … *t-15* (``max_obj_ptrs_in_encoder - 1``). Nothing later than *t*, nothing earlier than
*t-15* (apart from the prompted frame). So the session keeps exactly those and drops the rest, and
upstream's own predictor driven one frame at a time, with later frames unreadable, gives the same
logits as its offline run -- measured, bit for bit, before a line of this was written.

**Objects are independent.** The current predictor (2024-12, unlike the legacy one EdgeTAM
uses) runs each object on its own, batch size one. So an object prompted on a later frame is
exactly a session that started on that frame with that object alone, and the others never see it.
That is what lets :meth:`Tracker.update` take new prompts on any frame.

**The one place a stream differs from a file: length.** Upstream normalises the object pointers'
temporal encoding by ``min(num_frames, 16) - 1``, so on a clip shorter than 16 frames its answer
depends on how long the clip is. A stream has no length; it reads as a video of 16 frames or more,
which is what any clip of 16 frames or more gets from upstream anyway. ``num_frames`` exists so the
gate can check short clips; mozo does not pass it.

**Two numeric details that look like noise and are not** (both reproduced, both gated):

- Frames are resized the way upstream's *video* loader does it, not the way its image predictor
  does: PIL's default (bicubic) resize of the 8-bit image, divided by 255 in float64, stored as
  float32, normalised in float32 (``sam2/utils/misc.py``). The image path in :mod:`.image` uses an
  antialiased bilinear resize; the same frame gives slightly different features through the two.
  That is upstream's own behaviour, kept on both sides.
- Mask memories are stored in bfloat16 on every device (``SAM2VideoPredictor``) and read back into
  float32 attention.

**Holes are not filled.** Upstream's video predictor asks for holes up to 8 low-res pixels to be
filled, with a connected-components kernel from a CUDA extension it builds optionally; without the
extension -- every CPU install, and any install built without ``nvcc`` -- it catches the import
error and skips the step. This session reproduces the skip. PROVENANCE records the difference from
GPU installs that built the extension.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .image import MEAN, STD
from .predictor import BOX_BOTTOM_RIGHT, BOX_TOP_LEFT

__all__ = ["Frame", "Tracker", "load_frame", "split_prompts"]


def load_frame(image: np.ndarray, size: int) -> torch.Tensor:
    """Preprocess one RGB frame exactly as upstream's video loader does.

    ``_load_img_as_tensor`` resizes with PIL's default filter and divides in float64; the frame is
    then stored in a float32 buffer and normalised in place (``load_video_frames_from_jpg_images``).

    Args:
        image: ``HxWx3`` RGB ``uint8`` array.
        size: Square side the encoder runs at.

    Returns:
        The ``(1, 3, size, size)`` normalised float32 batch.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"expected an HxWx3 RGB uint8 array, got {image.dtype} {image.shape}")
    resized = np.array(Image.fromarray(image).convert("RGB").resize((size, size)))
    # float64, as upstream's ``img_np / 255.0`` is, then cast on the way into float32 storage.
    frame = torch.from_numpy(resized / 255.0).permute(2, 0, 1).to(torch.float32)
    mean = torch.tensor(MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(STD, dtype=torch.float32)[:, None, None]
    frame -= mean
    frame /= std
    return frame.unsqueeze(0)


def split_prompts(points=None, labels=None, boxes=None) -> list[dict]:
    """Turn :meth:`.Segmenter.predict`-shaped prompts into one prompt per new object.

    The same shapes and the same rules as the image path: ``points`` ``(N, 2)`` or ``(B, N, 2)``
    with ``labels`` ``(N,)`` or ``(B, N)``, ``boxes`` ``(4,)`` or ``(B, 4)``; given together, one
    point set per box. Prompt *b* becomes object *b*.

    Returns:
        A list of dicts with ``points``, ``labels`` and ``box`` (each possibly ``None``); empty
        when nothing was given.
    """
    if (points is None) != (labels is None):
        raise ValueError("points and labels go together; got one without the other")
    groups = []
    if boxes is not None:
        corners = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        groups = [{"points": None, "labels": None, "box": box} for box in corners]
    if points is not None:
        clicks = np.asarray(points, dtype=np.float32)
        flags = np.asarray(labels, dtype=np.int32)
        if clicks.ndim == 2:
            clicks, flags = clicks[None], flags[None]
        if clicks.ndim != 3 or clicks.shape[-1] != 2 or clicks.shape[:2] != flags.shape[:2]:
            raise ValueError(f"points must be (N, 2) or (B, N, 2) with labels (N,) or (B, N); "
                             f"got {np.shape(points)} and {np.shape(labels)}")
        if groups and len(groups) != len(clicks):
            raise ValueError(f"{len(groups)} boxes and {len(clicks)} point sets: give one point "
                             "set per box, or prompt with only one of the two")
        if not groups:
            groups = [{"points": None, "labels": None, "box": None} for _ in clicks]
        for group, xy, flag in zip(groups, clicks, flags):
            group["points"], group["labels"] = xy, flag
    return groups


@dataclass
class Frame:
    """What one frame of tracking returns, for every object the session holds.

    Attributes:
        object_ids: The objects, in the order they were prompted.
        logits: ``(N, H, W)`` float32 mask logits at the frame's own size. An object the model
            says is absent has every logit at -1024 (upstream's placeholder).
        object_scores: ``(N,)`` the model's visibility logit for each object; > 0 means present.
    """

    object_ids: list[int]
    logits: torch.Tensor
    object_scores: torch.Tensor

    @property
    def masks(self) -> np.ndarray:
        """``(N, H, W)`` bool, logits above zero, as upstream's ``video_res_masks > 0``."""
        return (self.logits > 0).cpu().numpy()


class Tracker:
    """A tracking session over one stream of frames. Built by :meth:`.Segmenter.tracker`.

    Not thread-safe: one session follows one stream. Any number of sessions may share a network.

    Args:
        network: The loaded :class:`.network.Sam2`.
        device: Where it runs.
        num_frames: The video's length, if known. Only the gate passes it (see the module text).
    """

    #: How many frames back a frame reads, besides its prompted frame: the larger of the memory
    #: window (``num_maskmem - 1``) and the pointer window (``max_obj_ptrs_in_encoder - 1``).
    def __init__(self, network, device, num_frames: int | None = None):
        self.network = network
        self.device = device
        self.num_frames = num_frames
        self.reach = max(network.num_maskmem - 1, network.max_obj_ptrs_in_encoder - 1)
        self.frame_idx = -1
        self.shape: tuple[int, int] | None = None
        #: Per object: its prompted frame's output and the recent frames' outputs, keyed by frame.
        self.outputs: list[dict[str, dict[int, dict]]] = []
        #: ``maskmem_pos_enc`` is the same for every frame and object; upstream keeps one copy.
        self._maskmem_pos_enc: list[torch.Tensor] | None = None

    @property
    def objects(self) -> int:
        """How many objects the session has been given."""
        return len(self.outputs)

    @torch.inference_mode()
    def update(self, image: np.ndarray, prompts: list[dict] | None = None) -> Frame:
        """Track every object onto *image*, after starting one object per entry of *prompts*.

        Args:
            image: ``HxWx3`` RGB ``uint8``, the next frame of the stream.
            prompts: One dict per new object, with ``points`` ``(P, 2)`` and ``labels`` ``(P,)``,
                ``box`` ``(4,)``, or both, in the frame's pixels (upstream's
                ``add_new_points_or_box`` arguments).

        Returns:
            This frame's :class:`Frame`.
        """
        shape = image.shape[:2]
        if self.shape is None:
            if not prompts:
                raise ValueError("the first frame needs a prompt: there is nothing to track yet")
            self.shape = shape
        elif shape != self.shape:
            raise ValueError(f"every frame must be the size of the first, {self.shape[1]}x"
                             f"{self.shape[0]}; this one is {shape[1]}x{shape[0]}")
        self.frame_idx += 1
        t = self.frame_idx
        net = self.network

        pixels = load_frame(image, net.image_size).to(self.device)
        backbone_out = net.forward_image(pixels)
        _, feats, pos, sizes = net._prepare_backbone_features(backbone_out)

        # ``add_new_points_or_box``: a prompted frame is decoded without memory and without the
        # memory encoder, which runs afterwards from the stored low-res logits.
        for prompt in prompts or []:
            out = net.track_step(
                frame_idx=t, is_init_cond_frame=True, current_vision_feats=feats,
                current_vision_pos_embeds=pos, feat_sizes=sizes,
                point_inputs=self._point_inputs(prompt), output_dict=None,
                num_frames=self.num_frames, run_mem_encoder=False,
            )
            # ``propagate_in_video_preflight``: memory from the low-res logits, upsampled to the
            # encoder's size and binarised (the mask came from clicks).
            high_res = F.interpolate(out["pred_masks"], size=(net.image_size, net.image_size),
                                     mode="bilinear", align_corners=False)
            features, pos_enc = net._encode_new_memory(
                current_vision_feats=feats, feat_sizes=sizes, pred_masks_high_res=high_res,
                object_score_logits=out["object_score_logits"], is_mask_from_pts=True,
            )
            self.outputs.append({"cond_frame_outputs": {t: self._compact(out, features, pos_enc)},
                                 "non_cond_frame_outputs": {}})

        # ``propagate_in_video`` for frame t, object by object.
        masks, scores = [], []
        for output in self.outputs:
            current = output["cond_frame_outputs"].get(t)
            if current is None:
                out = net.track_step(
                    frame_idx=t, is_init_cond_frame=False, current_vision_feats=feats,
                    current_vision_pos_embeds=pos, feat_sizes=sizes, point_inputs=None,
                    output_dict=output, num_frames=self.num_frames, run_mem_encoder=True,
                )
                current = self._compact(out, out["maskmem_features"], out["maskmem_pos_enc"])
                output["non_cond_frame_outputs"][t] = current
            masks.append(current["pred_masks"])
            scores.append(current["object_score_logits"])
            # Forget what no later frame will read.
            output["non_cond_frame_outputs"].pop(t - self.reach, None)

        logits = torch.cat(masks, dim=0) if len(masks) > 1 else masks[0]
        logits = F.interpolate(logits, size=self.shape, mode="bilinear", align_corners=False)
        return Frame(
            object_ids=list(range(len(self.outputs))),
            logits=logits[:, 0],
            object_scores=torch.cat(scores, dim=0)[:, 0],
        )

    def _point_inputs(self, prompt: dict) -> dict:
        """Upstream's ``add_new_points_or_box`` arithmetic: box corners first, normalise, scale."""
        points = prompt.get("points")
        labels = prompt.get("labels")
        box = prompt.get("box")
        points = (torch.zeros(0, 2, dtype=torch.float32) if points is None
                  else torch.tensor(np.asarray(points), dtype=torch.float32))
        labels = (torch.zeros(0, dtype=torch.int32) if labels is None
                  else torch.tensor(np.asarray(labels), dtype=torch.int32))
        points, labels = points.unsqueeze(0), labels.unsqueeze(0)
        if box is not None:
            box_coords = torch.tensor(np.asarray(box), dtype=torch.float32).reshape(1, 2, 2)
            box_labels = torch.tensor([BOX_TOP_LEFT, BOX_BOTTOM_RIGHT], dtype=torch.int32)
            points = torch.cat([box_coords, points], dim=1)
            labels = torch.cat([box_labels.reshape(1, 2), labels], dim=1)
        height, width = self.shape
        # Normalised then scaled, as upstream does: dividing by an integer tensor, then
        # multiplying by the encoder's size.
        points = points / torch.tensor([width, height])
        points = points * self.network.image_size
        return {"point_coords": points.to(self.device), "point_labels": labels.to(self.device)}

    def _compact(self, out: dict, maskmem_features, maskmem_pos_enc) -> dict:
        """What upstream keeps of a frame's output (``_run_single_frame_inference``)."""
        if self._maskmem_pos_enc is None:
            self._maskmem_pos_enc = [x[0:1].clone() for x in maskmem_pos_enc]
        return {
            "maskmem_features": maskmem_features.to(torch.bfloat16),
            "maskmem_pos_enc": self._maskmem_pos_enc,
            "pred_masks": out["pred_masks"],
            "obj_ptr": out["obj_ptr"],
            "object_score_logits": out["object_score_logits"],
        }

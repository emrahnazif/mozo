# SPDX-License-Identifier: Apache-2.0
"""A batch of objects tracked together, and everything they remember.

Upstream tracks objects in batches: each batch is one tracker state, decoded as one batch and
encoded into memory as one batch. In concept tracking a batch is "the objects that appeared on the
same frame"; in click tracking it is "every object prompted on the first frame". Batched and
unbatched forwards are not bit-identical, and some heuristics act on a whole batch (a re-seed
re-encodes every object in it; memory selection scores a frame by the batch's average), so this
module keeps upstream's batches exactly.

**What is kept, and why that is enough.** A frame reads its batch's conditioning frames (the four
temporally closest of them) and, through memory selection, at most the fifteen most recent past
frames whose score clears the threshold, plus the frame just before. So a batch keeps its five most
recent conditioning frames (five, not four: with more than four, upstream orders the four it picks
"closest before, then by distance", and five kept reproduces that order), the fifteen most recent
selectable frames, and every frame of the last :data:`RECENT` frames. The last clause exists for
one event: removing an object re-scores every stored frame of its batch. Objects are only removed
within their first fifteen frames, and a batch's objects all start on the same frame, so a batch an
object can still be removed from is younger than :data:`RECENT` frames and has lost nothing.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F

from .network import TrackerNet

__all__ = ["ObjectGroup", "RECENT"]

#: Frames kept unconditionally, more than the hotstart probation (see the module text).
RECENT = 20

#: Conditioning frames kept: one more than are read (see the module text).
COND_KEPT = 5

#: A store with nothing in it, for steps that read no memory.
_EMPTY = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}

#: Where an object's given mask or clicks are stored: every frame given a prompt is a
#: conditioning frame, tracked before or not (upstream's ``add_all_frames_to_correct_as_cond``).
_GIVEN = "cond_frame_outputs"


def non_overlapping(masks: torch.Tensor) -> torch.Tensor:
    """Keep each pixel only for the object with the highest logit there; clamp the others to -10."""
    if masks.size(0) == 1:
        return masks
    winner = torch.argmax(masks, dim=0, keepdim=True)
    index = torch.arange(masks.size(0), device=masks.device)[:, None, None, None]
    return torch.where(winner == index, masks, torch.clamp(masks, max=-10.0))


class ObjectGroup:
    """One batch of objects: upstream's per-batch tracker state, bounded.

    Args:
        net: The tracker.
        features: ``features(frame, batch)`` -> ``(feats, pos)``: the frame's three levels and
            their encodings, ``(HW, batch, C)`` each, finest first.
        video_hw: The frame's ``(height, width)``.
        num_frames: The video's length when known (the gate only).
    """

    def __init__(self, net: TrackerNet, features, video_hw: tuple[int, int],
                 num_frames: int | None = None):
        self.net = net
        self.features = features
        self.video_hw = video_hw
        self.num_frames = num_frames
        self.obj_ids: list[int] = []
        self.store = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        self.temp: list[dict] = []          # per object: {"cond_frame_outputs": {}, ...}
        self.inputs: list[set[int]] = []    # per object: frames carrying a mask or click
        self.tracked: set[int] = set()
        self.consolidated = {"cond_frame_outputs": set(), "non_cond_frame_outputs": set()}
        self.started = False
        self._maskmem_pos_enc: list[torch.Tensor] | None = None

    def __len__(self) -> int:
        return len(self.obj_ids)

    # --- adding objects ------------------------------------------------------------------------

    def _index(self, obj_id: int) -> int:
        if obj_id in self.obj_ids:
            return self.obj_ids.index(obj_id)
        if self.started:
            raise RuntimeError("objects cannot be added to a batch once tracking has started")
        self.obj_ids.append(obj_id)
        self.temp.append({"cond_frame_outputs": {}, "non_cond_frame_outputs": {}})
        self.inputs.append(set())
        return len(self.obj_ids) - 1

    def add_mask(self, frame: int, obj_id: int, mask: torch.Tensor) -> None:
        """Give an object a mask on ``frame`` (upstream's ``add_new_mask``).

        Args:
            mask: ``(H, W)`` boolean, at any size; resized to the encoder's mask size.
        """
        net = self.net
        index = self._index(obj_id)
        size = net.input_mask_size
        original = mask[None, None].float()
        inputs = original if mask.shape == (size, size) else F.interpolate(
            original, size=(size, size), align_corners=False, mode="bilinear", antialias=True)
        height, width = self.video_hw
        video = original if mask.shape == (height, width) else F.interpolate(
            original, size=(height, width), align_corners=False, mode="bilinear", antialias=True)
        video = video > 0.5

        self.inputs[index].add(frame)
        is_init = frame not in self.tracked
        key = _GIVEN
        # A given mask is taken as the output; the decoder only makes its pointer, reading no
        # memory, so the object's past is not needed here.
        out = self._run(frame, 1, is_init, None, inputs, _EMPTY, run_mem_encoder=False)
        out["pred_masks"] = None
        out["pred_masks_video_res"] = torch.where(video, -net.spec.no_object_score,
                                                  net.spec.no_object_score)
        self.temp[index][key][frame] = out
        # A later mask claims its pixels from the masks already given on this frame.
        for other, temp in enumerate(self.temp):
            if other == index:
                continue
            given = temp[key].get(frame)
            if given is not None and "pred_masks_video_res" in given:
                given["pred_masks_video_res"] = torch.where(video, net.spec.no_object_score,
                                                            given["pred_masks_video_res"])

    def add_points(self, frame: int, obj_id: int, points: torch.Tensor, labels: torch.Tensor) -> None:
        """Give an object clicks (and/or box corners) on ``frame``, in the encoder's square."""
        index = self._index(obj_id)
        self.inputs[index].add(frame)
        is_init = frame not in self.tracked
        key = _GIVEN
        if not is_init:
            raise RuntimeError("clicks are taken on the first frame only")
        out = self._run(frame, 1, is_init, {"point_coords": points, "point_labels": labels}, None,
                        _EMPTY, run_mem_encoder=False)
        self.temp[index][key][frame] = out

    @staticmethod
    def _slice(out: dict, index: int) -> dict:
        part = slice(index, index + 1)
        sliced = {
            "maskmem_features": None if out["maskmem_features"] is None
            else out["maskmem_features"][part],
            "maskmem_pos_enc": None if out["maskmem_pos_enc"] is None
            else [x[part] for x in out["maskmem_pos_enc"]],
            "pred_masks": out["pred_masks"][part],
            "obj_ptr": out["obj_ptr"][part],
            "object_score_logits": out["object_score_logits"][part],
            "iou_score": out["iou_score"][part],
        }
        return sliced

    # --- running the network -----------------------------------------------------------------

    def _run(self, frame: int, batch: int, is_init: bool, points, masks, store: dict,
             run_mem_encoder: bool) -> dict:
        """Upstream's ``_run_single_frame_inference``: one step, then a compact copy kept."""
        feats, pos = self.features(frame, batch)
        out = self.net.track_step(frame, is_init, feats, pos, points, masks, store,
                                  self.num_frames, run_mem_encoder)
        memory = out["maskmem_features"]
        if memory is not None:
            memory = memory.to(torch.bfloat16)
        compact = {
            "maskmem_features": memory,
            "maskmem_pos_enc": self._pos_enc(out["maskmem_pos_enc"]),
            "pred_masks": out["pred_masks"],
            "obj_ptr": out["obj_ptr"],
            "object_score_logits": out["object_score_logits"],
            "iou_score": out["iou_score"],
            "eff_iou_score": out["eff_iou_score"],
        }
        return compact

    def _pos_enc(self, pos_enc):
        """The memory position encoding is the same for every frame and object: keep one copy."""
        if pos_enc is None:
            return None
        if self._maskmem_pos_enc is None:
            self._maskmem_pos_enc = [x[0:1].clone() for x in pos_enc]
        return [x.expand(pos_enc[0].size(0), -1, -1, -1) for x in self._maskmem_pos_enc]

    def _encode(self, frame: int, masks: torch.Tensor, scores: torch.Tensor,
                from_points: bool) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Upstream's ``_run_memory_encoder``: encode a batch's masks, store as bfloat16."""
        feats, _ = self.features(frame, masks.size(0))
        memory, pos = self.net.encode_memory(feats[-1], masks, scores, from_points)
        return memory.to(torch.bfloat16), self._pos_enc(pos)

    def _empty_pointer(self, frame: int) -> torch.Tensor:
        """The pointer an empty mask gives: for an object with nothing on a consolidated frame."""
        size = self.net.image_size
        masks = torch.zeros((1, 1, size, size), dtype=torch.float32)
        feats, pos = self.features(frame, 1)
        masks = masks.to(feats[-1].device)
        out = self.net.track_step(frame, True, feats, pos, None, masks, _EMPTY,
                                  self.num_frames, run_mem_encoder=False)
        return out["obj_ptr"]

    # --- consolidation ---------------------------------------------------------------------

    def _consolidate(self, frame: int, key: str, run_mem_encoder: bool) -> dict:
        """Merge the objects' fresh outputs on ``frame`` into one batch output."""
        net = self.net
        batch = len(self.obj_ids)
        low = net.low_res
        device = self.features(frame, 1)[0][-1].device
        merged = {
            "maskmem_features": None,
            "maskmem_pos_enc": None,
            "pred_masks": torch.full((batch, 1, low, low), net.spec.no_object_score,
                                     dtype=torch.float32, device=device),
            "obj_ptr": torch.full((batch, net.hidden), net.spec.no_object_score,
                                  dtype=torch.float32, device=device),
            "object_score_logits": torch.full((batch, 1), 10.0, dtype=torch.float32, device=device),
            "iou_score": torch.full((batch, 1), 0.0, dtype=torch.float32, device=device),
        }
        empty = None
        for index in range(batch):
            out = self.temp[index][key].get(frame)
            if out is None:
                out = self.store["cond_frame_outputs"].get(frame)
                if out is None:
                    out = self.store["non_cond_frame_outputs"].get(frame)
                out = None if out is None else self._slice(out, index)
            if out is None:
                if run_mem_encoder:
                    if empty is None:
                        empty = self._empty_pointer(frame)
                    merged["obj_ptr"][index:index + 1] = empty
                continue
            mask = out.get("pred_masks_video_res", out["pred_masks"])
            if mask.shape[-2:] == merged["pred_masks"].shape[-2:]:
                merged["pred_masks"][index:index + 1] = mask
            else:
                merged["pred_masks"][index:index + 1] = F.interpolate(
                    mask, size=merged["pred_masks"].shape[-2:], mode="bilinear",
                    align_corners=False, antialias="pred_masks_video_res" in out)
            merged["obj_ptr"][index:index + 1] = out["obj_ptr"]
            merged["object_score_logits"][index:index + 1] = out["object_score_logits"]
            merged["iou_score"][index:index + 1] = out["iou_score"]
        if run_mem_encoder:
            high = F.interpolate(merged["pred_masks"], size=(net.image_size, net.image_size),
                                 mode="bilinear", align_corners=False)
            high = non_overlapping(high)
            merged["maskmem_features"], merged["maskmem_pos_enc"] = self._encode(
                frame, high, merged["object_score_logits"], from_points=True)
        return merged

    def preflight(self, run_mem_encoder: bool = True) -> None:
        """Fold the objects' fresh outputs into the batch's store (``propagate_in_video_preflight``)."""
        self.started = True
        for key in ("non_cond_frame_outputs", "cond_frame_outputs"):
            frames = set()
            for temp in self.temp:
                frames.update(temp[key].keys())
            self.consolidated[key].update(frames)
            for frame in frames:
                self.store[key][frame] = self._consolidate(frame, key, run_mem_encoder)
            for temp in self.temp:
                temp[key].clear()
        for frame in self.store["cond_frame_outputs"]:
            self.store["non_cond_frame_outputs"].pop(frame, None)
        for frame in self.consolidated["cond_frame_outputs"]:
            self.consolidated["non_cond_frame_outputs"].discard(frame)

    # --- tracking --------------------------------------------------------------------------

    def propagate(self, frame: int, run_mem_encoder: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Track the batch onto ``frame``: ``(B, 1, 288, 288)`` logits and ``(B, 1)`` scores.

        A frame already holding a consolidated output (it was given masks or clicks) returns that.
        """
        if not self.store["cond_frame_outputs"]:
            raise RuntimeError("nothing to track: no conditioning frame")
        if frame in self.consolidated["cond_frame_outputs"]:
            out = self.store["cond_frame_outputs"][frame]
        elif frame in self.consolidated["non_cond_frame_outputs"]:
            out = self.store["non_cond_frame_outputs"][frame]
        else:
            out = self._run(frame, len(self.obj_ids), False, None, None, self.store,
                            run_mem_encoder)
            self.store["non_cond_frame_outputs"][frame] = out
        self.tracked.add(frame)
        return out["pred_masks"], out["object_score_logits"]

    def set_memory(self, frame: int, masks: torch.Tensor, scores: torch.Tensor) -> None:
        """Encode this batch's (post-processed) masks on ``frame`` as its memory there."""
        memory, pos = self._encode(frame, masks, scores, from_points=False)
        for key in ("cond_frame_outputs", "non_cond_frame_outputs"):
            out = self.store[key].get(frame)
            if out is not None:
                out["maskmem_features"] = memory
                out["maskmem_pos_enc"] = list(pos)

    # --- removing objects ------------------------------------------------------------------

    def remove(self, obj_id: int) -> None:
        """Remove an object from every stored frame (upstream's ``remove_object``)."""
        if obj_id not in self.obj_ids:
            return
        index = self.obj_ids.index(obj_id)
        if len(self.obj_ids) == 1:
            self.obj_ids.clear()
            self.temp.clear()
            self.inputs.clear()
            self._reset()
            return
        for frame in sorted(self.inputs[index]):
            self._clear_inputs(frame, index)
        keep = [i for i in range(len(self.obj_ids)) if i != index]
        self.obj_ids = [self.obj_ids[i] for i in keep]
        self.temp = [self.temp[i] for i in keep]
        self.inputs = [self.inputs[i] for i in keep]
        for key in ("cond_frame_outputs", "non_cond_frame_outputs"):
            for out in self.store[key].values():
                out["maskmem_features"] = out["maskmem_features"][keep]
                out["maskmem_pos_enc"] = self._pos_enc([x[keep] for x in out["maskmem_pos_enc"]])
                out["pred_masks"] = out["pred_masks"][keep]
                out["obj_ptr"] = out["obj_ptr"][keep]
                out["object_score_logits"] = out["object_score_logits"][keep]
                out["iou_score"] = out["iou_score"][keep]
                out["eff_iou_score"] = self.net.memory_score(out["object_score_logits"],
                                                             out["iou_score"])

    def _clear_inputs(self, frame: int, index: int) -> None:
        """Upstream's ``clear_all_points_in_frame``: drop an object's prompt on a frame."""
        self.inputs[index].discard(frame)
        self.temp[index]["cond_frame_outputs"].pop(frame, None)
        self.temp[index]["non_cond_frame_outputs"].pop(frame, None)
        if any(frame in inputs for inputs in self.inputs):
            return
        self.consolidated["cond_frame_outputs"].discard(frame)
        self.consolidated["non_cond_frame_outputs"].discard(frame)
        out = self.store["cond_frame_outputs"].pop(frame, None)
        if out is not None:
            self.store["non_cond_frame_outputs"][frame] = out
            self.tracked.discard(frame)
        if not self.store["cond_frame_outputs"]:
            self._reset()

    def _reset(self) -> None:
        for inputs in self.inputs:
            inputs.clear()
        for temp in self.temp:
            temp["cond_frame_outputs"].clear()
            temp["non_cond_frame_outputs"].clear()
        self.store["cond_frame_outputs"].clear()
        self.store["non_cond_frame_outputs"].clear()
        self.consolidated["cond_frame_outputs"].clear()
        self.consolidated["non_cond_frame_outputs"].clear()
        self.started = False
        self.tracked.clear()

    # --- forgetting ------------------------------------------------------------------------

    def prune(self, frame: int) -> None:
        """Drop what no later frame can read (see the module text)."""
        cond = self.store["cond_frame_outputs"]
        if len(cond) > COND_KEPT:
            for t in sorted(cond)[:-COND_KEPT]:
                del cond[t]
        non_cond = self.store["non_cond_frame_outputs"]
        selectable = []
        threshold = self.net.spec.memory_selection_threshold
        for t in sorted(non_cond, reverse=True):
            out = non_cond[t]
            if "eff_iou_score" in out and out["eff_iou_score"] > threshold:
                selectable.append(t)
                if len(selectable) >= self.net.spec.max_obj_ptrs:
                    break
        keep = set(selectable)
        for t in list(non_cond):
            if t not in keep and t < frame - RECENT:
                del non_cond[t]
        for t in list(self.tracked):
            if t < frame - RECENT and t not in cond:
                self.tracked.discard(t)
        for key in self.consolidated:
            self.consolidated[key] = {t for t in self.consolidated[key]
                                      if t in self.store[key] or t >= frame - RECENT}

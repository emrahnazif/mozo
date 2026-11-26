# SPDX-License-Identifier: Apache-2.0
"""Concept tracking: one text prompt, followed through a video, instances coming and going.

Derived from ``transformers/models/sam3_video`` (``Sam3VideoModel``, Apache-2.0) and diverging
from it where Meta's published model behaves otherwise (``PROVENANCE.md``): objects are tracked in
per-arrival batches (:class:`.group.ObjectGroup`), NMS and hole filling run (:mod:`.ops`), and the
output for a frame is computed from that frame and the ones before it only -- Meta's hotstart
buffer, which holds output back fifteen frames, is not reproduced; its decisions are (below).

Per frame:

1. **Detect**: the concept head on the frame (done by the caller, shared by nothing -- one prompt).
2. **Track**: every batch is propagated onto the frame; small holes and specks are cleaned.
3. **Match**: detections against tracks by mask IoU. An unmatched detection above 0.7 starts a
   new object (on the very first frame, every detection does).
4. **Probation** ("hotstart"): an object younger than fifteen frames that went unmatched on eight
   frames, or kept overlapping an older object (both matched to one detection) on eight, is removed.
5. **Re-seed**: every sixteenth frame, a track confidently matched (score and IoU above 0.8) by a
   confident detection is given that detection's mask, re-encoded with its whole batch.
6. **Occlusion**: of two tracks overlapping above 0.7 IoU, the one occluded more recently is blanked
   on this frame.
7. **Remember**: the tracks' masks (after the above, and after resolving overlaps pixel by pixel)
   are encoded as this frame's memory.
8. **Add / remove** objects, then build the frame's masks: tracks from step 6, new objects from their
   detections.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from ..config import VIDEO, VideoSpec
from .group import ObjectGroup, non_overlapping
from .network import TrackerNet
from .ops import fill_holes, mask_iou

__all__ = ["ConceptSession", "FrameOutput"]

#: How long an object's probation bookkeeping is kept past its probation.
_FORGET = 2


class FrameOutput:
    """One frame's result for one prompt.

    Attributes:
        obj_ids: Object ids shown on this frame, ascending.
        masks: ``(N, H, W)`` bool at the frame's size, after the object-wise overlap rule.
        scores: ``(N,)`` the tracker's visibility probability on this frame (the detection score
            on the frame an object starts).
        spawn_scores: ``(N,)`` the detection score each object started with.
    """

    def __init__(self, obj_ids, masks, scores, spawn_scores):
        self.obj_ids = obj_ids
        self.masks = masks
        self.scores = scores
        self.spawn_scores = spawn_scores


class ConceptSession:
    """Tracking state for one text prompt.

    Args:
        net: The tracker.
        features: ``features(frame, batch)``, as :class:`.group.ObjectGroup` takes it.
        video_hw: Frame ``(height, width)``.
        spec: The decision thresholds.
        num_frames: Video length, when known (the gate only).
        trace: Optional ``trace(stage, frame, value)`` callback, for the gate.
    """

    def __init__(self, net: TrackerNet, features, video_hw, spec: VideoSpec = VIDEO,
                 num_frames: int | None = None, trace=None):
        self.net = net
        self.features = features
        self.video_hw = video_hw
        self.spec = spec
        self.num_frames = num_frames
        self.trace = trace or (lambda *a: None)
        self.groups: list[ObjectGroup] = []
        self.obj_ids = np.array([], np.int64)
        self.max_obj_id = -1
        self.spawn_score: dict[int, float] = {}
        self.last_occluded: dict[int, torch.Tensor] = {}
        self.first_frame: dict[int, int] = {}
        self.unmatched: defaultdict[int, list] = defaultdict(list)
        self.overlaps: defaultdict[tuple, list] = defaultdict(list)
        self.removed: set[int] = set()

    # --- step 2 ----------------------------------------------------------------------------

    def _propagate(self, frame: int):
        masks, scores = [], []
        for group in self.groups:
            if not len(group):
                continue
            low, score = group.propagate(frame, run_mem_encoder=False)
            masks.append(low.squeeze(1))
            scores.append(score.squeeze(1))
        low = self.net.low_res
        if masks:
            masks = fill_holes(torch.cat(masks, dim=0).unsqueeze(1), self.spec.hole_area).squeeze(1)
            scores = torch.cat(scores, dim=0)
        else:
            device = self.features(frame, 1)[0][-1].device
            masks = torch.zeros(0, low, low, device=device)
            scores = torch.zeros(0, device=device)
        return masks, scores

    # --- step 3 ----------------------------------------------------------------------------

    def _associate(self, det_masks, det_scores, trk_masks, trk_ids):
        spec = self.spec
        if trk_masks.size(0) == 0:
            return (np.arange(det_masks.size(0)), np.array([], np.int64), {}, {},
                    np.array([], np.int64))
        if det_masks.size(0) == 0:
            nonempty = (trk_masks > 0).any(dim=(1, 2)).cpu().numpy()
            return (np.array([], np.int64), trk_ids[nonempty], {}, {}, trk_ids[~nonempty])
        if det_masks.shape[-2:] != trk_masks.shape[-2:]:
            raise ValueError("detection and track masks must share a resolution")
        ious = mask_iou(det_masks > 0, trk_masks > 0).cpu().numpy()
        trk_matched = (ious >= spec.track_match_iou).any(axis=0)
        nonempty = (trk_masks > 0).any(dim=(1, 2)).cpu().numpy()
        unmatched = trk_ids[np.logical_and(nonempty, ~trk_matched)]
        empty = trk_ids[~nonempty]
        is_new = np.logical_and(det_scores >= spec.new_object_threshold,
                                np.logical_not(np.any(ious >= spec.association_iou, axis=1)))
        new = np.nonzero(is_new)[0]
        matched = {}
        reseed = {}
        best = np.argmax(ious, axis=1)
        confident = (det_scores >= spec.high_confidence) & ~is_new
        close = np.max(ious, axis=1) >= spec.high_iou
        both = set(np.nonzero(confident & close)[0])
        for d in range(det_masks.size(0)):
            matched[d] = trk_ids[ious[d, :] >= spec.association_iou]
            if d in both:
                reseed[trk_ids[best[d]].item()] = d
        return new, unmatched, matched, reseed, empty

    # --- step 4 ----------------------------------------------------------------------------

    def _probation(self, frame, matched, new_ids, unmatched):
        spec = self.spec
        removed_now = set()
        horizon = frame - spec.hotstart_delay
        for obj_id in new_ids:
            if obj_id not in self.first_frame:
                self.first_frame[obj_id] = frame
        for obj_id in unmatched:
            self.unmatched[obj_id].append(frame)
        for obj_id, frames in self.unmatched.items():
            if obj_id in self.removed or obj_id in removed_now:
                continue
            if len(frames) >= spec.hotstart_unmatched and self.first_frame[obj_id] > horizon:
                removed_now.add(obj_id)
        for ids in matched.values():
            if len(ids) < 2:
                continue
            first = min(ids, key=lambda x: self.first_frame[x])
            for obj_id in ids:
                if obj_id != first:
                    self.overlaps[(first, obj_id)].append(frame)
        for (first, obj_id), frames in self.overlaps.items():
            if obj_id in self.removed or obj_id in removed_now:
                continue
            if self.first_frame[obj_id] > horizon and len(frames) >= spec.hotstart_duplicate:
                removed_now.add(obj_id)
        self.removed.update(removed_now)
        return removed_now

    # --- step 5 ----------------------------------------------------------------------------

    def _reseed(self, frame, det_masks, reseed, trk_scores):
        size = self.net.input_mask_size
        for obj_id, d in reseed.items():
            mask = F.interpolate(det_masks[d:d + 1].unsqueeze(1), size=(size, size),
                                 mode="bilinear", align_corners=False).squeeze(1)[0] > 0
            index = np.where(self.obj_ids == obj_id)[0].item()
            score = trk_scores[index]
            touched = []
            for position, group in enumerate(self.groups):
                if obj_id in group.obj_ids and score > self.spec.high_confidence:
                    group.add_mask(frame, obj_id, mask)
                    touched.append(position)
            for position in touched:
                self.groups[position].preflight(run_mem_encoder=True)

    # --- step 6 ----------------------------------------------------------------------------

    def _suppress_occluded(self, frame, masks, removed_now, last_occluded_new):
        never, always = -1, 100000
        binary = masks > 0
        last = torch.cat([
            self.last_occluded.get(obj_id, torch.full(
                (1,), fill_value=always if obj_id in removed_now else never,
                device=binary.device, dtype=torch.long))
            for obj_id in self.obj_ids
        ], dim=0)
        suppress = torch.zeros(binary.size(0), device=binary.device, dtype=torch.bool)
        if len(self.obj_ids) > 1:
            iou = mask_iou(binary, binary)
            pairs = torch.triu(iou >= self.spec.occlusion_suppression_iou, diagonal=1)
            last_i, last_j = last.unsqueeze(1), last.unsqueeze(0)
            suppress_i = pairs & torch.gt(last_i, last_j) & (last_j > -1)
            suppress_j = pairs & torch.gt(last_j, last_i) & (last_i > -1)
            suppress = suppress_i.any(dim=1) | suppress_j.any(dim=0)
        occluded = ~(binary.any(dim=(-1, -2)))
        updated = last.clone()
        updated[occluded | suppress] = frame
        last_occluded_new.clear()
        last_occluded_new.update({obj_id: updated[i:i + 1] for i, obj_id in enumerate(self.obj_ids)})
        masks[suppress] = self.spec.occluded_logit
        return masks

    # --- step 7 ----------------------------------------------------------------------------

    def _remember(self, frame, masks):
        size = self.net.input_mask_size
        high = F.interpolate(masks.unsqueeze(1), size=(size, size), mode="bilinear",
                             align_corners=False)
        resolved = non_overlapping(high)
        before = (high > 0).sum(dim=(-1, -2))
        after = (resolved > 0).sum(dim=(-1, -2))
        ratio = after / torch.clamp(before, min=1.0)
        keep = (ratio >= self.spec.shrink_ratio)[..., None, None].expand_as(high)
        high = torch.where(keep, high, torch.clamp(high, max=self.spec.occluded_logit))
        scores = torch.where((high > 0).any(dim=(-1, -2)), 10.0, -10.0)
        start = 0
        for group in self.groups:
            count = len(group)
            if count == 0:
                continue
            group.set_memory(frame, high[start:start + count], scores[start:start + count])
            start += count

    # --- the frame -------------------------------------------------------------------------

    def step(self, frame: int, det_masks: torch.Tensor, det_scores: torch.Tensor) -> FrameOutput:
        """Track onto one frame.

        Args:
            det_masks: ``(D, 288, 288)`` logits of the detections kept (score above threshold,
                after NMS), in query order.
            det_scores: ``(D,)`` their scores.
        """
        spec = self.spec
        trk_masks, trk_scores = self._propagate(frame)
        self.trace("propagate", frame, (trk_masks, trk_scores))

        scores_np = det_scores.float().cpu().numpy()
        prev_ids = self.obj_ids
        new_inds, unmatched, matched, reseed, _ = self._associate(det_masks, scores_np, trk_masks,
                                                                  prev_ids)
        new_ids = self.max_obj_id + 1 + np.arange(len(new_inds))
        removed_now = self._probation(frame, matched, new_ids, unmatched)
        self.trace("plan", frame, (new_inds, new_ids, unmatched, matched, reseed, removed_now))

        if frame % spec.recondition_every == 0 and reseed:
            self._reseed(frame, det_masks, reseed, trk_scores)

        last_occluded_new: dict = {}
        if trk_masks.size(0) > 0:
            trk_masks = self._suppress_occluded(frame, trk_masks, removed_now, last_occluded_new)
            self._remember(frame, trk_masks)

        # Metadata: the new ids join, the removed ones leave.
        ids = prev_ids
        if len(new_ids):
            ids = np.concatenate([ids, new_ids])
        if removed_now:
            ids = ids[~np.isin(ids, list(removed_now))]
        self.obj_ids = ids
        tracker_scores = {}
        for obj_id, d in zip(new_ids, new_inds):
            self.spawn_score[obj_id] = scores_np[d]
            tracker_scores[obj_id] = scores_np[d]
        if len(new_ids):
            self.max_obj_id = max(self.max_obj_id, int(np.max(new_ids)))
        for obj_id in removed_now:
            self.spawn_score[obj_id] = -1e4
            last_occluded_new.pop(obj_id, None)
        self.last_occluded = last_occluded_new
        tracker_scores.update(dict(zip(prev_ids.tolist(), trk_scores.sigmoid().tolist())))

        # Add the new objects as one batch, then remove the removed.
        if len(new_inds):
            self._add_objects(frame, det_masks, new_inds, new_ids)
        for obj_id in removed_now:
            for group in self.groups:
                group.remove(obj_id)
            self.groups = [g for g in self.groups if len(g)]

        out = self._outputs(frame, prev_ids, trk_masks, det_masks, new_inds, new_ids,
                            tracker_scores)
        self._forget(frame)
        return out

    def _add_objects(self, frame, det_masks, new_inds, new_ids) -> None:
        """Start the frame's new objects, together, as one new batch seeded by their detections."""
        group = ObjectGroup(self.net, self.features, self.video_hw, self.num_frames)
        size = self.net.input_mask_size
        seeds = F.interpolate(det_masks[torch.from_numpy(new_inds)].unsqueeze(1),
                              size=(size, size), mode="bilinear", align_corners=False)
        for obj_id, seed in zip(new_ids, seeds.squeeze(1) > 0):
            group.add_mask(frame, int(obj_id), seed)
        group.preflight(run_mem_encoder=True)
        self.groups.append(group)

    def _outputs(self, frame, prev_ids, trk_masks, det_masks, new_inds, new_ids, tracker_scores):
        height, width = self.video_hw
        masks = {}
        if len(prev_ids):
            video = F.interpolate(trk_masks.unsqueeze(1), size=(height, width), mode="bilinear",
                                  align_corners=False) > 0
            masks.update(zip(prev_ids.tolist(), video))
        if len(new_ids):
            seeds = fill_holes(det_masks[torch.from_numpy(new_inds)].unsqueeze(1),
                               self.spec.hole_area)
            video = F.interpolate(seeds, size=(height, width), mode="bilinear",
                                  align_corners=False) > 0
            masks.update(zip(new_ids.tolist(), video))
        self.trace("masks", frame, masks)

        ids = sorted(masks)
        device = trk_masks.device
        if not ids:
            return FrameOutput([], torch.zeros(0, height, width, dtype=torch.bool), torch.zeros(0),
                               torch.zeros(0))
        all_ids = torch.tensor(ids, dtype=torch.int64)
        spawn = torch.tensor([self.spawn_score[i] for i in ids])
        tracker = torch.tensor([tracker_scores.get(i, 0.0) for i in ids])
        binary = torch.cat([masks[i] for i in ids], dim=0)
        keep = binary.any(dim=(1, 2)).cpu()
        if self.removed:
            keep &= ~torch.isin(all_ids, torch.tensor(sorted(self.removed), dtype=torch.int64))
        index = torch.nonzero(keep, as_tuple=True)[0]
        all_ids = all_ids[index]
        spawn = spawn[index]
        tracker = tracker[index]
        binary = binary[index.to(binary.device)]
        if binary.shape[0] > 1:
            # One object per pixel, by the tracker's score (upstream's object-wise rule).
            single = torch.where(binary.unsqueeze(1) > 0, tracker.unsqueeze(1)[..., None, None]
                                 .to(binary.device), 0.0)
            resolved = non_overlapping(single)
            binary = (torch.where(resolved > 0, binary.unsqueeze(1).float(),
                                  torch.clamp(binary.unsqueeze(1).float(), max=0.0)) > 0)[:, 0]
        del device
        return FrameOutput(all_ids.tolist(), binary, tracker, spawn)

    def _forget(self, frame):
        """Drop probation bookkeeping no later frame can act on, and old batch memory."""
        horizon = frame - self.spec.hotstart_delay - _FORGET
        live = set(self.obj_ids.tolist())
        for obj_id in list(self.unmatched):
            if obj_id not in live or self.first_frame.get(obj_id, frame) <= horizon:
                del self.unmatched[obj_id]
        for key in list(self.overlaps):
            if key[1] not in live or self.first_frame.get(key[1], frame) <= horizon:
                del self.overlaps[key]
        for obj_id in list(self.first_frame):
            if obj_id not in live:
                del self.first_frame[obj_id]
        for obj_id in list(self.spawn_score):
            if obj_id not in live:
                del self.spawn_score[obj_id]
        for group in self.groups:
            group.prune(frame)

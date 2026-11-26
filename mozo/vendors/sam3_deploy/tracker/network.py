# SPDX-License-Identifier: Apache-2.0
"""One tracking step: condition a frame on an object's memories, decode, encode a new memory.

Derived from ``transformers/models/sam3_tracker_video`` (``Sam3TrackerVideoModel``, Apache-2.0).
The decoder and prompt encoder are the click head's own (:class:`..click.ClickHead`): one set of
weights serves a click on a photograph and every frame of a video.

What a frame reads, per object, is decided here (:meth:`TrackerNet.condition`):

- up to four **conditioning frames** -- frames where the object was given (by a detection, a
  click, or a re-seed) rather than tracked -- the temporally closest, at temporal position 0;
- up to six **recent memories**, chosen by *memory selection*: walking back from the previous
  frame, only frames whose score (visibility times predicted IoU, averaged over the objects
  tracked together) is above 0.01 count, so an occluded stretch does not fill the memory with
  nothing; the previous frame is always included;
- up to fifteen **object pointers** (one 256-vector per frame, split into four 64-tokens) from the
  same selected frames, with a sine encoding of how far back each one is.

All numbers are float32 except the stored memories, which are bfloat16 (as upstream stores them,
on every device).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import TRACKER, TrackerSpec
from ..grounding.layers import Mlp
from .memory import MemoryAttention, MemoryEncoder

__all__ = ["TrackerNet", "closest_cond_frames", "sine_1d"]


def sine_1d(positions: Tensor, dim: int, temperature: float = 10000) -> Tensor:
    """1-D sine/cosine encoding (sines first, then cosines), as for object-pointer distances."""
    half = dim // 2
    bands = torch.arange(half, dtype=torch.float32, device=positions.device)
    bands = temperature ** (2 * (bands // 2) / half)
    scaled = positions.unsqueeze(-1) / bands
    return torch.cat([scaled.sin(), scaled.cos()], dim=-1)


def closest_cond_frames(frame: int, cond: dict, limit: int) -> tuple[dict, dict]:
    """Split the conditioning frames into the (at most) ``limit`` a frame attends to and the rest.

    The closest before ``frame``, the closest at or after it, then the temporally closest others.
    """
    if limit == -1 or len(cond) <= limit:
        return cond, {}
    selected = {}
    before = max((t for t in cond if t < frame), default=None)
    if before is not None:
        selected[before] = cond[before]
    after = min((t for t in cond if t >= frame), default=None)
    if after is not None:
        selected[after] = cond[after]
    remain = limit - len(selected)
    others = sorted((t for t in cond if t not in selected), key=lambda x: abs(x - frame))[:remain]
    selected.update((t, cond[t]) for t in others)
    return selected, {t: v for t, v in cond.items() if t not in selected}


class TrackerNet(nn.Module):
    """The tracker's own weights, plus the click head it decodes with.

    Args:
        click: The loaded :class:`..click.ClickHead` (prompt encoder and mask decoder).
        spec: Tracker geometry.
    """

    def __init__(self, click: nn.Module, spec: TrackerSpec = TRACKER):
        super().__init__()
        self.spec = spec
        self.hidden = spec.hidden
        self.memory_dim = spec.memory_dim
        self.num_maskmem = spec.num_maskmem
        self.image_size = click.image_size
        self.low_res = 4 * click.grid
        self.input_mask_size = spec.mask_input_size

        self.memory_attention = MemoryAttention(spec, click.grid)
        self.memory_encoder = MemoryEncoder(spec)
        self.mask_downsample = nn.Conv2d(1, 1, kernel_size=4, stride=4)
        self.obj_ptr_proj = Mlp((spec.hidden,) * 4)
        self.obj_ptr_tpos_proj = nn.Linear(spec.hidden, spec.memory_dim)
        self.maskmem_tpos_enc = nn.Parameter(torch.zeros(spec.num_maskmem, 1, 1, spec.memory_dim))
        self.no_mem_pos_enc = nn.Parameter(torch.zeros(1, 1, spec.hidden))
        self.no_obj_ptr = nn.Parameter(torch.zeros(1, spec.hidden))
        self.no_obj_embed_spatial = nn.Parameter(torch.zeros(1, spec.memory_dim))
        # Shared, not owned: its weights load (and are counted) with the click head.
        object.__setattr__(self, "click", click)

    # --- the decoder ---------------------------------------------------------------------------

    def heads(self, features: Tensor, high_res: list[Tensor], points: Tensor | None,
              labels: Tensor | None, mask_inputs: Tensor | None, multimask: bool) -> dict:
        """Decode a batch of objects, one image (or one conditioned image) per object.

        Args:
            features: ``(B, 256, 72, 72)``.
            high_res: The two finer levels, already projected, ``(B, 32, 288, 288)`` and
                ``(B, 64, 144, 144)``.
            points: ``(B, P, 2)`` in the encoder's square, or ``None`` (an empty prompt).
            labels: ``(B, P)``.
            mask_inputs: ``(B, 1, 288, 288)`` dense mask prompt, or ``None``.
            multimask: Return the best of three candidates rather than the single-mask token.
        """
        batch = features.shape[0]
        device = features.device
        if points is None:
            points = torch.zeros(batch, 1, 2, device=device)
            labels = -torch.ones(batch, 1, dtype=torch.int32, device=device)
        encoder = self.click.prompt_encoder
        if mask_inputs is not None and mask_inputs.shape[-2:] != (encoder.mask_input_size,) * 2:
            mask_inputs = F.interpolate(mask_inputs.float(), size=(encoder.mask_input_size,) * 2,
                                        align_corners=False, mode="bilinear", antialias=True)
        sparse, dense = encoder(points, labels, mask_inputs)
        # The prompt encoder puts a batch of prompts on the prompt axis of one image; here each
        # object is its own image with one prompt.
        sparse = sparse.transpose(0, 1)
        if not self.click._dense_pe_ready:
            self.click.dense_pe.copy_(encoder.dense_positions())
            self.click._dense_pe_ready = True
        decoder = self.click.mask_decoder
        masks, ious, tokens, object_score = decoder.decode(
            features, self.click.dense_pe, sparse, dense, high_res[0], high_res[1], projected=True)
        masks, ious, tokens, object_score = masks[:, 0], ious[:, 0], tokens[:, 0], object_score[:, 0]

        if multimask:
            masks, ious = masks[:, 1:], ious[:, 1:]
            tokens = tokens[:, 1:]
        else:
            masks, ious = decoder._stable(masks[:, None], ious[:, None])
            masks, ious = masks[:, 0], ious[:, 0]
            tokens = tokens[:, 0:1]

        appearing = object_score > 0
        multimasks = torch.where(appearing[:, None, None], masks, self.spec.no_object_score)
        multimasks = multimasks.float()
        high = F.interpolate(multimasks, size=(self.image_size, self.image_size),
                             mode="bilinear", align_corners=False)
        token = tokens[:, 0]
        if multimask:
            best = torch.argmax(ious, dim=-1)
            rows = torch.arange(batch, device=device)
            low_res = multimasks[rows, best].unsqueeze(1)
            high = high[rows, best].unsqueeze(1)
            if tokens.size(1) > 1:
                token = tokens[rows, best]
        else:
            low_res = multimasks

        pointer = self.obj_ptr_proj(token)
        weight = appearing.float()
        pointer = weight * pointer
        pointer = pointer + (1 - weight) * self.no_obj_ptr
        return {"pred_masks": low_res, "pred_masks_high_res": high, "ious": ious,
                "obj_ptr": pointer, "object_score_logits": object_score}

    def mask_as_output(self, features: Tensor, high_res: list[Tensor], masks: Tensor) -> dict:
        """Take a given mask as the output, and only ask the decoder for its object pointer."""
        scale, bias = 20.0, -10.0
        masks = masks.float()
        high = masks * scale + bias
        low_res = F.interpolate(
            high, size=(high.size(-2) // 14 * 4, high.size(-1) // 14 * 4),
            align_corners=False, mode="bilinear", antialias=True)
        ious = masks.new_ones(masks.size(0), 1).float()
        pointer = self.heads(features, high_res, None, None, self.mask_downsample(masks),
                             multimask=False)["obj_ptr"]
        appearing = torch.any(masks.flatten(1).float() > 0.0, dim=1)[..., None]
        weight = appearing.float()
        object_score = scale * weight + bias
        pointer = weight * pointer
        pointer = pointer + (1 - weight) * self.no_obj_ptr
        return {"pred_masks": low_res, "pred_masks_high_res": high, "ious": ious,
                "obj_ptr": pointer, "object_score_logits": object_score}

    # --- memory ----------------------------------------------------------------------------------

    @staticmethod
    def memory_score(object_score: Tensor, iou: Tensor) -> Tensor:
        """A frame's worth as a memory: visibility rescaled to [0, 1] times IoU, over objects."""
        visible = torch.where(object_score > 0, object_score.sigmoid() * 2 - 1,
                              torch.zeros_like(object_score))
        return (visible * iou).mean()

    def select_frames(self, outputs: dict, frame: int, reach: int) -> list[int]:
        """Memory selection: the recent non-conditioning frames worth reading, oldest first."""
        if frame == 0:
            return []
        chosen = []
        for t in range(frame - 1, 0, -1):
            out = outputs.get(t)
            if out is None or "eff_iou_score" not in out:
                continue
            if out["eff_iou_score"] > self.spec.memory_selection_threshold:
                chosen.insert(0, t)
            if len(chosen) >= reach - 1:
                break
        if frame - 1 not in chosen:
            chosen.append(frame - 1)
        return chosen

    def condition(self, frame: int, is_init: bool, feats: Tensor, pos: Tensor, store: dict,
                  num_frames: int | None) -> Tensor:
        """Fuse the current frame's coarsest features with an object batch's memories.

        Args:
            feats: ``(HW, B, 256)``.
            pos: ``(HW, B, 256)``.
            store: ``cond_frame_outputs`` / ``non_cond_frame_outputs`` dicts.
            num_frames: The video's length when known (only the gate passes it).

        Returns:
            ``(B, 256, 72, 72)``.
        """
        batch = feats.size(1)
        side = int(feats.size(0) ** 0.5)
        if is_init:
            return (feats + self.click.no_mem_embed).permute(1, 2, 0).view(batch, self.hidden,
                                                                          side, side)
        reach = self.spec.max_obj_ptrs if num_frames is None else min(num_frames,
                                                                     self.spec.max_obj_ptrs)
        prompt, prompt_pos = [], []
        selected, unselected = closest_cond_frames(frame, store["cond_frame_outputs"],
                                                   self.spec.max_cond_frames)
        slots = [(0, out) for out in selected.values()]
        valid = self.select_frames(store["non_cond_frame_outputs"], frame, reach)
        for t_pos in range(1, self.num_maskmem):
            back = self.num_maskmem - t_pos
            if back > len(valid):
                continue
            t = valid[-back]
            out = store["non_cond_frame_outputs"].get(t, unselected.get(t))
            slots.append((t_pos, out))
        for t_pos, out in slots:
            if out is None:
                continue
            memory = out["maskmem_features"]
            prompt.append(memory.flatten(2).permute(2, 0, 1))
            enc = out["maskmem_pos_enc"][-1].flatten(2).permute(2, 0, 1)
            enc = enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
            prompt_pos.append(enc)

        pointers = [(frame - t, out["obj_ptr"]) for t, out in selected.items() if t <= frame]
        for t_diff in range(1, reach):
            if -t_diff <= -len(valid):
                break
            t = valid[-t_diff]
            out = store["non_cond_frame_outputs"].get(t, unselected.get(t))
            if out is not None:
                pointers.append((t_diff, out["obj_ptr"]))
        tokens = 0
        if pointers:
            distances, vectors = zip(*pointers)
            ptrs = torch.stack(vectors, dim=0)
            ptr_pos = torch.tensor(distances).to(device=feats.device) / (reach - 1)
            ptr_pos = self.obj_ptr_tpos_proj(sine_1d(ptr_pos, self.hidden))
            ptr_pos = ptr_pos.unsqueeze(1).expand(-1, batch, -1)
            split = self.hidden // self.memory_dim
            ptrs = ptrs.reshape(-1, batch, split, self.memory_dim).permute(0, 2, 1, 3).flatten(0, 1)
            ptr_pos = ptr_pos.repeat_interleave(split, dim=0)
            prompt.append(ptrs)
            prompt_pos.append(ptr_pos)
            tokens = ptrs.shape[0]

        memory = torch.cat(prompt, dim=0)
        memory_pos = torch.cat(prompt_pos, dim=0)
        fused = self.memory_attention(feats, pos, memory, memory_pos, tokens)
        return fused.permute(1, 2, 0).view(batch, self.hidden, side, side)

    def encode_memory(self, feats: Tensor, high_res_masks: Tensor, object_score: Tensor,
                      from_points: bool) -> tuple[Tensor, list[Tensor]]:
        """Encode masks into memories, with the no-object embedding where the object is absent.

        Args:
            feats: ``(HW, B, 256)`` the coarsest features.
            high_res_masks: ``(B, 1, H, W)`` logits.
            object_score: ``(B, 1)``.
            from_points: The masks came from a prompt: binarise rather than squash.
        """
        batch = feats.size(1)
        side = int(feats.size(0) ** 0.5)
        pix = feats.permute(1, 2, 0).view(batch, self.hidden, side, side)
        masks = (high_res_masks > 0).float() if from_points else torch.sigmoid(high_res_masks)
        masks = masks * self.spec.sigmoid_scale
        masks = masks + self.spec.sigmoid_bias
        memory, pos = self.memory_encoder(pix, masks)
        appearing = (object_score > 0).float()
        memory += (1 - appearing[..., None, None]) * self.no_obj_embed_spatial[..., None, None].expand(
            *memory.shape)
        return memory, [pos]

    # --- one step ------------------------------------------------------------------------------

    def track_step(self, frame: int, is_init: bool, feats: list[Tensor], pos: list[Tensor],
                   points: dict | None, mask_inputs: Tensor | None, store: dict,
                   num_frames: int | None, run_mem_encoder: bool,
                   prev_mask_logits: Tensor | None = None) -> dict:
        """Upstream's ``track_step`` for a batch of objects on one frame.

        Args:
            feats: The three levels, ``(HW, B, C)``, finest first (the finer two projected).
            pos: The coarsest level's position encoding is the only one read.
            points: ``point_coords``/``point_labels`` in the encoder's square, or ``None``.
            mask_inputs: ``(B, 1, 1152, 1152)`` given masks, or ``None``.
        """
        sizes = [(int(f.size(0) ** 0.5),) * 2 for f in feats]
        high_res = [f.permute(1, 2, 0).view(f.size(1), f.size(2), *s)
                    for f, s in zip(feats[:-1], sizes[:-1])]
        if mask_inputs is not None:
            pix = feats[-1].permute(1, 2, 0).view(-1, self.hidden, *sizes[-1])
            out = self.mask_as_output(pix, high_res, mask_inputs)
        else:
            pix = self.condition(frame, is_init, feats[-1], pos[-1], store, num_frames)
            if prev_mask_logits is not None:
                mask_inputs = prev_mask_logits
            # Three candidates for no prompt or a single click, on every frame (upstream's
            # ``multimask_output_for_tracking``); the stable single mask for anything more.
            count = 0 if points is None else points["point_labels"].size(1)
            multimask = count <= 1
            out = self.heads(pix, high_res, None if points is None else points["point_coords"],
                             None if points is None else points["point_labels"], mask_inputs,
                             multimask)
        iou = out["ious"].max(-1)[0]
        result = {"pred_masks": out["pred_masks"], "pred_masks_high_res": out["pred_masks_high_res"],
                  "obj_ptr": out["obj_ptr"], "object_score_logits": out["object_score_logits"],
                  "iou_score": iou,
                  "eff_iou_score": self.memory_score(out["object_score_logits"], iou)}
        if run_mem_encoder:
            memory, memory_pos = self.encode_memory(feats[-1], out["pred_masks_high_res"],
                                                    out["object_score_logits"], points is not None)
            result["maskmem_features"], result["maskmem_pos_enc"] = memory, memory_pos
        else:
            result["maskmem_features"], result["maskmem_pos_enc"] = None, None
        return result

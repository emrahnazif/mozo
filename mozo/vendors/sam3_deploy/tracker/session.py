# SPDX-License-Identifier: Apache-2.0
"""A video tracking session, one frame at a time. Built by :meth:`..predictor.Segmenter.tracker`.

Two kinds of session, decided by the first call:

- **Text** (concept tracking): one or more phrases on the first frame. Each phrase is tracked as
  its own session -- exactly what Meta's model does with that phrase alone, since it takes one
  phrase per session -- and the phrases share each frame's image encode, the expensive part.
- **Clicks**: points and/or boxes on the first frame, one object each, followed by SAM 3's tracker
  without the detector (Meta's "SAM 3 for the SAM 2 task"). Upstream batches these objects
  together and takes no new ones once tracking has started, so neither does this.

**The first frame of a text session is visited twice**, because that is what Meta's API does to
every caller: ``add_prompt`` runs the frame, then ``propagate_in_video`` starts from that same
frame and runs it again on the state the first visit left (the second visit re-seeds the objects
the first one found, and encodes their memory from the tracker's masks). The second visit is what
is returned.

**Two numeric details are Meta's, kept** (and the first is the only accommodation of the
reference that touches a number; ``PROVENANCE.md``):

- In a text session the tracker's image features are rounded to bfloat16 before use: Meta casts
  them for a cross-GPU gather on every run, one GPU included. Its own runs then continue in
  bfloat16 autocast; this package continues in float32.
- Frames are preprocessed as Meta's loaders do: in a text session as its video loader does, in
  float16 (:func:`.ops.load_frame`); in a click session as its tracker's own loader does, SAM 2's
  way (:func:`.ops.load_click_frame`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..config import VIDEO, VideoSpec
from ..image import to_model_coords
from .concept import ConceptSession
from .group import ObjectGroup
from .ops import load_click_frame, load_frame, nms

__all__ = ["Frame", "VideoTracker", "split_prompts"]


@dataclass
class Frame:
    """One frame's result.

    Attributes:
        obj_ids: Object ids shown on this frame, ascending: 0, 1, 2… in order of first appearance,
            across all prompts, never reused.
        prompts: ``(N,)`` which prompt each object belongs to (its index).
        masks: ``(N, H, W)`` bool at the frame's size.
        scores: ``(N,)`` the probability the object is visible on this frame.
    """

    obj_ids: list[int]
    prompts: list[int]
    masks: torch.Tensor
    scores: torch.Tensor


def split_prompts(points=None, labels=None, boxes=None) -> list[dict]:
    """Turn :meth:`..predictor.Segmenter.segment`-shaped prompts into one prompt per new object.

    ``points`` ``(N, 2)`` or ``(B, N, 2)`` with ``labels`` ``(N,)`` or ``(B, N)``, ``boxes``
    ``(4,)`` or ``(B, 4)``; given together, one point set per box. Prompt *b* becomes object *b*.

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


def _inverse_sigmoid(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    x = x.clamp(min=0, max=1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


class _Features:
    """The current frame's tracker features, handed out expanded to a batch as upstream does."""

    def __init__(self):
        self.frame = -1
        self.levels: list[torch.Tensor] = []
        self.positions: torch.Tensor | None = None

    def set(self, frame: int, levels: list[torch.Tensor], positions: torch.Tensor) -> None:
        self.frame, self.levels, self.positions = frame, levels, positions

    def __call__(self, frame: int, batch: int):
        if frame != self.frame:
            raise RuntimeError(f"features for frame {frame} asked for on frame {self.frame}")
        feats = [x.expand(batch, -1, -1, -1).flatten(2).permute(2, 0, 1) for x in self.levels]
        pos = self.positions.expand(batch, -1, -1, -1).flatten(2).permute(2, 0, 1)
        return feats, [pos] * len(feats)


class VideoTracker:
    """Follow concepts, or clicked objects, through a stream of frames.

    Not thread-safe: one session follows one stream. Any number may share a :class:`Segmenter`.

    Args:
        segmenter: The loaded :class:`..predictor.Segmenter`.
        num_frames: The video's length, if known. Only the gate passes it: it changes how pointers
            from a clip shorter than 16 frames are encoded, as upstream's offline run does.
        trace: ``trace(stage, frame, value)``, for the gate.
        spec: The concept tracker's thresholds.
    """

    def __init__(self, segmenter, num_frames: int | None = None, trace=None,
                 spec: VideoSpec = VIDEO):
        if segmenter.click is None or segmenter.tracker_net is None:
            raise RuntimeError("video tracking needs the torch image encoder and the tracker")
        self.segmenter = segmenter
        self.num_frames = num_frames
        self.trace = trace or (lambda *a: None)
        self.spec = spec
        self.frame = -1
        self.shape: tuple[int, int] | None = None
        self.mode: str | None = None
        self.texts: list[str] = []
        self.sessions: list[ConceptSession] = []
        self.group: ObjectGroup | None = None
        self.click_count = 0
        self._features = _Features()
        self._ids: dict[tuple[int, int], int] = {}
        self._texts: dict[str, dict] = {}

    # --- the frame ---------------------------------------------------------------------------

    @torch.inference_mode()
    def update(self, image: np.ndarray, texts: list[str] | None = None,
               prompts: list[dict] | None = None) -> Frame:
        """Track onto the next frame.

        Args:
            image: ``HxWx3`` RGB ``uint8``.
            texts: Phrases, on the first call of a text session (repeating them later is allowed).
            prompts: One dict per new object, ``points`` ``(P, 2)`` + ``labels`` ``(P,)``, ``box``
                ``(4,)``, or both, in the frame's pixels -- on the first call of a click session.
        """
        shape = image.shape[:2]
        if self.shape is None:
            if not texts and not prompts:
                raise ValueError("the first frame needs a prompt: there is nothing to track yet")
            if texts and prompts:
                raise ValueError("a session tracks phrases or clicks, not both")
            self.shape = shape
            self.mode = "text" if texts else "clicks"
        elif shape != self.shape:
            raise ValueError(f"every frame must be the size of the first, {self.shape[1]}x"
                             f"{self.shape[0]}; this one is {shape[1]}x{shape[0]}")
        elif texts and self.mode == "text" and list(texts) != self.texts:
            raise ValueError("the phrases are fixed for a session; start a new tracker to track "
                             "something else")
        elif texts and self.mode == "clicks":
            raise ValueError("a session tracks phrases or clicks, not both")
        elif prompts:
            raise ValueError("clicks are taken on the first frame only: SAM 3's tracker batches "
                             "the objects it is given and takes no new ones once tracking starts")
        self.frame += 1
        frame = self.frame
        seg = self.segmenter
        if self.mode == "text":
            pixels = load_frame(image).to(seg.device)
            encoded = seg.vision(pixels, stacks=("concept", "click"))
            self._pixels = pixels
            return self._text_frame(frame, encoded, texts)
        pixels = load_click_frame(image).to(seg.device)
        return self._click_frame(frame, seg.vision(pixels, stacks=("click",)), prompts)

    # --- text --------------------------------------------------------------------------------

    def _text_frame(self, frame: int, encoded: dict, texts) -> Frame:
        seg = self.segmenter
        decoder = seg.click.mask_decoder
        fine, middle, coarse = (self._round(x) for x in encoded["click"])
        levels = [decoder.conv_s0(fine), decoder.conv_s1(middle), coarse]
        self._features.set(frame, levels, encoded["positions"])
        if frame == 0:
            self.texts = list(texts)
            self.sessions = [ConceptSession(seg.tracker_net, self._features, self.shape, self.spec,
                                            self.num_frames,
                                            lambda s, f, v, i=i: self.trace(f"{i}.{s}", f, v))
                             for i in range(len(self.texts))]
        outputs = []
        for index, text in enumerate(self.texts):
            masks, scores = self._detect(encoded, text)
            session = self.sessions[index]
            for _ in range(2 if frame == 0 else 1):
                self.trace(f"{index}.pixels", frame, self._pixels)
                self.trace(f"{index}.features", frame, levels)
                self.trace(f"{index}.detections", frame, (masks, scores))
            if frame == 0:
                first = session.step(frame, masks, scores)
                self.trace(f"{index}.output", frame, (first.obj_ids, first.masks, first.scores))
            out = session.step(frame, masks, scores)
            self.trace(f"{index}.output", frame, (out.obj_ids, out.masks, out.scores))
            outputs.append(out)

        ids, prompts, masks, scores = [], [], [], []
        for index, out in enumerate(outputs):
            for row, obj_id in enumerate(out.obj_ids):
                key = (index, obj_id)
                if key not in self._ids:
                    self._ids[key] = len(self._ids)
                ids.append(self._ids[key])
                prompts.append(index)
                masks.append(out.masks[row])
                scores.append(out.scores[row])
        return self._frame(ids, prompts, masks, scores)

    @staticmethod
    def _round(x: torch.Tensor) -> torch.Tensor:
        """Meta's bfloat16 cast of the tracker's features, handed on as float32."""
        return x.to(torch.bfloat16).float()

    def _detect(self, encoded: dict, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """The concept head on one frame, scored and thinned as Meta's video detector does it."""
        seg = self.segmenter
        prompt = self._text(text)
        result = seg.concept(encoded["concept"], encoded["positions"], prompt["features"],
                             prompt["mask"], None, None)
        # Meta's video model folds the presence score into each query's logit (clamped), and the
        # detections are then thresholded on the sigmoid of that.
        logits = _inverse_sigmoid(result["logits"][0].sigmoid() * result["presence"][0].sigmoid())
        logits = logits.clamp(min=-10.0, max=10.0)
        masks = result["masks"][0]
        keep = nms(logits.sigmoid(), masks, self.spec.detection_threshold, self.spec.nms_iou)
        logits = logits - 1e4 * (~keep).float()
        probs = logits.sigmoid()
        chosen = torch.where(probs > self.spec.detection_threshold)[0]
        return masks[chosen], probs[chosen]

    def _text(self, text: str) -> dict:
        """The phrase encoded as Meta's video model encodes it: batched with the word "visual".

        Meta's video detector always encodes two prompts together -- the phrase, and the
        placeholder it uses for box exemplars -- and reads the first. A batch of two is not the
        same float computation as a batch of one, so the phrase is encoded the same way here.
        """
        if text not in self._texts:
            seg = self.segmenter
            out = seg.text(seg.tokenizer([text, "visual"]).to(seg.device))
            self._texts[text] = {"features": out["features"][:, :1], "mask": out["mask"][:1]}
        return self._texts[text]

    # --- clicks ------------------------------------------------------------------------------

    def _click_frame(self, frame: int, encoded: dict, prompts) -> Frame:
        seg = self.segmenter
        decoder = seg.click.mask_decoder
        fine, middle, coarse = encoded["click"]
        levels = [decoder.conv_s0(fine), decoder.conv_s1(middle), coarse]
        self._features.set(frame, levels, encoded["positions"])
        if frame == 0:
            self.group = ObjectGroup(seg.tracker_net, self._features, self.shape, self.num_frames)
            for prompt in prompts:
                points, labels = self._click_inputs(prompt)
                self.group.add_points(frame, self.click_count, points, labels)
                self.click_count += 1
            self.group.preflight(run_mem_encoder=True)
        low, scores = self.group.propagate(frame, run_mem_encoder=True)
        self.group.prune(frame)
        video = torch.nn.functional.interpolate(low, size=self.shape, mode="bilinear",
                                                align_corners=False)
        self.trace("clicks.output", frame, (list(self.group.obj_ids), low, video, scores))
        scores = scores[:, 0]
        visible = (scores > 0).nonzero().flatten().tolist()
        return self._frame(visible, visible, [video[i, 0] > 0 for i in visible],
                           [scores[i].sigmoid() for i in visible])

    def _click_inputs(self, prompt: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Box corners first (labels 2 and 3), then clicks, in the encoder's square."""
        points = prompt.get("points")
        labels = prompt.get("labels")
        box = prompt.get("box")
        coords = [] if points is None else [np.asarray(points, dtype=np.float32).reshape(-1, 2)]
        marks = [] if labels is None else [np.asarray(labels, dtype=np.int32).reshape(-1)]
        if box is not None:
            coords.insert(0, np.asarray(box, dtype=np.float32).reshape(2, 2))
            marks.insert(0, np.array([2, 3], dtype=np.int32))
        coords = to_model_coords(np.concatenate(coords), self.shape)
        marks = torch.from_numpy(np.concatenate(marks))
        device = self.segmenter.device
        return coords[None].to(device), marks[None].to(device)

    # --- results -----------------------------------------------------------------------------

    def _frame(self, ids, prompts, masks, scores) -> Frame:
        height, width = self.shape
        if not ids:
            return Frame([], [], torch.zeros(0, height, width, dtype=torch.bool), torch.zeros(0))
        order = sorted(range(len(ids)), key=lambda i: ids[i])
        return Frame([ids[i] for i in order], [prompts[i] for i in order],
                     torch.stack([masks[i] for i in order]).cpu(),
                     torch.stack([torch.as_tensor(scores[i]) for i in order]).float().cpu())

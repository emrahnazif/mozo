#!/usr/bin/env python3
"""Check that mozo's SAM 3 tracker follows concepts and clicks exactly as Meta's video model does.

Two paths reach the same checkpoint. Meta's own SAM 3 video model (``facebookresearch/sam3`` at
the commit ``PROVENANCE.md`` pins, ``build_sam3_video_model()`` with its defaults: temporal
disambiguation on), driven the way its example notebooks drive it; and mozo's
``Segmenter.tracker()``, one frame at a time. Meta's code is run, never copied -- it ships under
the SAM License -- exactly as ``tools/verify/sam3_reference.py`` runs the image model.

The comparison is exact (``torch.equal``; set equality on ids), per frame, at every stage both
paths expose:

- the frame as the detector reads it, and the tracker's image features;
- the detections kept (mask logits and scores, after NMS);
- the tracks propagated onto the frame (low-resolution logits after hole filling, object scores);
- the update plan: new detections and their ids, unmatched tracks, matches, re-seeds, removals;
- the masks built for the frame (video resolution, per object);
- the frame's output: ids shown, masks after the object-wise overlap rule, per-frame scores.

Meta's output is read **before its hotstart buffer**: each frame post-processed with what was known
at that frame (``PROVENANCE.md``, "Hotstart"). Every clip here is at least 16 frames long unless a
case says otherwise, so the length of the clip -- which Meta's offline run reads -- does not
enter, and the vendor is told it anyway.

**Accommodations of the reference** (none changes a value but the second; ``PROVENANCE.md``):

1. Meta's process-wide bfloat16 autocast is exited.
2. Meta's bfloat16 cast of the tracker's features "for all-gather" is kept, and handed on as float32.
3. Hardcoded ``.cuda()`` / ``device="cuda"``, ``pin_memory``, and the tracker's state storage stay
   on the CPU (the image gate's own ``install`` plus three more).
4. Meta's CPU connected-components fails on an empty batch (a frame with no objects); it returns the
   empty result its CUDA path returns.

**Cases**, each reaching something the others do not:

- *people*: "person" on Meta's ``0001`` clip -- four objects, the re-seed on frame 16;
- *hands*: "hand" on ``0001`` -- objects arriving mid-clip in several batches, occlusions;
- *nothing*: "toy" on ``bedroom`` -- a concept that is never found;
- *two prompts*: "person" and "hand" in one mozo session, each against Meta run with that phrase alone;
- *clicks*: a click, a box and a click pair on ``bedroom`` frame 0, against Meta's tracker without
  the detector (``examples/sam3_for_sam2_video_task_example.ipynb``);
- *short*: "person" on 10 frames, the clip's length given (it changes pointer encodings);
- *end to end*: "person" through ``mozo.get_model`` -> ``tracker()`` -> PixelFlow, length not given.

Run from the repository root (CPU; 12 threads take about an hour)::

    git clone https://github.com/facebookresearch/sam3 ~/src/sam3
    git -C ~/src/sam3 checkout 5eb25fb
    MKL_CBWR=AVX2,STRICT python tools/verify/sam3_video.py --reference ~/src/sam3
    MKL_CBWR=AVX2,STRICT python tools/verify/sam3_video.py --reference ~/src/sam3 --falsify

``--falsify`` runs the vendored side once per deliberate perturbation against the same reference
runs, and passes only if every one is caught.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))

from mozo.vendors.sam3_deploy import Segmenter  # noqa: E402
from mozo.vendors.sam3_deploy.tracker import concept as vendor_concept  # noqa: E402
from mozo.vendors.sam3_deploy.tracker import group as vendor_group  # noqa: E402
from mozo.vendors.sam3_deploy.tracker import network as vendor_network  # noqa: E402
from mozo.vendors.sam3_deploy.tracker import ops as vendor_ops  # noqa: E402
from mozo.vendors.sam3_deploy.tracker import session as vendor_session  # noqa: E402

#: The commit of ``facebookresearch/sam3`` the reference must be at.
PINNED = "5eb25fb"

#: Click prompts on the bedroom clip (960x540), as Meta's own notebook places them: a single click
#: (three candidates), a box (two points, the stability fallback), a click pair with a negative.
CLICKS = [
    {"points": [[210, 350]], "labels": [1]},
    {"box": [300, 0, 500, 400]},
    {"points": [[400, 150], [275, 175]], "labels": [1, 0]},
]


def published() -> Path:
    """The checkpoint mozo serves, located the way mozo itself would."""
    from mozo.weights import resolve

    return Path(resolve("sam3", "sam3", "torch-fp32"))


# --- clips ---------------------------------------------------------------------------------

def clip(reference: Path, name: str, count: int) -> list[np.ndarray]:
    """The first ``count`` frames of one of Meta's example clips, decoded once."""
    if name == "0001":
        folder = reference / "assets" / "videos" / "0001"
        return [np.array(Image.open(folder / f"{i}.jpg").convert("RGB")) for i in range(count)]
    import cv2

    capture = cv2.VideoCapture(str(reference / "assets" / "videos" / f"{name}.mp4"))
    frames = []
    while len(frames) < count:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if len(frames) < count:
        raise SystemExit(f"{name} has {len(frames)} frames; asked for {count}")
    return frames


def as_jpeg(frames: list[np.ndarray]) -> tuple[list[bytes], list[np.ndarray]]:
    """Encode frames as JPEG once, and decode them back: what both sides of the click case read.

    Meta's tracker-only loader reads a folder of JPEG files and nothing else; the arrays mozo is
    given are those same files, decoded, so both see identical pixels.
    """
    import io

    encoded, decoded = [], []
    for frame in frames:
        buffer = io.BytesIO()
        Image.fromarray(frame).save(buffer, format="JPEG", quality=95)
        encoded.append(buffer.getvalue())
        decoded.append(np.array(Image.open(io.BytesIO(encoded[-1])).convert("RGB")))
    return encoded, decoded


@contextlib.contextmanager
def frame_folder(frames: list[np.ndarray], jpeg: list[bytes] | None = None):
    """The frames in a folder named the way Meta's loaders expect: lossless PNGs, or given JPEGs."""
    with tempfile.TemporaryDirectory(prefix="sam3_video_") as folder:
        if jpeg is not None:
            for i, data in enumerate(jpeg):
                Path(f"{folder}/{i}.jpg").write_bytes(data)
        else:
            for i, frame in enumerate(frames):
                Image.fromarray(frame).save(f"{folder}/{i}.png")
        yield folder


# --- the reference -------------------------------------------------------------------------

def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu(x) for x in value)
    if isinstance(value, dict):
        return {k: _cpu(x) for k, x in value.items()}
    if isinstance(value, set):
        return {int(x) for x in value}
    return value


class Reference:
    """Meta's SAM 3 video model on the CPU, with the accommodations in the module text."""

    def __init__(self, reference: Path, checkpoint: Path):
        from tools.verify.sam3_reference import install

        install(reference)
        torch.Tensor.cuda = lambda self, *a, **k: self
        torch.nn.Module.cuda = lambda self, *a, **k: self

        from sam3.model.sam3_tracking_predictor import Sam3TrackerPredictor
        from sam3.model_builder import build_sam3_video_model
        import sam3.perflib.connected_components as components

        init = Sam3TrackerPredictor.init_state
        Sam3TrackerPredictor.init_state = lambda self, *a, **k: init(
            self, *a, **{**k, "offload_state_to_cpu": True})
        cpu_components = components.connected_components_cpu

        def guarded(tensor):
            if tensor.shape[0] == 0:
                empty = torch.zeros(tensor.shape, dtype=torch.int32, device=tensor.device)
                return empty, empty.clone()
            return cpu_components(tensor)

        components.connected_components_cpu = guarded

        self.model = build_sam3_video_model(
            checkpoint_path=str(checkpoint), load_from_HF=False,
            bpe_path=str(reference / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"),
            apply_temporal_disambiguation=True, device="cpu").eval()
        context = getattr(self.model.tracker, "bf16_context", None)
        if context is not None:
            context.__exit__(None, None, None)
            self.model.tracker.bf16_context = None
        detector = type(self.model.detector)
        gather = detector._gather_tensor

        def gather_fp32(self, x):
            got, handle = gather(self, x)
            if self.world_size == 1 and x.dtype == torch.bfloat16:
                got = [g.float() for g in got]
            return got, handle

        detector._gather_tensor = gather_fp32

    @torch.inference_mode()
    def concept(self, frames: list[np.ndarray], text: str) -> list:
        """Run one phrase over the frames; return the stage trace."""
        model = self.model
        trace = []
        record = lambda stage, frame, value: trace.append((stage, frame, _cpu(value)))  # noqa
        saved = {}

        def wrap(name, fn):
            saved[name] = getattr(model, name)
            setattr(model, name, lambda *a, **k: fn(saved[name], *a, **k))

        def detection(original, *a, **k):
            out = original(*a, **k)
            frame = k["frame_idx"]
            record("pixels", frame, k["input_batch"].img_batch[frame].float().unsqueeze(0))
            cache = k["feature_cache"][frame][1]["tracker_backbone_out"]
            record("features", frame, list(cache["backbone_fpn"]))
            record("detections", frame, (out["mask"], out["scores"]))
            return out

        def propagation(original, *a, **k):
            out = original(*a, **k)
            record("propagate", k["frame_idx"], out)
            return out

        def planning(original, *a, **k):
            plan, meta = original(*a, **k)
            record("plan", k["frame_idx"], (
                plan["new_det_fa_inds"], plan["new_det_obj_ids"], plan["unmatched_trk_obj_ids"],
                plan["det_to_matched_trk_obj_ids"], plan["trk_id_to_max_iou_high_conf_det"],
                plan["obj_ids_newly_removed"]))
            return plan, meta

        def outputs(original, *a, **k):
            out = original(*a, **k)
            record("masks", k["frame_idx"], out)
            return out

        def single(original, inference_state, frame_idx, reverse):
            out = original(inference_state, frame_idx, reverse)
            shown = model._postprocess_output(
                inference_state, out, removed_obj_ids=set(out.get("removed_obj_ids", set())),
                suppressed_obj_ids=set(out.get("suppressed_obj_ids", set())))
            scores = out["obj_id_to_tracker_score"]
            ids = [int(i) for i in shown["out_obj_ids"]]
            record("output", frame_idx, (ids, torch.from_numpy(shown["out_binary_masks"]),
                                         [float(scores.get(i, 0.0)) for i in ids]))
            return out

        wrap("run_backbone_and_detection", detection)
        wrap("run_tracker_propagation", propagation)
        wrap("run_tracker_update_planning_phase", planning)
        wrap("build_outputs", outputs)
        wrap("_run_single_frame_inference", single)
        try:
            with frame_folder(frames) as folder:
                state = model.init_state(resource_path=folder, offload_video_to_cpu=True)
                model.add_prompt(state, frame_idx=0, text_str=text)
                for _ in model.propagate_in_video(state, start_frame_idx=0,
                                                  max_frame_num_to_track=len(frames)):
                    pass
        finally:
            for name, original in saved.items():
                setattr(model, name, original)
        return trace

    @torch.inference_mode()
    def clicks(self, frames: list[np.ndarray], jpeg: list[bytes], prompts: list[dict]) -> list:
        """Run click prompts on frame 0 through Meta's tracker without the detector."""
        tracker = self.model.tracker
        tracker.backbone = self.model.detector.backbone
        height, width = frames[0].shape[:2]
        trace = []
        try:
            with frame_folder(frames, jpeg) as folder:
                state = tracker.init_state(video_path=folder, offload_video_to_cpu=True,
                                           offload_state_to_cpu=True)
                for obj, prompt in enumerate(prompts):
                    points = labels = box = None
                    if prompt.get("points") is not None:
                        points = torch.tensor(prompt["points"], dtype=torch.float32)
                        points[:, 0] /= width
                        points[:, 1] /= height
                        labels = torch.tensor(prompt["labels"], dtype=torch.int32)
                    if prompt.get("box") is not None:
                        box = torch.tensor(prompt["box"], dtype=torch.float32).reshape(2, 2)
                        box[:, 0] /= width
                        box[:, 1] /= height
                        box = box.reshape(-1)
                    tracker.add_new_points_or_box(state, frame_idx=0, obj_id=obj, points=points,
                                                  labels=labels, box=box)
                for frame, obj_ids, low, video, scores in tracker.propagate_in_video(
                        state, start_frame_idx=0, max_frame_num_to_track=len(frames),
                        reverse=False, propagate_preflight=True):
                    trace.append(("clicks", frame, _cpu((list(obj_ids), low, video, scores))))
        finally:
            tracker.backbone = None
        return trace


# --- mozo ----------------------------------------------------------------------------------

def run_vendor(segmenter: Segmenter, frames, texts=None, prompts=None,
               num_frames: int | None = None) -> list:
    """Run mozo's session over the frames; return its trace (stage names as the reference's)."""
    trace = []

    def record(stage, frame, value):
        trace.append((stage, frame, _cpu(value)))

    session = segmenter.tracker(num_frames=num_frames, trace=record)
    for i, frame in enumerate(frames):
        if i == 0:
            session.update(frame, texts=texts, prompts=prompts)
        else:
            session.update(frame)
    return trace


def per_prompt(trace: list, index: int) -> list:
    """One prompt's stages from a vendor trace, named as a single-prompt reference names them."""
    return [(stage.split(".", 1)[1], frame, value) for stage, frame, value in trace
            if stage.startswith(f"{index}.")]


# --- comparison ----------------------------------------------------------------------------

class Caught(Exception):
    """Raised by a :class:`Comparison` that stops at its first failure (falsification)."""


def _equal(a, b) -> str | None:
    """None if ``a`` and ``b`` are exactly equal, else a short description of the difference."""
    if isinstance(a, torch.Tensor) and isinstance(b, (list, tuple)):
        b = torch.tensor(b, dtype=a.dtype if a.numel() else torch.float32)
    if isinstance(b, torch.Tensor) and isinstance(a, (list, tuple)):
        a = torch.tensor(a, dtype=b.dtype if b.numel() else torch.float32)
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape:
            return f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
        if a.dtype == torch.bool or b.dtype == torch.bool:
            n = int((a.bool() != b.bool()).sum())
            return None if n == 0 else f"{n} pixels differ"
        if a.dtype != b.dtype:
            return f"dtype {a.dtype} vs {b.dtype}"
        if torch.equal(a, b):
            return None
        return f"max|d| {(a.float() - b.float()).abs().max().item():.3g}"
    if isinstance(a, dict) and isinstance(b, dict):
        if sorted(int(k) for k in a) != sorted(int(k) for k in b):
            return f"keys {sorted(int(k) for k in a)} vs {sorted(int(k) for k in b)}"
        a2 = {int(k): v for k, v in a.items()}
        b2 = {int(k): v for k, v in b.items()}
        for k in a2:
            d = _equal(a2[k], b2[k])
            if d:
                return f"[{k}] {d}"
        return None
    if isinstance(a, set) or isinstance(b, set):
        sa, sb = {int(x) for x in a}, {int(x) for x in b}
        return None if sa == sb else f"{sorted(sa)} vs {sorted(sb)}"
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return None if a.shape == b.shape and np.array_equal(a, b) else f"{a.tolist()} vs {b.tolist()}"
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return f"length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = _equal(x, y)
            if d:
                return f".{i} {d}"
        return None
    return None if a == b else f"{a} vs {b}"


class Comparison:
    """Collects exact comparisons and reports each failure."""

    def __init__(self, quiet: bool = False, stop: bool = False):
        self.quiet, self.stop = quiet, stop
        self.checked = 0
        self.failed: list[str] = []

    def traces(self, label: str, got: list, want: list) -> None:
        """Compare two traces stage by stage, call by call."""
        grouped_got, grouped_want = defaultdict(list), defaultdict(list)
        for stage, frame, value in got:
            grouped_got[stage].append((frame, value))
        for stage, frame, value in want:
            grouped_want[stage].append((frame, value))
        if set(grouped_got) != set(grouped_want):
            self._fail(f"{label}: stages {sorted(grouped_got)} vs {sorted(grouped_want)}")
            return
        # In frame order, so the first failure reported is the earliest.
        calls = []
        for stage in grouped_want:
            g, w = grouped_got[stage], grouped_want[stage]
            if len(g) != len(w):
                self._fail(f"{label} {stage}: {len(g)} calls vs {len(w)}")
                continue
            calls += [(w[i][0], i, stage, g[i], w[i]) for i in range(len(w))]
        for frame, i, stage, (fg, vg), (fw, vw) in sorted(calls, key=lambda c: (c[0], c[1])):
            self.checked += 1
            d = f"frame {fg} vs {fw}" if fg != fw else _equal(vg, vw)
            if d:
                self._fail(f"{label} frame {fw} {stage} #{i}: {d}")

    def _fail(self, message: str) -> None:
        self.failed.append(message)
        if self.stop:
            raise Caught(message)
        if not self.quiet:
            print(f"  FAIL {message}")


# --- cases ---------------------------------------------------------------------------------

class Cases:
    """The reference runs, computed once and reused by the gate and by every falsifier."""

    def __init__(self, reference: Path, checkpoint: Path, count: int):
        self.reference_path = reference
        self.checkpoint = checkpoint
        self.count = count
        self.people = clip(reference, "0001", count)
        self.bedroom = clip(reference, "bedroom", count)
        self.bedroom_jpeg, self.bedroom_decoded = as_jpeg(self.bedroom)
        self._reference = None
        self._runs: dict[str, list] = {}

    def want(self, key: str) -> list:
        if key not in self._runs:
            if self._reference is None:
                self._reference = Reference(self.reference_path, self.checkpoint)
            ref = self._reference
            print(f"  reference: {key}", flush=True)
            if key == "people":
                self._runs[key] = ref.concept(self.people, "person")
            elif key == "hands":
                self._runs[key] = ref.concept(self.people, "hand")
            elif key == "nothing":
                self._runs[key] = ref.concept(self.bedroom, "toy")
            elif key == "clicks":
                self._runs[key] = ref.clicks(self.bedroom_decoded, self.bedroom_jpeg, CLICKS)
            elif key == "short":
                self._runs[key] = ref.concept(self.people[:10], "person")
        return self._runs[key]


def check(cases: Cases, segmenter: Segmenter, same: Comparison, end_to_end: bool = True) -> None:
    n = cases.count
    print("people: 'person' on 0001", flush=True)
    got = run_vendor(segmenter, cases.people, texts=["person"], num_frames=n)
    same.traces("people", per_prompt(got, 0), cases.want("people"))
    print("hands: 'hand' on 0001", flush=True)
    got = run_vendor(segmenter, cases.people, texts=["hand"], num_frames=n)
    same.traces("hands", per_prompt(got, 0), cases.want("hands"))
    print("nothing: 'toy' on bedroom", flush=True)
    got = run_vendor(segmenter, cases.bedroom, texts=["toy"], num_frames=n)
    same.traces("nothing", per_prompt(got, 0), cases.want("nothing"))
    print("two prompts: 'person' + 'hand' in one session", flush=True)
    got = run_vendor(segmenter, cases.people, texts=["person", "hand"], num_frames=n)
    same.traces("two prompts/person", per_prompt(got, 0), cases.want("people"))
    same.traces("two prompts/hand", per_prompt(got, 1), cases.want("hands"))
    print("clicks: click, box, click pair on bedroom", flush=True)
    got = run_vendor(segmenter, cases.bedroom_decoded, prompts=CLICKS, num_frames=n)
    same.traces("clicks", [(s.split(".", 1)[0], f, v) for s, f, v in got if s == "clicks.output"],
                cases.want("clicks"))
    print("short: 'person' on 10 frames, length given", flush=True)
    got = run_vendor(segmenter, cases.people[:10], texts=["person"], num_frames=10)
    same.traces("short", per_prompt(got, 0), cases.want("short"))
    if end_to_end:
        check_end_to_end(cases, same)


def check_end_to_end(cases: Cases, same: Comparison) -> None:
    """``mozo.get_model`` -> ``tracker()`` -> PixelFlow, length not given, against the reference."""
    import mozo

    print("end to end: mozo.get_model('sam3').tracker(), 'person'", flush=True)
    model = mozo.get_model("sam3/sam3", device="cpu")
    tracker = model.tracker()
    want = [v for s, f, v in cases.want("people") if s == "output"]
    # Meta's API visits frame 0 twice; the second visit is what both sides show.
    want = want[1:]
    for i, frame in enumerate(cases.people):
        found = list(tracker.update(frame, text="person"))
        ids, masks, _ = want[i]
        same.checked += 1
        got_ids = [row.tracker_id for row in found]
        got_masks = (torch.from_numpy(np.stack([row.masks[0] for row in found])) if found
                     else torch.zeros(0, *frame.shape[:2], dtype=torch.bool))
        d = (f"ids {got_ids} vs {ids}" if got_ids != ids else _equal(got_masks, masks))
        if d:
            same._fail(f"end to end frame {i}: {d}")


# --- falsification -------------------------------------------------------------------------

FALSIFIERS = {
    "fp32-frames": "frames stored in float32, not Meta's float16 video loader",
    "no-bf16-features": "the tracker's features not rounded through bfloat16",
    "text-alone": "the phrase encoded alone, not batched with 'visual' as Meta's detector does",
    "no-nms": "detections not thinned by mask NMS",
    "no-hole-filling": "tracks not cleaned of small holes and specks",
    "float32-memory": "memories stored in float32, not bfloat16",
    "real-rotary": "memory attention's rotary encoding in real arithmetic, not complex",
    "single-first-visit": "the first frame visited once, not twice as Meta's API does",
    "reseed-period-15": "re-seeding every 15 frames instead of 16",
    "no-memory-selection": "memory selection off: every past frame a candidate",
    "one-batch": "one batch per object, not one per arrival frame (objects arriving together)",
}

_MISSING = object()


@contextlib.contextmanager
def perturbed(name: str):
    """Apply one :data:`FALSIFIERS` perturbation to the vendored side, and undo it after."""
    undo = []

    def patch(owner, attribute, value):
        undo.append((owner, attribute, owner.__dict__.get(attribute, _MISSING)))
        setattr(owner, attribute, value)

    if name == "fp32-frames":
        def load(image, spec=vendor_ops.SPEC):
            size = spec.trunk.image_size
            resized = Image.fromarray(image).resize((size, size), Image.BILINEAR)
            frame = torch.from_numpy(np.array(resized)).permute(2, 0, 1).float().div(255)
            mean = torch.tensor(spec.mean)[:, None, None]
            std = torch.tensor(spec.std)[:, None, None]
            return ((frame - mean) / std).unsqueeze(0)
        patch(vendor_session, "load_frame", load)
    elif name == "no-bf16-features":
        patch(vendor_session.VideoTracker, "_round", staticmethod(lambda x: x))
    elif name == "text-alone":
        def text(self, phrase):
            seg = self.segmenter
            return seg.encode_text(phrase)
        patch(vendor_session.VideoTracker, "_text", text)
    elif name == "no-nms":
        patch(vendor_session, "nms", lambda probs, masks, threshold, iou: probs > threshold)
    elif name == "no-hole-filling":
        patch(vendor_concept, "fill_holes", lambda logits, max_area: logits)
    elif name == "float32-memory":
        original_encode = vendor_group.ObjectGroup._encode

        def encode(self, frame, masks, scores, from_points):
            feats, _ = self.features(frame, masks.size(0))
            memory, pos = self.net.encode_memory(feats[-1], masks, scores, from_points)
            return memory, self._pos_enc(pos)
        patch(vendor_group.ObjectGroup, "_encode", encode)
        del original_encode
    elif name == "real-rotary":
        from mozo.vendors.sam3_deploy.tracker import memory as vendor_memory

        def rotate(x, table):
            cos = table.real.repeat_interleave(2, -1).view(1, 1, *x.shape[-2:])
            sin = table.imag.repeat_interleave(2, -1).view(1, 1, *x.shape[-2:])
            pairs = x.float().view(*x.shape[:-1], -1, 2)
            turned = torch.stack((-pairs[..., 1], pairs[..., 0]), -1).flatten(-2)
            return (x.float() * cos + turned * sin).type_as(x)
        patch(vendor_memory, "_rotate", rotate)
    elif name == "single-first-visit":
        original_step = vendor_concept.ConceptSession.step

        def step(self, frame, masks, scores):
            if frame == 0 and getattr(self, "_visited", False):
                return self._first
            self._visited = True
            out = original_step(self, frame, masks, scores)
            self._first = out
            return out
        patch(vendor_concept.ConceptSession, "step", step)
    elif name == "reseed-period-15":
        original_init = vendor_concept.ConceptSession.__init__

        def init(self, *a, **k):
            original_init(self, *a, **k)
            from dataclasses import replace
            self.spec = replace(self.spec, recondition_every=15)
        patch(vendor_concept.ConceptSession, "__init__", init)
    elif name == "no-memory-selection":
        def select(self, outputs, frame, reach):
            return [t for t in range(max(0, frame - reach + 1), frame) if t in outputs] or \
                ([frame - 1] if frame else [])
        patch(vendor_network.TrackerNet, "select_frames", select)
    elif name == "one-batch":
        original_add = vendor_concept.ConceptSession._add_objects

        def add(self, frame, det_masks, new_inds, new_ids):
            for index, obj_id in zip(new_inds, new_ids):
                original_add(self, frame, det_masks, np.array([index]), np.array([obj_id]))
        patch(vendor_concept.ConceptSession, "_add_objects", add)
    else:
        raise SystemExit(f"unknown falsifier {name!r}; choose from {sorted(FALSIFIERS)}")
    try:
        yield
    finally:
        for owner, attribute, value in reversed(undo):
            if value is _MISSING:
                delattr(owner, attribute)
            else:
                setattr(owner, attribute, value)


def falsify(cases: Cases, segmenter: Segmenter, names: list[str] | None = None) -> int:
    """Run the vendored side under each perturbation; every one must be caught."""
    caught, missed = [], []
    for name, what in FALSIFIERS.items():
        if names and name not in names:
            continue
        same = Comparison(quiet=True, stop=True)
        with perturbed(name), contextlib.suppress(Caught):
            with contextlib.redirect_stdout(open(os.devnull, "w")):
                check(cases, segmenter, same, end_to_end=False)
        (caught if same.failed else missed).append(name)
        first = same.failed[0] if same.failed else "-"
        print(f"  {'caught' if same.failed else 'MISSED':<7} {name:<20} after {same.checked} "
              f"comparisons; first failure: {first}", flush=True)
        print(f"          {what}")
    print()
    if missed:
        print(f"{len(missed)} perturbation(s) passed the gate: {missed}")
        return 1
    print(f"all {len(caught)} perturbations caught.\nSAM 3 video falsification: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, required=True,
                        help="checkout of facebookresearch/sam3 at the pinned commit")
    parser.add_argument("--frames", type=int, default=20,
                        help="frames per clip (default 20: covers the frame-16 re-seed)")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="a checkpoint to check instead of the published one")
    parser.add_argument("--falsify", action="store_true",
                        help="run each deliberate perturbation; every one must fail")
    parser.add_argument("--falsifier", action="append", choices=sorted(FALSIFIERS),
                        help="with --falsify, run only this perturbation (repeatable)")
    parser.add_argument("--allow-commit-drift", action="store_true",
                        help="run against a reference other than the pinned commit")
    arguments = parser.parse_args()
    if arguments.frames < 16:
        raise SystemExit("--frames must be at least 16 (the short case covers fewer)")

    reference = arguments.reference.resolve()
    if not (reference / "sam3" / "model" / "sam3_video_base.py").is_file():
        raise SystemExit(f"no sam3 checkout at {reference}")
    head = subprocess.run(["git", "-C", str(reference), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    if not head.startswith(PINNED) and not arguments.allow_commit_drift:
        raise SystemExit(f"the reference is at {head[:12] or 'an unknown commit'}; "
                         f"PROVENANCE.md pins {PINNED}. Check it out, or pass --allow-commit-drift.")

    torch.set_grad_enabled(False)
    checkpoint = (arguments.checkpoint or published()).resolve()
    print(f"reference  {reference} @ {head[:12]}")
    print(f"weights    {checkpoint}")
    print(f"versions   torch {torch.__version__}, numpy {np.__version__}; device cpu, "
          f"{torch.get_num_threads()} threads, MKL_CBWR={os.environ.get('MKL_CBWR')}")
    cases = Cases(reference, checkpoint, arguments.frames)
    segmenter = Segmenter(checkpoint, device="cpu")

    if arguments.falsify:
        return falsify(cases, segmenter, arguments.falsifier)

    same = Comparison()
    check(cases, segmenter, same)
    print()
    if same.failed:
        print(f"{len(same.failed)} of {same.checked} comparisons differ.\nSAM 3 video gate: FAIL")
        return 1
    print(f"all {same.checked} comparisons bit-identical.\nSAM 3 video gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

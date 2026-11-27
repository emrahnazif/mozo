#!/usr/bin/env python3
"""Check that mozo's SAM 2 tracker follows objects exactly as Meta's video predictor does.

Three paths reach the same weights. ``facebookresearch/sam2`` itself, driven through its own
``SAM2VideoPredictor`` (``build_sam2_video_predictor`` with its defaults, fed a folder of frames);
``mozo/vendors/sam2_deploy`` -- ``Segmenter.tracker()``, one frame at a time; and mozo end to end
-- registry, weights, ``Sam2Predictor.tracker()``, PixelFlow. Between upstream and the vendor sit
a whole-video predictor rewritten as a bounded session, ``SAM2Base`` cut down to its tracking
step, and a frame loader rewritten around arrays -- and any of those could quietly change a number.

The comparison is exact (``torch.equal``), per frame and per object, at every stage: the
preprocessed frame, the image features, the memory-conditioned features (memory attention's
output), the decoder's low-res logits, the object pointer, the object score, the bfloat16 mask
memory, and the logits at the video's own size.

**Scenarios**, each reaching something the others do not:

- *three objects*: a click, a box, and a positive + negative click pair on frame 0, tracked
  through every frame; compared against upstream's offline ``propagate_in_video``.
- *added later*: an object on frame 0 and another started on a later frame; compared against
  upstream's own predictor given the second object's prompt part way through, driven frame by
  frame (its current predictor allows adding objects after tracking starts).
- *short clip*: fewer than 16 frames, where upstream's answer depends on the clip's length; the
  vendor is told the length (mozo never is: a stream has none).
- *reversed*: frames fed backwards from a prompted middle frame, against upstream's
  ``reverse=True``.
- *image*: ``Segmenter.predict`` against ``SAM2ImagePredictor``, including single-mask calls, which
  reach the decoder's stability fallback this package now configures.
- *end to end*: ``mozo.get_model`` -> ``tracker()`` -> PixelFlow rows, against upstream's masks.

**On CUDA (the primary gate) and on the CPU.** Both sides run in one process, on one device, in
fp32 -- mozo's only torch precision for SAM 2 (``torch-fp32`` is the only runtime it publishes and
nothing derives a half-precision one, so ``auto`` is ``torch-fp32`` on CUDA too). The global
switches stay as mozo's runtime leaves them, which is torch's defaults (TF32 off for matmuls, on
for cuDNN convolutions, ``cudnn.benchmark`` off, SDPA free to pick its kernel); they are printed
with the stack and the GPU. Each side records everything ``--repeats`` times (default twice) and
must repeat itself byte for byte before the two are compared.

**The reference is not patched.** Upstream at this commit has no attention-backend pin, and the
CUDA extension it uses to fill holes is not built (the checkout is imported from source, as on
every install without ``nvcc``) -- so it skips that step, which is what the vendor reproduces
(PROVENANCE).

Run from the repository root, in an environment with torch, Pillow, hydra-core and iopath::

    git clone https://github.com/facebookresearch/sam2 ~/src/sam2
    git -C ~/src/sam2 checkout <commit in PROVENANCE.md>
    python tools/verify/sam2_video.py --upstream ~/src/sam2 --device cuda tiny small base_plus large
    python tools/verify/sam2_video.py --upstream ~/src/sam2 --device cuda --falsify \\
        --reference-cache /tmp/sam2_video-ref
    MKL_CBWR=AVX2,STRICT python tools/verify/sam2_video.py --upstream ~/src/sam2      # CPU, tiny
    MKL_CBWR=AVX2,STRICT python tools/verify/sam2_video.py --upstream ~/src/sam2 --falsify

``--falsify`` runs the gate once per deliberate perturbation of the vendored side and passes only
if every one is caught. The reference does not change between perturbations, so it is recorded
once (as digests, under ``--reference-cache`` when given, and reused by later runs on the same
stack). The clip is ``notebooks/videos/bedroom`` from the checkout itself.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import NamedTuple

import numpy as np
import PIL
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
# This folder would otherwise shadow nothing today, but a ``sam2.py`` here one day would shadow
# the reference package; keep the reference import unambiguous.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))

from mozo.vendors.sam2_deploy import Segmenter  # noqa: E402
from mozo.vendors.sam2_deploy import network as vendor_network  # noqa: E402
from mozo.vendors.sam2_deploy import video as vendor_video  # noqa: E402
from mozo.vendors.sam2_deploy.predictor import LOGIT_LIMIT  # noqa: E402

FAMILY = "sam2"
PROVENANCE = ROOT / "mozo" / "vendors" / "sam2_deploy" / "PROVENANCE.md"
CONFIGS = {"tiny": "t", "small": "s", "base_plus": "b+", "large": "l"}
#: Bumped whenever what a side records changes, so a stale ``--reference-cache`` is not reused.
RECORD_VERSION = 2

#: Prompts in the bedroom clip's own pixels (960x540), as upstream's ``add_new_points_or_box``
#: takes them. A click alone reaches the three-candidate choice on the prompted frame; a box is
#: two points and reaches the single-mask stability fallback; a click pair has a negative.
CLICK = {"points": [[210, 350]], "labels": [1], "box": None}
BOX = {"points": None, "labels": None, "box": [300, 0, 500, 400]}
PAIR = {"points": [[460, 60], [510, 220]], "labels": [1, 0], "box": None}
SECOND_BOX = {"points": None, "labels": None, "box": [560, 180, 760, 470]}

#: Deliberate perturbations of the vendored side, each of which must make the gate fail: what it
#: does and the stage it should first fail at, and ``"cuda"`` for those only a GPU run can catch --
#: a global switch, a memory layout or a host/device split that changes kernels on CUDA and
#: changes nothing on a CPU.
FALSIFIERS = {
    "image-path-resize": ("frames resized like the image predictor (antialiased bilinear) instead "
                          "of the video loader's PIL bicubic (fails at: pixels)", None),
    "no-bf16-memory": ("keep mask memories in float32 (fails at: frame 0's stored memory)", None),
    "pointer-window": ("object-pointer temporal encoding normalised by 16 instead of 15 "
                       "(fails at: memory attention, frame 1)", None),
    "short-reach": ("forget frames older than 6 instead of 15 (fails at: memory attention, "
                    "frame 8)", None),
    "memory-from-high-res": ("store the prompted frame's logits round-tripped through the "
                             "decoder's high-res masks (fails at: frame 0 logits)", None),
    "no-stability-fallback": ("decoder without the dynamic multimask fallback (fails at: frame 0 "
                              "logits)", None),
    "frozen": ("every parameter frozen (requires_grad False); upstream's builders leave them "
               "trainable, and torch.matmul folds a batched product by that flag (fails at: the "
               "requires-grad stage, recorded last; on an RTX 3060 no arithmetic moved)", None),
    "tf32-matmul": ("TF32 matmuls on for the vendor only; mozo leaves torch's default, off "
                    "(fails at: image features, frame 0)", "cuda"),
    "cudnn-tf32-off": ("TF32 cuDNN convolutions off for the vendor only; mozo leaves torch's "
                       "default, on (fails at: image features, frame 0)", "cuda"),
    "channels-last": ("frames in channels-last memory, same values: cuDNN picks other kernels "
                      "(fails at: image features, frame 0)", "cuda"),
    "sdpa-math": ("SDPA pinned to the math kernel for the vendor only; upstream lets torch pick "
                  "(fails at: image features, frame 0)", "cuda"),
    "memory-attention-host": ("memory attention run on the host, the rest on the GPU "
                              "(fails at: memory attention call 0)", "cuda"),
    "prompt-coords-host": ("image-path prompt scaled on the host; upstream scales on the device, "
                           "where a divide by a Python number is a multiply by its reciprocal "
                           "(fails at: image, unstable click, IoU)", "cuda"),
}


def pinned_commit() -> str:
    """The upstream commit ``PROVENANCE.md`` records this package as extracted from."""
    match = re.search(r"Upstream commit\s*\|\s*`([0-9a-f]{40})`", PROVENANCE.read_text())
    if match is None:
        raise SystemExit(f"no upstream commit recorded in {PROVENANCE}")
    return match.group(1)


def published(variant: str) -> Path:
    """Locate a variant's checkpoint the way mozo itself would."""
    from mozo.weights import resolve

    return Path(resolve(FAMILY, variant, "torch-fp32"))


def release() -> None:
    """Hand a dropped network's memory back, so the next one has the GPU to itself."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --- recording -----------------------------------------------------------------------------
#
# Each side records every stage into an ordered ``{stage: value}`` on the host. The records are
# then compared stage by stage in the vendor's order, which is execution order, so the first
# disagreement is the first stage that moved. Recording the sides apart is what lets a side be run
# twice (determinism), the reference be cached across falsifiers, and one network at a time sit
# on the GPU.

class Digest(NamedTuple):
    """What the reference cache and the repeat runs keep of a stage: shape, dtype, sha256."""

    shape: tuple
    dtype: str
    sha256: str


def host(value) -> torch.Tensor:
    """A stage's value as a host tensor, whatever it came as."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return torch.as_tensor(np.asarray(value))


def digest(value) -> Digest:
    if isinstance(value, Digest):
        return value
    flat = value.contiguous().reshape(-1)
    return Digest(tuple(value.shape), str(value.dtype),
                  hashlib.sha256(flat.view(torch.uint8).numpy().tobytes()).hexdigest())


def match(got, want) -> tuple[bool, str]:
    """Exact equality, with a description of the difference when there is one."""
    if isinstance(got, Digest) or isinstance(want, Digest):
        a, b = digest(got), digest(want)
        if a == b:
            return True, ""
        if a.shape != b.shape or a.dtype != b.dtype:
            return False, f"{a.shape} {a.dtype} vs {b.shape} {b.dtype}"
        return False, "bytes differ"
    if got.shape == want.shape and got.dtype == want.dtype and torch.equal(got, want):
        return True, ""
    if got.shape == want.shape and got.is_floating_point() and want.is_floating_point():
        return False, f"max|d| {(got.float() - want.float()).abs().max().item():.3g}"
    return False, f"{tuple(got.shape)} {got.dtype} vs {tuple(want.shape)} {want.dtype}"


class Caught(Exception):
    """Raised by a :class:`Record` checking against a reference at its first difference."""


class Record:
    """One side's stages, in the order they were recorded.

    Args:
        keep: ``"full"`` host tensors, ``"digest"`` only their digests, or ``"none"``.
        against: A reference :class:`Record` to check each stage against as it arrives; the first
            difference raises :class:`Caught` (falsification: a perturbation needs only to be
            caught, and the run that catches it then stops).
    """

    def __init__(self, keep: str = "full", against: "Record | None" = None):
        self.keep = keep
        self.against = against
        self.values: dict[str, torch.Tensor | Digest] = {}
        self.checked = 0
        self.failure: str | None = None

    def __call__(self, stage: str, value) -> None:
        if stage in self.values:
            raise RuntimeError(f"stage recorded twice: {stage}")
        value = host(value)
        if self.against is not None:
            self.checked += 1
            want = self.against.values.get(stage)
            ok, delta = (False, "the reference has no such stage") if want is None else match(value, want)
            if not ok:
                self.failure = f"{stage} ({delta})"
                raise Caught(stage)
        self.values[stage] = (value if self.keep == "full"
                              else digest(value) if self.keep == "digest" else None)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([[k, list(d.shape), d.dtype, d.sha256]
                                    for k, d in ((k, digest(v)) for k, v in self.values.items())]))

    @classmethod
    def load(cls, path: Path) -> "Record":
        record = cls(keep="digest")
        for stage, shape, dtype, sha in json.loads(path.read_text()):
            record.values[stage] = Digest(tuple(shape), dtype, sha)
        return record


def compare(got: Record, want: Record, label: str, quiet: bool = False) -> tuple[int, list[str]]:
    """Every stage of *got* against *want*, in *got*'s order; stages either lacks fail too."""
    failed = []
    for stage, value in got.values.items():
        if stage not in want.values:
            failed.append(f"{stage} (only in {label}'s first record)")
            continue
        ok, delta = match(value, want.values[stage])
        if not ok:
            failed.append(f"{stage} ({delta})")
    failed += [f"{stage} (missing)" for stage in want.values if stage not in got.values]
    if not quiet:
        for line in failed[:40]:
            print(f"  FAIL {label}: {line}")
        if len(failed) > 40:
            print(f"  ... and {len(failed) - 40} more")
    return len(got.values), failed


def trainable(model: torch.nn.Module) -> torch.Tensor:
    """Which parameters require grad: how many of how many, and the sha256 of their sorted names.

    Recorded because it can change arithmetic: ``torch.matmul`` folds a batched, strided product
    into one matrix multiply or runs it batched depending on whether the weight requires grad, and
    on CUDA those are different sums (found by HQ-SAM's GPU gate).
    """
    params = list(model.named_parameters())
    names = sorted(name for name, p in params if p.requires_grad)
    sha = hashlib.sha256("\n".join(names).encode()).digest()
    return torch.tensor([len(names), len(params), *sha], dtype=torch.int64)


def clip_frames(upstream: Path, count: int) -> np.ndarray:
    """The first *count* frames of upstream's bedroom clip, decoded once by PIL."""
    folder = upstream / "notebooks" / "videos" / "bedroom"
    names = sorted(os.listdir(folder), key=lambda p: int(os.path.splitext(p)[0]))[:count]
    return np.stack([np.array(Image.open(folder / n).convert("RGB")) for n in names])


@contextlib.contextmanager
def frame_folder(frames: np.ndarray):
    """The frames as upstream's loader reads them: ``<index>.jpg`` names, stored losslessly.

    PNG bytes under ``.jpg`` names -- upstream lists files by extension and PIL reads them by
    content -- so both sides see the same pixels rather than two JPEG decoders' opinions.
    """
    folder = Path(tempfile.mkdtemp(prefix="sam2_video_gate_"))
    # Upstream warns on every frame that its hole-filling extension is missing (it is, by design).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            for index, frame in enumerate(frames):
                Image.fromarray(frame).save(folder / f"{index:05d}.jpg", format="PNG")
            yield str(folder)
        finally:
            shutil.rmtree(folder)


@contextlib.contextmanager
def recording(net, put, label: str):
    """Record the inputs and outputs of ``forward_image`` and memory attention on one network."""
    forward_image, attention = net.forward_image, net.memory_attention.forward
    calls = {"image": 0, "attention": 0}

    def record_image(batch):
        out = forward_image(batch)
        i = calls["image"]
        calls["image"] += 1
        put(f"{label}: frame {i} pixels", batch[0])
        for level, x in enumerate(out["backbone_fpn"]):
            put(f"{label}: frame {i} image features level {level}", x)
        return out

    def record_attention(*args, **kwargs):
        out = attention(*args, **kwargs)
        put(f"{label}: memory attention call {calls['attention']}", out)
        calls["attention"] += 1
        return out

    net.forward_image = record_image
    net.memory_attention.forward = record_attention
    try:
        yield
    finally:
        del net.__dict__["forward_image"]
        del net.memory_attention.__dict__["forward"]


def frame_output(per_obj: dict, frame_idx: int) -> dict:
    """An object's stored output for a frame, prompted or tracked."""
    return (per_obj["cond_frame_outputs"].get(frame_idx)
            or per_obj["non_cond_frame_outputs"][frame_idx])


OUTPUT_KEYS = ("pred_masks", "obj_ptr", "object_score_logits", "maskmem_features")


def put_outputs(put, label: str, i: int, outputs: list[dict]) -> None:
    for obj, out in enumerate(outputs):
        for key in OUTPUT_KEYS:
            put(f"{label}: frame {i} object {obj} {key}", out[key])


def end_to_end_rows(put, i: int, expected: dict[int, tuple]) -> None:
    """What mozo must hand PixelFlow for frame *i*: per object, a row or not, and the row."""
    import pixelflow as pf

    for obj, (present, mask, logit) in expected.items():
        put(f"end to end: frame {i} object {obj} has a row", torch.tensor(present))
        if present:
            put(f"end to end: frame {i} object {obj} mask", mask)
            # mozo's own number (upstream has no confidence): the sigmoid of the object score,
            # taken on the host as the adapter takes it, then stored by PixelFlow, which rounds
            # to three decimals. The logit itself is gated exactly above.
            stored = pf.Detection(bbox=[0, 0, 1, 1],
                                  confidence=float(torch.sigmoid(host(logit).reshape(()))))
            put(f"end to end: frame {i} object {obj} confidence", torch.tensor(stored.confidence))
            # class id, tracker id, and whether the name is the one given
            put(f"end to end: frame {i} object {obj} ids and name", torch.tensor([obj, obj, 1]))


# --- the reference -------------------------------------------------------------------------

def build_reference(upstream: Path, variant: str, checkpoint: Path, device: str):
    """Upstream's video and image predictors, built the documented way, on *device*."""
    if str(upstream) not in sys.path:
        sys.path.insert(1, str(upstream))
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    config = f"configs/sam2.1/sam2.1_hiera_{CONFIGS[variant]}.yaml"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        video = build_sam2_video_predictor(config, str(checkpoint), device=device)
        image = SAM2ImagePredictor(build_sam2(config, str(checkpoint), device=device))
    return video, image


def reference_offline(video, folder: str, prompts: dict[int, list[tuple[int, dict]]],
                      reverse_from: int | None = None):
    """Upstream offline: prompts {frame: [(obj_id, prompt)]}, then ``propagate_in_video``."""
    state = video.init_state(folder)
    for frame_idx, items in prompts.items():
        for obj_id, prompt in items:
            video.add_new_points_or_box(state, frame_idx=frame_idx, obj_id=obj_id, **prompt)
    logits = {}
    walk = (video.propagate_in_video(state) if reverse_from is None
            else video.propagate_in_video(state, start_frame_idx=reverse_from, reverse=True))
    for frame_idx, _, masks in walk:
        logits[frame_idx] = masks[:, 0].clone()
    return state, logits


def reference_streamed(video, folder: str, count: int, prompts: dict[int, list[tuple[int, dict]]]):
    """Upstream's own predictor driven one frame at a time, prompts added as their frame arrives."""
    import sam2.sam2_video_predictor as predictor_module

    images, height, width = predictor_module.load_video_frames(
        folder, video.image_size, False, compute_device=video.device)

    class Arriving:
        """Frames handed over one at a time; reading one that has not arrived raises."""

        arrived = 0

        def __len__(self):
            return 10**9

        def __getitem__(self, index):
            if index > self.arrived:
                raise RuntimeError(f"frame {index} read while frame {self.arrived} is current")
            return images[index]

    arriving = Arriving()
    original = predictor_module.load_video_frames
    predictor_module.load_video_frames = lambda **_: (arriving, height, width)
    try:
        state = video.init_state(folder)
    finally:
        predictor_module.load_video_frames = original
    logits = {}
    for frame_idx in range(count):
        arriving.arrived = frame_idx
        for obj_id, prompt in prompts.get(frame_idx, []):
            video.add_new_points_or_box(state, frame_idx=frame_idx, obj_id=obj_id, **prompt)
        for index, _, masks in video.propagate_in_video(
                state, start_frame_idx=frame_idx, max_frame_num_to_track=0):
            logits[index] = masks[:, 0].clone()
    return state, logits


def image_calls(frames: np.ndarray):
    """The image-path calls, as upstream's ``predict`` keywords."""
    click = dict(point_coords=np.array(CLICK["points"]), point_labels=np.array(CLICK["labels"]))
    return (("click, three masks", dict(click, multimask_output=True)),
            ("click, single mask", dict(click, multimask_output=False)),
            # A click whose single mask is unstable, so the decoder's fallback fires (it changes
            # the mask by 1,252 pixels here; 21 of 40 random clicks).
            ("unstable click, single mask", dict(point_coords=np.array(PAIR["points"][:1]),
                                                 point_labels=np.array([1]),
                                                 multimask_output=False)),
            ("box, single mask", dict(box=np.array(BOX["box"]), multimask_output=False)))


def record_reference(video, image_predictor, frames: np.ndarray, put) -> None:
    """Every stage upstream produces, through its own predictors, in its own order."""
    count = len(frames)

    # three objects, offline
    with frame_folder(frames) as folder, recording(video, put, "three objects"):
        state, logits = reference_offline(video, folder, {0: [(1, CLICK), (2, BOX), (3, PAIR)]})
    for i in range(count):
        put(f"three objects: frame {i} logits at video size", logits[i])
        put_outputs(put, "three objects", i,
                    [frame_output(o, i) for o in state["output_dict_per_obj"].values()])

    # added later: frame 0 a box, frame `later` a click, streamed
    later = min(8, count - 1)
    with frame_folder(frames) as folder:
        _, logits = reference_streamed(video, folder, count, {0: [(1, BOX)], later: [(2, CLICK)]})
    for i in range(count):
        put(f"added later: frame {i} object count", torch.tensor(logits[i].shape[0]))
        put(f"added later: frame {i} logits at video size", logits[i])

    # short clip: 10 frames
    short = frames[:10]
    with frame_folder(short) as folder:
        state, logits = reference_offline(video, folder, {0: [(1, CLICK), (2, BOX)]})
    for i in range(len(short)):
        put(f"short clip: frame {i} logits at video size", logits[i])
        put_outputs(put, "short clip", i,
                    [frame_output(o, i) for o in state["output_dict_per_obj"].values()])

    # reversed: prompt the middle frame, reverse=True
    middle = count // 2
    with frame_folder(frames) as folder:
        _, logits = reference_offline(video, folder, {middle: [(1, BOX), (2, CLICK)]},
                                      reverse_from=middle)
    for i, frame_idx in enumerate(range(middle, -1, -1)):
        put(f"reversed: frame {i} logits at video size", logits[frame_idx])

    # image path, including single-mask calls
    photo = frames[0]
    image_predictor.set_image(photo)
    for label, kw in image_calls(frames):
        masks, scores, low_res = image_predictor.predict(return_logits=True, **kw)
        put(f"image: {label}: masks", masks > image_predictor.mask_threshold)
        put(f"image: {label}: IoU", scores)
        put(f"image: {label}: low-res logits, clamped as mozo returns them",
            torch.from_numpy(low_res).clamp(-LOGIT_LIMIT, LOGIT_LIMIT))

    # end to end: two boxes; what mozo's rows must say
    with frame_folder(frames) as folder:
        state, logits = reference_offline(video, folder, {0: [(1, BOX), (2, SECOND_BOX)]})
    for i in range(count):
        expected = {}
        for obj in (0, 1):
            score = frame_output(state["output_dict_per_obj"][obj], i)["object_score_logits"]
            mask = logits[i][obj] > 0
            present = bool(score.reshape(()) > 0) and bool(mask.any())
            expected[obj] = (present, mask, score)
        end_to_end_rows(put, i, expected)

    put("model: parameters that require grad (video)", trainable(video))
    put("model: parameters that require grad (image)", trainable(image_predictor.model))


# --- the vendor and mozo -------------------------------------------------------------------

def run_vendor(segmenter, frames: np.ndarray, prompts: dict[int, list[dict]], put, label: str,
               num_frames: int | None = None, outputs: bool = True, order=None) -> None:
    """The vendored session over *frames*, recording each frame as it is returned."""
    tracker = segmenter.tracker(num_frames=num_frames)
    for i, frame in enumerate(frames):
        result = tracker.update(frame, prompts.get(i))
        if label == "added later":
            put(f"{label}: frame {i} object count", torch.tensor(result.logits.shape[0]))
        put(f"{label}: frame {i} logits at video size", result.logits)
        if outputs:
            put_outputs(put, label, i, [frame_output(o, i) for o in tracker.outputs])


def record_vendor(segmenter, model, frames: np.ndarray, put) -> None:
    """The same stages from the vendor, and from mozo end to end when *model* is given."""
    count = len(frames)
    with recording(segmenter.network, put, "three objects"):
        run_vendor(segmenter, frames, {0: [CLICK, BOX, PAIR]}, put, "three objects")

    later = min(8, count - 1)
    run_vendor(segmenter, frames, {0: [BOX], later: [CLICK]}, put, "added later", outputs=False)

    short = frames[:10]
    run_vendor(segmenter, short, {0: [CLICK, BOX]}, put, "short clip", num_frames=len(short))

    middle = count // 2
    run_vendor(segmenter, frames[middle::-1], {0: [BOX, CLICK]}, put, "reversed", outputs=False)

    photo = frames[0]
    for label, kw in image_calls(frames):
        found = segmenter.predict(
            photo, points=kw.get("point_coords"), labels=kw.get("point_labels"),
            boxes=kw.get("box"), multimask_output=kw["multimask_output"])
        put(f"image: {label}: masks", found.masks[0])
        put(f"image: {label}: IoU", found.scores[0])
        put(f"image: {label}: low-res logits, clamped as mozo returns them", found.logits[0])

    if model is not None:
        record_end_to_end(model, frames, put)

    put("model: parameters that require grad (video)", trainable(segmenter.network))
    put("model: parameters that require grad (image)", trainable(segmenter.network))


def record_end_to_end(model, frames: np.ndarray, put) -> None:
    """Registry -> weights -> ``Sam2Predictor.tracker()`` -> PixelFlow rows.

    Two boxes, because one ``update`` takes prompts of one shape (as ``predict`` does).
    """
    tracker = model.tracker()
    for i, frame in enumerate(frames):
        found = (tracker.update(frame, boxes=[BOX["box"], SECOND_BOX["box"]],
                                name=["first", "second"]) if i == 0 else tracker.update(frame))
        rows = {row.tracker_id: row for row in found}
        for obj, label in ((0, "first"), (1, "second")):
            row = rows.get(obj)
            put(f"end to end: frame {i} object {obj} has a row", torch.tensor(row is not None))
            if row is not None:
                put(f"end to end: frame {i} object {obj} mask", np.asarray(row.masks[0]))
                put(f"end to end: frame {i} object {obj} confidence", torch.tensor(row.confidence))
                put(f"end to end: frame {i} object {obj} ids and name",
                    torch.tensor([row.class_id, row.tracker_id, int(row.class_name == label)]))


# --- the environment and the reference cache -----------------------------------------------

def environment(device: str) -> dict[str, str]:
    """The stack a result was measured on: printed with every run, and the cache's key."""
    env = {"torch": torch.__version__, "Pillow": PIL.__version__, "numpy": np.__version__,
           "device": device}
    if device == "cuda":
        try:
            driver = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "-i",
                 str(torch.cuda.current_device())],
                capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            driver = "unknown"
        b = torch.backends
        env.update({
            "CUDA": str(torch.version.cuda), "cuDNN": str(b.cudnn.version()),
            "GPU": torch.cuda.get_device_name(), "driver": driver,
            "TF32 matmul": str(b.cuda.matmul.allow_tf32), "TF32 cuDNN": str(b.cudnn.allow_tf32),
            "cudnn.benchmark": str(b.cudnn.benchmark),
            "cudnn.deterministic": str(b.cudnn.deterministic),
            "deterministic algorithms": str(torch.are_deterministic_algorithms_enabled()),
            "SDPA flash/efficient/math/cudnn": "/".join(str(f()) for f in (
                b.cuda.flash_sdp_enabled, b.cuda.mem_efficient_sdp_enabled,
                b.cuda.math_sdp_enabled, b.cuda.cudnn_sdp_enabled)),
        })
    else:
        env.update({"threads": str(torch.get_num_threads()),
                    "MKL_CBWR": str(os.environ.get("MKL_CBWR"))})
    return env


def cache_path(cache: Path, env: dict, variant: str, checkpoint: Path, frames: np.ndarray,
               head: str) -> Path:
    key = json.dumps([RECORD_VERSION, env, variant, checkpoint.stat().st_size, head,
                      hashlib.sha256(frames.tobytes()).hexdigest()], sort_keys=True)
    stem = f"{variant}-{env['device']}-fp32-{len(frames)}f"
    return cache / f"{stem}-{hashlib.sha256(key.encode()).hexdigest()[:16]}.json"


def repeated(runs: int, record_once, label: str) -> tuple[Record, int, list[str]]:
    """Record a side *runs* times; every later run must equal the first byte for byte."""
    first = Record(keep="full")
    record_once(first)
    checked, failed = 0, []
    for run in range(1, runs):
        again = Record(keep="digest")
        record_once(again)
        n, bad = compare(again, first, f"{label} run {run + 1} vs run 1")
        checked += n
        failed += bad
    if runs > 1:
        print(f"  determinism, {label}: {runs} runs, "
              f"{'repeatable' if not failed else f'{len(failed)} stages moved'} "
              f"({len(first.values)} stages each)", flush=True)
    return first, checked, failed


def reference_record(upstream: Path, variant: str, checkpoint: Path, frames: np.ndarray,
                     device: str, runs: int, cache: Path | None, env: dict,
                     head: str) -> tuple[Record, int, list[str]]:
    """The reference, from the cache when it holds this stack's record, else recorded now."""
    path = cache_path(cache, env, variant, checkpoint, frames, head) if cache else None
    if path is not None and path.is_file():
        print(f"  reference: {path} (cached; its determinism was checked when it was recorded)")
        return Record.load(path), 0, []
    video, image_predictor = build_reference(upstream, variant, checkpoint, device)
    record, checked, failed = repeated(
        runs, lambda put: record_reference(video, image_predictor, frames, put), "upstream")
    video = image_predictor = None  # off the GPU before the vendor is built
    release()
    if path is not None and not failed:
        record.save(path)
        print(f"  reference saved to {path}")
    return record, checked, failed


def check_variant(upstream: Path, variant: str, checkpoint: Path, frames: np.ndarray,
                  device: str, runs: int, cache: Path | None, env: dict, head: str) -> tuple:
    """The gate for one variant: (comparisons, failures, determinism comparisons, failures)."""
    print(f"\n{variant}  ({checkpoint.name})", flush=True)
    want, det_n, det_failed = reference_record(upstream, variant, checkpoint, frames, device,
                                               runs, cache, env, head)

    import mozo

    segmenter = Segmenter(checkpoint, variant=variant, device=device)
    model = mozo.get_model(f"{FAMILY}/{variant}", device=device)
    if model.runtime != "torch-fp32":
        raise SystemExit(f"mozo chose {model.runtime} on {device}; this gate checks torch-fp32")
    got, n, bad = repeated(runs, lambda put: record_vendor(segmenter, model, frames, put), "mozo")
    det_n += n
    det_failed += bad
    segmenter = model = None
    release()

    checked, failed = compare(got, want, "mozo vs upstream")
    print(f"  {checked} stages compared; {len(failed)} differ"
          + (f"; first: {failed[0]}" if failed else ""), flush=True)
    return checked, failed, det_n, det_failed


# --- falsification -------------------------------------------------------------------------

@contextlib.contextmanager
def perturbed(name: str):
    """Apply one :data:`FALSIFIERS` perturbation to the vendored side, and undo it after."""
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from mozo.vendors.sam2_deploy import image as image_module
    from mozo.vendors.sam2_deploy import memory_attention
    from mozo.vendors.sam2_deploy import predictor as predictor_module
    from mozo.vendors.sam2_deploy import tracking

    undo = []
    switches = contextlib.ExitStack()

    def patch(owner, attribute, value):
        undo.append((owner, attribute, owner.__dict__.get(attribute, _MISSING)))
        setattr(owner, attribute, value)

    if name == "image-path-resize":
        patch(vendor_video, "load_frame", lambda image, size: image_module.preprocess(image, size))
    elif name == "no-bf16-memory":
        original = vendor_video.Tracker._compact

        def compact(self, out, features, pos_enc):
            kept = original(self, out, features, pos_enc)
            kept["maskmem_features"] = features
            return kept
        patch(vendor_video.Tracker, "_compact", compact)
    elif name == "pointer-window":
        original = vendor_network.Sam2._build_tracker

        def build(self, hidden):
            original(self, hidden)
            self.max_obj_ptrs_in_encoder = 17
        patch(vendor_network.Sam2, "_build_tracker", build)
    elif name == "short-reach":
        original = vendor_video.Tracker.__init__

        def init(self, *args, **kwargs):
            original(self, *args, **kwargs)
            self.reach = 6
        patch(vendor_video.Tracker, "__init__", init)
    elif name == "memory-from-high-res":
        original = tracking.TrackingStep.track_step

        def track_step(self, *args, **kwargs):
            out = original(self, *args, **kwargs)
            if not kwargs.get("run_mem_encoder", True):
                # make the stored low-res logits round-trip to the high-res ones
                out["pred_masks"] = F.interpolate(out["pred_masks_high_res"], size=(256, 256),
                                                  mode="bilinear", align_corners=False)
            return out
        patch(tracking.TrackingStep, "track_step", track_step)
    elif name == "no-stability-fallback":
        original = vendor_network.Sam2.__init__

        def init(self, spec):
            original(self, spec)
            self.sam_mask_decoder.dynamic_multimask_via_stability = False
        patch(vendor_network.Sam2, "__init__", init)
    elif name == "frozen":
        original = Segmenter.__init__

        def init(self, *args, **kwargs):
            original(self, *args, **kwargs)
            self.network.requires_grad_(False)
        patch(Segmenter, "__init__", init)
    elif name == "tf32-matmul":
        flag = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        switches.callback(setattr, torch.backends.cuda.matmul, "allow_tf32", flag)
    elif name == "cudnn-tf32-off":
        flag = torch.backends.cudnn.allow_tf32
        torch.backends.cudnn.allow_tf32 = False
        switches.callback(setattr, torch.backends.cudnn, "allow_tf32", flag)
    elif name == "channels-last":
        original = vendor_video.load_frame
        patch(vendor_video, "load_frame", lambda image, size: original(image, size).contiguous(
            memory_format=torch.channels_last))
    elif name == "sdpa-math":
        switches.enter_context(sdpa_kernel(SDPBackend.MATH))
    elif name == "memory-attention-host":
        original = memory_attention.MemoryAttention.forward

        def to(value, device):
            if isinstance(value, torch.Tensor):
                return value.to(device)
            if isinstance(value, list):
                return [to(v, device) for v in value]
            return value

        def forward(self, *args, **kwargs):
            device = next(self.parameters()).device
            twin = self.__dict__.setdefault("_host_twin", [copy.deepcopy(self).cpu()])[0]
            out = original(twin, *to(list(args), "cpu"),
                           **{k: to(v, "cpu") for k, v in kwargs.items()})
            return out.to(device)
        patch(memory_attention.MemoryAttention, "forward", forward)
    elif name == "prompt-coords-host":
        original = image_module.to_model_coords
        patch(predictor_module, "to_model_coords",
              lambda coords, shape, size, device=None: original(coords, shape, size).to(device))
    else:
        raise SystemExit(f"unknown falsifier {name!r}; choose from {sorted(FALSIFIERS)}")
    try:
        with switches:
            yield
    finally:
        for owner, attribute, value in reversed(undo):
            if value is _MISSING:
                delattr(owner, attribute)
            else:
                setattr(owner, attribute, value)


_MISSING = object()


def falsify(upstream: Path, variant: str, checkpoint: Path, frames: np.ndarray, device: str,
            cache: Path | None, env: dict, head: str, names: list[str] | None = None) -> int:
    """Run the gate under each perturbation (or the *names* given); every one must fail."""
    print(f"\n{variant}  ({checkpoint.name}), falsification on {device}", flush=True)
    want, _, failed = reference_record(upstream, variant, checkpoint, frames, device, 1, cache,
                                       env, head)
    if failed:
        raise SystemExit("the reference did not repeat itself")
    caught, missed = [], []
    for name, (what, only) in FALSIFIERS.items():
        if names and name not in names:
            continue
        if only == "cuda" and device != "cuda":
            print(f"  skip    {name:<22} (catchable only on CUDA)")
            continue
        got = Record(keep="none", against=want)
        with perturbed(name), contextlib.redirect_stdout(None), contextlib.suppress(Caught):
            segmenter = Segmenter(checkpoint, variant=variant, device=device)
            record_vendor(segmenter, None, frames, got)
        segmenter = None
        release()
        (caught if got.failure else missed).append(name)
        print(f"  {'caught' if got.failure else 'MISSED':<7} {name:<22} "
              f"after {got.checked} comparisons; first failure: {got.failure or '-'}", flush=True)
        print(f"          {what}")
    print()
    if missed:
        print(f"{len(missed)} perturbation(s) passed the gate: {missed}")
        return 1
    print(f"all {len(caught)} perturbations caught.\nSAM 2 video falsification: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("variants", nargs="*", default=["tiny"],
                        help="variants to check (default: tiny)")
    parser.add_argument("--upstream", type=Path, required=True,
                        help="checkout of facebookresearch/sam2")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                        help="where both sides run (default cpu; cuda is the primary gate)")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32",
                        help="precision; mozo runs SAM 2 in fp32 only, so only fp32 is gated")
    parser.add_argument("--frames", type=int, default=20,
                        help="frames of the bedroom clip to track (default 20, at least 16)")
    parser.add_argument("--repeats", type=int, default=2,
                        help="runs per side, each equal to the first byte for byte (default 2)")
    parser.add_argument("--reference-cache", type=Path, default=None,
                        help="folder for the reference's digests, reused on the same stack")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="a checkpoint to check instead of the published one (one variant)")
    parser.add_argument("--falsify", action="store_true",
                        help="run each deliberate perturbation; every one must fail")
    parser.add_argument("--falsifier", action="append", choices=sorted(FALSIFIERS),
                        help="with --falsify, run only this perturbation (repeatable)")
    parser.add_argument("--allow-commit-drift", action="store_true",
                        help="run against an upstream other than the extracted commit")
    arguments = parser.parse_args()

    for variant in arguments.variants:
        if variant not in CONFIGS:
            raise SystemExit(f"{variant!r} is not published; choose from {list(CONFIGS)}")
    if arguments.dtype != "fp32":
        raise SystemExit("mozo runs SAM 2 only in fp32 (torch-fp32 is its one runtime, on CUDA "
                         "too); there is no half-precision path to gate")
    if arguments.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda, but torch sees no CUDA device")
    if arguments.checkpoint is not None and len(arguments.variants) != 1:
        raise SystemExit("--checkpoint checks one variant; name it")
    if arguments.frames < 16:
        raise SystemExit("--frames must be at least 16 (the short-clip case covers fewer)")
    if arguments.repeats < 1:
        raise SystemExit("--repeats must be at least 1")

    upstream = arguments.upstream.resolve()
    if not (upstream / "sam2" / "sam2_video_predictor.py").is_file():
        raise SystemExit(f"no sam2 checkout at {upstream}. Clone it:\n"
                         f"  git clone https://github.com/facebookresearch/sam2 {upstream}")
    head = subprocess.run(["git", "-C", str(upstream), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    pinned = pinned_commit()
    if head != pinned and not arguments.allow_commit_drift:
        raise SystemExit(f"upstream is at {head[:12]}, PROVENANCE.md records {pinned[:12]}.\n"
                         "Check it out, or pass --allow-commit-drift.")

    torch.set_grad_enabled(False)
    device = arguments.device
    env = environment(device)
    print(f"upstream   {upstream} @ {head[:12]}")
    print("stack      " + ", ".join(f"{k} {v}" for k, v in env.items()))
    print(f"precision  fp32; {arguments.frames} frames; {arguments.repeats} runs per side")
    frames = clip_frames(upstream, arguments.frames)
    checkpoints = {v: (arguments.checkpoint or published(v)).absolute() for v in arguments.variants}

    if arguments.falsify:
        variant = arguments.variants[0]
        return falsify(upstream, variant, checkpoints[variant], frames, device,
                       arguments.reference_cache, env, head, arguments.falsifier)

    checked = det_checked = 0
    failed, det_failed = [], []
    for variant in arguments.variants:
        n, bad, dn, dbad = check_variant(upstream, variant, checkpoints[variant], frames, device,
                                         arguments.repeats, arguments.reference_cache, env, head)
        checked += n
        failed += [f"{variant}: {line}" for line in bad]
        det_checked += dn
        det_failed += [f"{variant}: {line}" for line in dbad]
    print()
    print(f"determinism: {det_checked - len(det_failed)} of {det_checked} repeat comparisons "
          "byte-identical")
    if failed or det_failed:
        print(f"{len(failed)} of {checked} comparisons differ.\nSAM 2 video gate: FAIL")
        return 1
    print(f"all {checked} comparisons bit-identical on {device} (fp32).\nSAM 2 video gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

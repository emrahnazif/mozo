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

**The reference is not patched.** Upstream at this commit has no attention-backend pin, and the
CUDA extension it uses to fill holes is absent here, as on every CPU install -- so it skips that
step, which is what the vendor reproduces (PROVENANCE).

Run from the repository root, in an environment with torch, Pillow, hydra-core and iopath::

    git clone https://github.com/facebookresearch/sam2 ~/src/sam2
    git -C ~/src/sam2 checkout <commit in PROVENANCE.md>
    MKL_CBWR=AVX2,STRICT python tools/verify/sam2_video.py --upstream ~/src/sam2            # tiny
    MKL_CBWR=AVX2,STRICT python tools/verify/sam2_video.py --upstream ~/src/sam2 small large
    MKL_CBWR=AVX2,STRICT python tools/verify/sam2_video.py --upstream ~/src/sam2 --falsify

``--falsify`` runs the gate once per deliberate perturbation of the vendored side and passes only
if every one is caught. The clip is ``notebooks/videos/bedroom`` from the checkout itself.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

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

FAMILY = "sam2"
PROVENANCE = ROOT / "mozo" / "vendors" / "sam2_deploy" / "PROVENANCE.md"
CONFIGS = {"tiny": "t", "small": "s", "base_plus": "b+", "large": "l"}

#: Prompts in the bedroom clip's own pixels (960x540), as upstream's ``add_new_points_or_box``
#: takes them. A click alone reaches the three-candidate choice on the prompted frame; a box is
#: two points and reaches the single-mask stability fallback; a click pair has a negative.
CLICK = {"points": [[210, 350]], "labels": [1], "box": None}
BOX = {"points": None, "labels": None, "box": [300, 0, 500, 400]}
PAIR = {"points": [[460, 60], [510, 220]], "labels": [1, 0], "box": None}
SECOND_BOX = {"points": None, "labels": None, "box": [560, 180, 760, 470]}

#: Deliberate perturbations of the vendored side, each of which must make the gate fail, with the
#: stage it should first fail at.
FALSIFIERS = {
    "image-path-resize": "frames resized like the image predictor (antialiased bilinear) instead "
                         "of the video loader's PIL bicubic (fails at: pixels)",
    "float32-divide": "divide the resized frame by 255 in float32, not float64 (fails at: pixels)",
    "no-bf16-memory": "keep mask memories in float32 (fails at: memory attention, frame 2)",
    "pointer-window": "object-pointer temporal encoding normalised by 16 instead of 15 "
                      "(fails at: memory attention, frame 2)",
    "short-reach": "forget frames older than 6 instead of 15 (fails at: memory attention, frame 8)",
    "memory-from-high-res": "store the prompted frame's logits round-tripped through the "
                            "decoder's high-res masks (fails at: frame 0 logits)",
    "no-stability-fallback": "decoder without the dynamic multimask fallback (fails at: an "
                             "unstable single-mask image click, if no tracking frame reaches it)",
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


class Caught(Exception):
    """Raised by a :class:`Comparison` that stops at its first failure (falsification)."""


class Comparison:
    """Collects exact comparisons and reports each failure.

    With *stop*, the first failure raises :class:`Caught`: a falsifier needs only to be caught,
    and the run that catches it then costs minutes instead of the whole gate.
    """

    def __init__(self, quiet: bool = False, stop: bool = False):
        self.quiet = quiet
        self.stop = stop
        self.checked = 0
        self.failed: list[str] = []

    def __call__(self, stage: str, got, want) -> None:
        self.checked += 1
        got = torch.as_tensor(got)
        want = torch.as_tensor(want)
        ok = got.shape == want.shape and got.dtype == want.dtype and torch.equal(got, want)
        if not ok:
            if got.shape == want.shape and got.is_floating_point() and want.is_floating_point():
                delta = f"max|d| {(got.float() - want.float()).abs().max().item():.3g}"
            else:
                delta = f"{tuple(got.shape)} {got.dtype} vs {tuple(want.shape)} {want.dtype}"
            self.failed.append(f"{stage} ({delta})")
            if self.stop:
                raise Caught(stage)
            if not self.quiet:
                print(f"  FAIL {stage}: {delta}")


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


class Recorder:
    """Records the inputs and outputs of ``forward_image`` and memory attention on one network."""

    def __init__(self, net):
        self.pixels, self.features, self.conditioned = [], [], []
        forward_image, attention = net.forward_image, net.memory_attention.forward

        def record_image(batch):
            out = forward_image(batch)
            self.pixels.append(batch[0].clone())
            self.features.append([x.clone() for x in out["backbone_fpn"]])
            return out

        def record_attention(*args, **kwargs):
            out = attention(*args, **kwargs)
            self.conditioned.append(out.clone())
            return out

        net.forward_image = record_image
        net.memory_attention.forward = record_attention
        self._undo = [(net, "forward_image"), (net.memory_attention, "forward")]

    def close(self):
        for owner, name in self._undo:
            del owner.__dict__[name]


def build_reference(upstream: Path, variant: str, checkpoint: Path):
    """Upstream's video and image predictors, built the documented way."""
    if str(upstream) not in sys.path:
        sys.path.insert(1, str(upstream))
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    config = f"configs/sam2.1/sam2.1_hiera_{CONFIGS[variant]}.yaml"
    video = build_sam2_video_predictor(config, str(checkpoint), device="cpu")
    image = SAM2ImagePredictor(build_sam2(config, str(checkpoint), device="cpu"))
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
        folder, video.image_size, False, compute_device=torch.device("cpu"))

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


def frame_output(per_obj: dict, frame_idx: int) -> dict:
    """An object's stored output for a frame, prompted or tracked."""
    return (per_obj["cond_frame_outputs"].get(frame_idx)
            or per_obj["non_cond_frame_outputs"][frame_idx])


def run_vendor(segmenter, frames: np.ndarray, prompts: dict[int, list[dict]],
               num_frames: int | None = None):
    """The vendored session over *frames*; returns per-frame logits and each frame's outputs."""
    tracker = segmenter.tracker(num_frames=num_frames)
    logits, outputs = [], []
    for index, frame in enumerate(frames):
        result = tracker.update(frame, prompts.get(index))
        logits.append(result.logits.clone())
        outputs.append([frame_output(output, index) for output in tracker.outputs])
    return logits, outputs


def compare_frames(same: Comparison, label: str, got_logits, want_logits, order,
                   got_outputs=None, want_state=None):
    """Logits (and stored outputs) per frame. *order*: vendor frame i -> upstream frame index."""
    keys = ("pred_masks", "obj_ptr", "object_score_logits", "maskmem_features")
    for i, frame_idx in enumerate(order):
        same(f"{label}: frame {i} logits at video size", got_logits[i], want_logits[frame_idx])
        if got_outputs is None:
            continue
        for obj, got_out in enumerate(got_outputs[i]):
            want_out = frame_output(want_state["output_dict_per_obj"][obj], frame_idx)
            for key in keys:
                same(f"{label}: frame {i} object {obj} {key}", got_out[key], want_out[key])


def compare_recorders(same: Comparison, label: str, got: Recorder, want: Recorder) -> None:
    """Pixels, image features and memory-conditioned features, call by call."""
    if len(got.pixels) != len(want.pixels) or len(got.conditioned) != len(want.conditioned):
        same(f"{label}: number of encoder / attention calls",
             torch.tensor([len(got.pixels), len(got.conditioned)]),
             torch.tensor([len(want.pixels), len(want.conditioned)]))
    for i, (a, b) in enumerate(zip(got.pixels, want.pixels)):
        same(f"{label}: frame {i} pixels", a, b)
    for i, (a, b) in enumerate(zip(got.features, want.features)):
        for level, (x, y) in enumerate(zip(a, b)):
            same(f"{label}: frame {i} image features level {level}", x, y)
    for i, (a, b) in enumerate(zip(got.conditioned, want.conditioned)):
        same(f"{label}: memory attention call {i}", a, b)


def check_variant(upstream: Path, variant: str, checkpoint: Path, frames: np.ndarray,
                  same: Comparison, end_to_end: bool = True) -> None:
    print(f"\n{variant}  ({checkpoint.name})")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        video, image_predictor = build_reference(upstream, variant, checkpoint)
    segmenter = Segmenter(checkpoint, variant=variant, device="cpu")
    count = len(frames)

    # three objects, offline
    with frame_folder(frames) as folder:
        want_rec = Recorder(video)
        state, want = reference_offline(video, folder, {0: [(1, CLICK), (2, BOX), (3, PAIR)]})
        want_rec.close()
    got_rec = Recorder(segmenter.network)
    got, outputs = run_vendor(segmenter, frames, {0: [CLICK, BOX, PAIR]})
    got_rec.close()
    compare_recorders(same, "three objects", got_rec, want_rec)
    compare_frames(same, "three objects", got, want, range(count), outputs, state)
    print(f"  three objects, {count} frames: done")

    # added later: frame 0 a box, frame `later` a click, streamed on both sides
    later = min(8, count - 1)
    with frame_folder(frames) as folder:
        _, want = reference_streamed(video, folder, count, {0: [(1, BOX)], later: [(2, CLICK)]})
    got, _ = run_vendor(segmenter, frames, {0: [BOX], later: [CLICK]})
    for i in range(count):
        same(f"added later: frame {i} object count", torch.tensor(got[i].shape[0]),
             torch.tensor(want[i].shape[0]))
    compare_frames(same, "added later", got, want, range(count))
    print(f"  added later (object 2 from frame {later}): done")

    # short clip: 10 frames, the vendor told the length
    short = frames[:10]
    with frame_folder(short) as folder:
        state, want = reference_offline(video, folder, {0: [(1, CLICK), (2, BOX)]})
    got, outputs = run_vendor(segmenter, short, {0: [CLICK, BOX]}, num_frames=len(short))
    compare_frames(same, "short clip", got, want, range(len(short)), outputs, state)
    print("  short clip (10 frames, length given): done")

    # reversed: prompt the middle frame, upstream reverse=True vs frames fed backwards
    middle = count // 2
    with frame_folder(frames) as folder:
        _, want = reference_offline(video, folder, {middle: [(1, BOX), (2, CLICK)]},
                                    reverse_from=middle)
    backwards = frames[middle::-1]
    got, _ = run_vendor(segmenter, backwards, {0: [BOX, CLICK]})
    compare_frames(same, "reversed", got, want, range(middle, -1, -1))
    print(f"  reversed (from frame {middle} to 0): done")

    # image path, including single-mask calls
    photo = frames[0]
    image_predictor.set_image(photo)
    click = dict(point_coords=np.array(CLICK["points"]), point_labels=np.array(CLICK["labels"]))
    for label, kw in (("click, three masks", dict(click, multimask_output=True)),
                      ("click, single mask", dict(click, multimask_output=False)),
                      # A click whose single mask is unstable, so the decoder's fallback fires
                      # (it changes the mask by 1,252 pixels here; 21 of 40 random clicks).
                      ("unstable click, single mask", dict(point_coords=np.array(PAIR["points"][:1]),
                                                           point_labels=np.array([1]),
                                                           multimask_output=False)),
                      ("box, single mask", dict(box=np.array(BOX["box"]), multimask_output=False))):
        masks, scores, low_res = image_predictor.predict(return_logits=True, **kw)
        found = segmenter.predict(
            photo, points=kw.get("point_coords"), labels=kw.get("point_labels"),
            boxes=kw.get("box"), multimask_output=kw["multimask_output"])
        same(f"image: {label}: masks", torch.from_numpy(found.masks[0]),
             torch.from_numpy(masks > image_predictor.mask_threshold))
        same(f"image: {label}: IoU", torch.from_numpy(found.scores[0]), torch.from_numpy(scores))
    print("  image path: done")

    if end_to_end:
        check_end_to_end(variant, frames, same, video)


def check_end_to_end(variant: str, frames: np.ndarray, same: Comparison, video) -> None:
    """Registry -> weights -> ``Sam2Predictor.tracker()`` -> PixelFlow, against upstream's masks.

    Two boxes, because one ``update`` takes prompts of one shape (as ``predict`` does).
    """
    import pixelflow as pf

    import mozo

    tracker = mozo.get_model(f"{FAMILY}/{variant}", device="cpu").tracker()
    with frame_folder(frames) as folder:
        state, want = reference_offline(video, folder, {0: [(1, BOX), (2, SECOND_BOX)]})
    for index, frame in enumerate(frames):
        found = (tracker.update(frame, boxes=[BOX["box"], SECOND_BOX["box"]],
                                name=["first", "second"]) if index == 0 else tracker.update(frame))
        rows = {row.tracker_id: row for row in found}
        for obj, label in ((0, "first"), (1, "second")):
            score = frame_output(state["output_dict_per_obj"][obj], index)[
                "object_score_logits"].reshape(())
            mask = want[index][obj] > 0
            expected = bool(score > 0) and bool(mask.any())
            same(f"end to end: frame {index} object {obj} has a row",
                 torch.tensor(obj in rows), torch.tensor(expected))
            if obj in rows and expected:
                row = rows[obj]
                same(f"end to end: frame {index} object {obj} mask",
                     torch.from_numpy(np.asarray(row.masks[0])), mask)
                # PixelFlow stores a confidence rounded to three decimals; compare through it.
                stored = pf.Detection(bbox=[0, 0, 1, 1], confidence=float(torch.sigmoid(score)))
                same(f"end to end: frame {index} object {obj} confidence",
                     torch.tensor(row.confidence), torch.tensor(stored.confidence))
                same(f"end to end: frame {index} object {obj} ids and name",
                     torch.tensor([row.class_id, row.tracker_id, row.class_name == label]),
                     torch.tensor([obj, obj, True]))
    print("  end to end (mozo.get_model -> tracker -> PixelFlow): done")


# --- falsification -------------------------------------------------------------------------

@contextlib.contextmanager
def perturbed(name: str):
    """Apply one :data:`FALSIFIERS` perturbation to the vendored side, and undo it after."""
    import torch.nn.functional as F

    from mozo.vendors.sam2_deploy import image as image_module
    from mozo.vendors.sam2_deploy import tracking

    undo = []

    def patch(owner, attribute, value):
        undo.append((owner, attribute, owner.__dict__.get(attribute, _MISSING)))
        setattr(owner, attribute, value)

    if name == "image-path-resize":
        patch(vendor_video, "load_frame", lambda image, size: image_module.preprocess(image, size))
    elif name == "float32-divide":

        def float32(image, size):
            resized = np.array(Image.fromarray(image).resize((size, size)))
            frame = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
            mean = torch.tensor(image_module.MEAN)[:, None, None]
            std = torch.tensor(image_module.STD)[:, None, None]
            return ((frame - mean) / std).unsqueeze(0)
        patch(vendor_video, "load_frame", float32)
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


_MISSING = object()


def falsify(upstream: Path, variant: str, checkpoint: Path, frames: np.ndarray,
            names: list[str] | None = None) -> int:
    """Run the gate under each perturbation (or the *names* given); every one must fail."""
    caught, missed = [], []
    for name, what in FALSIFIERS.items():
        if names and name not in names:
            continue
        same = Comparison(quiet=True, stop=True)
        with perturbed(name), contextlib.redirect_stdout(None), contextlib.suppress(Caught):
            check_variant(upstream, variant, checkpoint, frames, same, end_to_end=False)
        (caught if same.failed else missed).append(name)
        first = same.failed[0] if same.failed else "-"
        print(f"  {'caught' if same.failed else 'MISSED':<7} {name:<22} "
              f"after {same.checked} comparisons; first failure: {first}", flush=True)
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
    parser.add_argument("--frames", type=int, default=20,
                        help="frames of the bedroom clip to track (default 20, at least 16)")
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
    if arguments.checkpoint is not None and len(arguments.variants) != 1:
        raise SystemExit("--checkpoint checks one variant; name it")
    if arguments.frames < 16:
        raise SystemExit("--frames must be at least 16 (the short-clip case covers fewer)")

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
    print(f"upstream   {upstream} @ {head[:12]}")
    print(f"versions   torch {torch.__version__}, Pillow {PIL.__version__}, numpy {np.__version__}; "
          f"device cpu, {torch.get_num_threads()} threads, MKL_CBWR={os.environ.get('MKL_CBWR')}")
    frames = clip_frames(upstream, arguments.frames)
    checkpoints = {v: (arguments.checkpoint or published(v)).resolve() for v in arguments.variants}

    if arguments.falsify:
        variant = arguments.variants[0]
        return falsify(upstream, variant, checkpoints[variant], frames, arguments.falsifier)

    same = Comparison()
    for variant in arguments.variants:
        check_variant(upstream, variant, checkpoints[variant], frames, same)
    print()
    if same.failed:
        print(f"{len(same.failed)} of {same.checked} comparisons differ.\nSAM 2 video gate: FAIL")
        return 1
    print(f"all {same.checked} comparisons bit-identical.\nSAM 2 video gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

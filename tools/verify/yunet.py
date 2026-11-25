#!/usr/bin/env python3
"""Verify the vendored YuNet against the authors' PyTorch, stage by stage, exactly.

    MKL_CBWR=AVX2,STRICT python -I tools/verify/yunet.py \\
        --upstream /path/to/libfacedetection.train --wider /path/to/WIDER_val/images [--mozo]

The reference is ShiqiYu/libfacedetection.train at commit ``dca340aa`` (BSD-3-Clause), pure
PyTorch since its 2026 refactor: nothing to install beyond what mozo already needs, the checkout
only has to be on ``sys.path``, which this script does. Three paths are compared on CPU, with
``torch.equal`` / ``np.array_equal`` and no tolerance:

1. **upstream** -- ``build_yunet`` with a strict load of the *original* checkpoint file from the
   checkout's ``weights/``, upstream's evaluation transforms in "origin" mode (pad to a multiple
   of 32, no resize) on the BGR image, ``collate_face_samples``, and ``YuNetPostprocessor`` -- the
   pipeline its WIDER Face evaluation runs, not a re-creation of it.
2. **vendor** -- :class:`mozo.vendors.yunet_deploy.Predictor` on the republished state dict.
3. **mozo** -- ``YuNetPredictor`` through the registry, the weights tree and PixelFlow (``--mozo``),
   against upstream's arrays handed through the same PixelFlow conversion.

Stages compared per image: the float batch, **every named module's output** (the names are
upstream's, so the two models are walked side by side with forward hooks), and the final boxes,
scores and landmarks, at upstream's evaluation threshold of 0.02 and at mozo's default 0.5.

**Set ``MKL_CBWR=AVX2,STRICT``.** MKL picks kernels by buffer alignment, so two CPU torch
processes can differ in the last bit without it (VENDORING §5).

``--falsify NAME`` perturbs one thing in the vendor and expects the gate to fail at the stage
named; ``--falsify list`` names them, ``--falsify all`` runs every one (each in a fresh process)
and fails unless each was caught.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.stdout.reconfigure(line_buffering=True)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import fixtures as photographs  # noqa: E402
from tools.fetch.yunet import CHECKPOINTS, COMMIT  # noqa: E402

THRESHOLDS = (0.02, 0.5)

#: Perturbations of the vendor, each expected to be caught at the stage named.
FALSIFIERS = {
    "rgb": "channels left RGB instead of reversed to BGR (caught at the batch)",
    "pad": "padding 1 instead of 0 (caught at the batch, padded images only)",
    "bn_eps": "stem BN eps 1e-5 -> 1e-3 (caught at backbone.model0.bn1)",
    "upsample": "neck upsampling nearest -> bilinear (caught at the neck)",
    "exp": "box size as exp(offset + ln stride) (caught at the boxes only; network exact)",
    "sqrt": "OpenCV's score, sqrt(cls * obj) (caught at the scores; boxes of t=0.02 rows too)",
    "nms": "NMS IoU 0.45 -> 0.4 (caught at the final rows only)",
    "centre": "priors at cell centres instead of corners (caught at boxes and landmarks)",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixtures(wider: Path | None, count: int, seed: int = 0) -> list[tuple[str, np.ndarray]]:
    """BGR images: the repo's photographs, awkward synthetic cases, and random WIDER Face val."""
    frames = [(p.name, cv2.imread(str(p))) for p in photographs()]
    rng = np.random.default_rng(seed)
    example = frames[0][1]
    frames += [
        ("blank-640x480", np.full((480, 640, 3), 127, np.uint8)),
        ("noise-333x517", rng.integers(0, 256, (333, 517, 3), dtype=np.uint8)),
        ("tiny-37x23", cv2.resize(example, (37, 23))),
        ("one-pixel", example[:1, :1].copy()),
        ("odd-641x479", cv2.resize(example, (641, 479))),
        ("panorama-2400x300", cv2.resize(example, (2400, 300))),
        ("huge-3000x2000", cv2.resize(example, (3000, 2000))),
        # Faces cut by the right and top edges: boxes that run past the image, which upstream
        # does not clip.
        ("cut-900x1281", np.ascontiguousarray(example[:, :900])),
        ("cut-top", np.ascontiguousarray(example[300:])),
        ("grey", cv2.cvtColor(cv2.cvtColor(example, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)),
        ("small-faces-480x320", cv2.resize(example, (480, 320))),
    ]
    if wider is not None and count:
        names = sorted(glob.glob(str(wider / "*" / "*.jpg")))
        for path in random.Random(seed).sample(names, count):
            frames.append((Path(path).name, cv2.imread(path)))
    return frames


def capture(model: torch.nn.Module, names: set[str]) -> tuple[dict, list]:
    """Record the output of every module named in *names* (first call each), via forward hooks."""
    seen: dict[str, object] = {}
    handles = []
    for name, module in model.named_modules():
        if not name or name not in names:
            continue

        def hook(_module, _inputs, output, name=name):
            if name not in seen:
                seen[name] = tensors(output)
        handles.append(module.register_forward_hook(hook))
    return seen, handles


def tensors(value) -> list[torch.Tensor]:
    """Every tensor in *value*, cloned: ReLU runs in place, so a hooked output can be overwritten."""
    if torch.is_tensor(value):
        return [value.detach().clone()]
    if isinstance(value, (list, tuple)):
        return [t for item in value for t in tensors(item)]
    return []


def equal(a, b) -> bool:
    if torch.is_tensor(a):
        return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)


def delta(a, b) -> str:
    try:
        a = a.detach().double().cpu().numpy() if torch.is_tensor(a) else np.asarray(a, np.float64)
        b = b.detach().double().cpu().numpy() if torch.is_tensor(b) else np.asarray(b, np.float64)
        if a.shape != b.shape:
            return f"shape {a.shape} vs {b.shape}"
        return f"max|d| {np.abs(a - b).max() if a.size else 0:.3g}"
    except Exception as error:  # noqa: BLE001 - a report, not a control path
        return repr(error)


class Gate:
    def __init__(self):
        self.passed = 0
        self.failed: list[str] = []

    def check(self, stage: str, a, b) -> bool:
        return self.record(stage, equal(a, b), lambda: delta(a, b))

    def record(self, stage: str, ok: bool, detail=lambda: "differ") -> bool:
        if ok:
            self.passed += 1
        else:
            self.failed.append(f"{stage}: {detail()}")
        return ok


def falsify(name: str, vendor) -> None:
    """Apply one perturbation to the vendor (in this process only)."""
    # Through ``sys.modules``: the package re-exports names that shadow its submodules.
    network = sys.modules["mozo.vendors.yunet_deploy.network"]
    postprocess = sys.modules["mozo.vendors.yunet_deploy.postprocess"]
    predictor = sys.modules["mozo.vendors.yunet_deploy.predictor"]
    preprocess = sys.modules["mozo.vendors.yunet_deploy.preprocess"]
    if name == "rgb":
        original = preprocess.to_batch
        predictor.to_batch = lambda image, *a: original(np.ascontiguousarray(image[..., ::-1]), *a)
    elif name == "pad":
        original = preprocess.to_batch

        def to_batch(image, divisor, device):
            batch = original(image, divisor, device)
            height, width = image.shape[:2]
            batch[..., height:, :] = 1
            batch[..., :, width:] = 1
            return batch
        predictor.to_batch = to_batch
    elif name == "bn_eps":
        vendor.model.backbone.model0.bn1.eps = 1e-3
    elif name == "upsample":
        interpolate = network.F.interpolate

        class _F:
            def __getattr__(self, attr):
                return getattr(torch.nn.functional, attr)

            @staticmethod
            def interpolate(x, scale_factor, mode):
                return interpolate(x, scale_factor=scale_factor, mode="bilinear")
        network.F = _F()
    elif name == "centre":
        grid = postprocess.grid_priors

        def centred(*args):
            priors = grid(*args)
            return priors + torch.cat([priors[:, 2:] / 2, torch.zeros_like(priors[:, 2:])], 1)
        postprocess.grid_priors = centred
    elif name in ("exp", "sqrt"):
        predictor.decode = lambda *args: _opencv_decode(*args, which=name)
    elif name == "nms":
        import dataclasses
        vendor.spec = dataclasses.replace(vendor.spec, nms_threshold=0.4)
    else:
        raise SystemExit(f"unknown falsifier {name!r}; known: {sorted(FALSIFIERS)}")


def _opencv_decode(outputs, strides, threshold, iou_threshold, *, which):
    """The vendor's decode, written out, with one rewrite swapped in.

    OpenCV's ``(col + offset) * stride`` is not one of them: it rounds exactly as upstream's
    ``offset * stride + col * stride``, because every stride is a power of two and scaling by one
    is exact. Measured: no comparison moved.
    """
    from mozo.vendors.yunet_deploy import postprocess as pp

    cls_preds, bbox_preds, obj_preds, kps_preds = outputs
    priors = pp.grid_priors([tuple(c.shape[2:]) for c in cls_preds], strides, torch.float32,
                            torch.device("cpu"))
    cls, bbox = pp._flatten(cls_preds).sigmoid(), pp._flatten(bbox_preds)
    obj, kps = pp._flatten(obj_preds).squeeze(-1).sigmoid(), pp._flatten(kps_preds)
    stride = priors[:, 2:]
    xys = bbox[:, :2] * stride + priors[:, :2]
    points = torch.cat([kps[:, 2 * i:2 * i + 2] * stride + priors[:, :2] for i in range(5)], -1)
    whs = (bbox[:, 2:] + stride.log()).exp() if which == "exp" else bbox[:, 2:].exp() * stride
    boxes = torch.cat([xys - whs / 2, xys + whs / 2], -1)
    scores = cls[:, 0] * obj
    if which == "sqrt":
        scores = scores.sqrt()
    keep = scores >= threshold
    boxes, scores, points = boxes[keep], scores[keep], points[keep]
    kept = pp.nms(boxes, scores, iou_threshold)
    if kept.numel():
        kept = kept[scores[kept].sort(descending=True)[1]]
    return boxes[kept], scores[kept], points[kept]


def upstream_pipeline(upstream: Path):
    """Upstream's model builder, transforms, collate and postprocessor, from the checkout."""
    sys.path.insert(0, str(upstream))
    from yunet_train.tasks.face import (YuNetPostprocessor, build_eval_transforms, build_yunet,
                                        collate_face_samples)
    from yunet_train.tasks.face.types import FaceSample

    transform = build_eval_transforms(image_size=None, size_divisor=32)

    def batch_of(bgr: np.ndarray) -> torch.Tensor:
        empty4 = np.zeros((0, 4), np.float32)
        sample = FaceSample(image=bgr.copy(), boxes=empty4, labels=np.zeros(0, np.int64),
                            keypoints=np.zeros((0, 5, 3), np.float32), ignored_boxes=empty4,
                            ignored_labels=np.zeros(0, np.int64), filename="x",
                            original_shape=bgr.shape, image_shape=bgr.shape, pad_shape=bgr.shape)
        return collate_face_samples([transform(sample)]).images

    return build_yunet, batch_of, YuNetPostprocessor


def verify_variant(variant: str, args, checkpoint: Path, mozo, frames, gate: Gate) -> None:
    from mozo.vendors.yunet_deploy import Predictor

    build_yunet, batch_of, YuNetPostprocessor = upstream_pipeline(args.upstream)
    filename, _, pinned, _ = CHECKPOINTS[variant]
    original = args.upstream / "weights" / filename
    if sha256(original) != pinned:
        raise SystemExit(f"{original} is not the pinned upstream file (sha256 {pinned})")
    upstream = build_yunet(f"yunet_{filename[6]}")
    state = torch.load(original, map_location="cpu", weights_only=True)["state_dict"]
    upstream.load_state_dict(state, strict=True)
    upstream.eval()

    vendor = Predictor(checkpoint, variant, "cpu")
    if args.falsify:
        falsify(args.falsify, vendor)

    shared = ({n for n, _ in upstream.named_modules()} & {n for n, _ in vendor.model.named_modules()})
    compared = 0
    for label, bgr in frames:
        rgb = np.ascontiguousarray(bgr[..., ::-1])
        batch_up = batch_of(bgr)
        batch_v = sys.modules["mozo.vendors.yunet_deploy.predictor"].to_batch(
            rgb, vendor.spec.size_divisor, vendor.device)
        gate.check(f"{variant}/{label}/batch", batch_up, batch_v)

        up_seen, up_handles = capture(upstream, shared)
        v_seen, v_handles = capture(vendor.model, shared)
        try:
            outputs_up = upstream(batch_up)
            vendor.predict(rgb, THRESHOLDS[0])
        finally:
            for handle in up_handles + v_handles:
                handle.remove()
        # In upstream's execution order, so the first difference listed is the earliest.
        for name in [n for n in up_seen if n in v_seen]:
            a, b = up_seen[name], v_seen[name]
            if not gate.record(f"{variant}/{label}/module {name}", len(a) == len(b),
                               lambda: f"{len(a)} vs {len(b)} output tensors"):
                continue
            for index, (x, y) in enumerate(zip(a, b)):
                compared += 1
                gate.check(f"{variant}/{label}/module {name}[{index}]", x, y)

        for threshold in THRESHOLDS:
            result = YuNetPostprocessor(score_threshold=threshold)(outputs_up)[0]
            faces = vendor.predict(rgb, threshold)
            stage = f"{variant}/{label}/t={threshold}"
            gate.check(f"{stage}/boxes", result.boxes.numpy(), faces.boxes)
            gate.check(f"{stage}/scores", result.scores.numpy(), faces.scores)
            gate.check(f"{stage}/landmarks", result.keypoints.numpy(),
                       faces.keypoints.reshape(-1, 10))
            if args.verbose:
                print(f"  {stage}: {len(faces.scores)} faces")
            if mozo is not None:
                expected = mozo.detections(result.boxes.numpy(), result.scores.numpy(),
                                           result.keypoints.numpy().reshape(-1, 5, 2),
                                           mozo.models[variant]._labels).to_dict()
                actual = mozo.models[variant].predict(rgb, threshold=threshold).to_dict()
                gate.check(f"{stage}/mozo rows", np.array(len(expected)), np.array(len(actual)))
                gate.record(f"{stage}/mozo", expected == actual)

    print(f"  {variant}: {len(shared)} shared module names, {compared} module outputs compared "
          f"over {len(frames)} images")


def falsify_all(argv: list[str]) -> int:
    """Run every falsifier in a process of its own and report which were caught where."""
    base = [a for a in argv if not a.startswith("--falsify") and a != "all"]
    missed = []
    for name in FALSIFIERS:
        run = subprocess.run([sys.executable, "-I", __file__, *base, "--falsify", name],
                             capture_output=True, text=True)
        verdict = [line for line in run.stdout.splitlines() if line.startswith("falsifier")]
        print(verdict[-1] if verdict else f"falsifier {name!r}: no verdict\n{run.stderr[-2000:]}")
        if run.returncode != 0:
            missed.append(name)
    print(f"\n{len(FALSIFIERS) - len(missed)} of {len(FALSIFIERS)} falsifiers caught")
    return 1 if missed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--upstream", type=Path, required=True,
                        help=f"libfacedetection.train checkout at {COMMIT[:8]}")
    parser.add_argument("--repacked", type=Path, default=None,
                        help="republished state dicts as <variant>.pth (default: weights/ tree)")
    parser.add_argument("--wider", type=Path, default=None, help="WIDER Face val images")
    parser.add_argument("--count", type=int, default=24, help="random WIDER val images (seed 0)")
    parser.add_argument("--variants", nargs="*", default=list(CHECKPOINTS))
    parser.add_argument("--falsify", default=None)
    parser.add_argument("--mozo", action="store_true", help="also check YuNetPredictor")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.falsify == "list":
        for name, what in FALSIFIERS.items():
            print(f"{name:10s} {what}")
        return 0
    if args.falsify == "all":
        return falsify_all(sys.argv[1:])
    if os.environ.get("MKL_CBWR") != "AVX2,STRICT":
        print("warning: MKL_CBWR is not AVX2,STRICT; CPU results may differ between processes")

    head = (args.upstream / ".git" / "HEAD")
    commit = subprocess.run(["git", "-C", str(args.upstream), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip() if head.exists() else "?"
    print(f"torch {torch.__version__}, cv2 {cv2.__version__}, numpy {np.__version__}, "
          f"upstream {commit[:8] or '?'} (pinned {COMMIT[:8]})")
    if commit not in ("?", COMMIT):
        print("warning: the upstream checkout is not at the pinned commit")
    torch.set_grad_enabled(False)

    frames = fixtures(args.wider, args.count)
    mozo = None
    if args.mozo:
        import types

        from mozo.adapters.yunet import YuNetPredictor, detections
        mozo = types.SimpleNamespace(
            detections=detections,
            models={v: YuNetPredictor(v, device="cpu", runtime="torch-fp32")
                    for v in args.variants})

    def checkpoint_for(variant: str) -> Path:
        if args.repacked is not None:
            return args.repacked / f"{variant}.pth"
        from mozo.weights import resolve
        return resolve("yunet", variant, "torch-fp32")

    gate = Gate()
    for variant in args.variants:
        verify_variant(variant, args, checkpoint_for(variant), mozo, frames, gate)

    print(f"\n{gate.passed} comparisons exact, {len(gate.failed)} differ")
    for line in gate.failed[:40]:
        print(f"  DIFF {line}")
    if args.falsify:
        first = gate.failed[0].split(": ")[0] if gate.failed else "-"
        print(f"falsifier {args.falsify!r}: {'CAUGHT' if gate.failed else 'NOT CAUGHT'}; "
              f"{len(gate.failed)} differ, first at {first}")
        return 0 if gate.failed else 1
    return 1 if gate.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

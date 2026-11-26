#!/usr/bin/env python3
"""Verify the vendored YuNet against the authors' PyTorch, stage by stage, exactly.

    python -I tools/verify/yunet.py --device cuda \\
        --upstream /path/to/libfacedetection.train --wider /path/to/WIDER_val/images [--mozo]
    python -I tools/verify/yunet.py --device cuda ... --falsify all
    MKL_CBWR=AVX2,STRICT python -I tools/verify/yunet.py ...      # the CPU gate (the default)

The reference is ShiqiYu/libfacedetection.train at commit ``dca340aa`` (BSD-3-Clause), pure
PyTorch since its 2026 refactor: nothing to install beyond what mozo already needs, the checkout
only has to be on ``sys.path``, which this script does. Three paths are compared in one process
on one device, with ``torch.equal`` / ``np.array_equal`` and no tolerance:

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

``--device cuda`` is the primary gate: upstream, the vendor and mozo run on the same GPU, in one
process, with torch's default CUDA settings -- the ones mozo's runtime leaves alone (cuDNN may use
TF32 for convolutions; ``cudnn.benchmark`` and deterministic algorithms are off). Upstream runs as
its WIDER Face evaluation does on CUDA: the float batch is made on the host and moved to the
device, and the network, decode and NMS run there. The vendor makes its batch on the device; every
value is an integer 0..255 either way. Every stage is copied to the host, which is exact, and
compared there. Each side first runs every image twice and must repeat itself bit for bit, or the
comparison would mean nothing (VENDORING §2.7).

``--dtype`` is fp32 only. YuNet's CUDA runtimes through torch are ``torch-fp32`` alone (its
``auto`` on CUDA is a TensorRT fp32 engine, measured by parity and accuracy instead), mozo
publishes no torch half runtime for it, and upstream has no half-precision inference path.

**On CPU, set ``MKL_CBWR=AVX2,STRICT``.** MKL picks kernels by buffer alignment, so two CPU torch
processes can differ in the last bit without it (VENDORING §5).

``--falsify NAME`` perturbs one thing in the vendor and expects the gate to fail at the stage
named; ``--falsify list`` names them, ``--falsify all`` runs every one (each in a fresh process)
and fails unless each was caught. On CUDA two falsifiers only a GPU has are added (see
:data:`FALSIFIERS_CUDA`), and ``all`` first computes upstream's stages once, unperturbed, into a
reference cache (``--reference-cache``) that every falsifier reuses: upstream never changes
between them.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.stdout.reconfigure(line_buffering=True)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import digest, fixtures as photographs  # noqa: E402
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

#: Perturbations only a GPU run can catch, added on ``--device cuda``. A falsifier that makes the
#: vendor's batch on the host, as upstream does, is not among them: the batch holds integers
#: 0..255, exact on either side, so it is the same arithmetic and could never be caught.
FALSIFIERS_CUDA = {
    "tf32-off": "cuDNN TF32 off on the vendor's side alone (caught at the first conv that uses it)",
    "channels-last": "the batch in channels-last memory, values equal (caught at the network: "
                     "cuDNN picks other kernels)",
}


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
    """Every tensor in *value*, copied to the host (exact): ReLU runs in place, so a hooked output
    can be overwritten, and the stages are compared, and cached, on the CPU."""
    if torch.is_tensor(value):
        return [value.detach().to("cpu", copy=True)]
    if isinstance(value, (list, tuple)):
        return [t for item in value for t in tensors(item)]
    return []


def fingerprint(tensor: torch.Tensor) -> tuple:
    """A host tensor's shape, dtype and the sha256 of its bytes: what the reference cache keeps
    of each intermediate. The full tensors of 72 images run to tens of GB (one 3000x2000 image's
    module outputs alone are about 3 GB); the final rows are kept whole."""
    return (tuple(tensor.shape), str(tensor.dtype),
            hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest())


def equal(a, b) -> bool:
    if isinstance(a, tuple):  # a cached fingerprint: the bytes must be the same
        return a == fingerprint(b)
    if torch.is_tensor(a):
        return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)


def delta(a, b) -> str:
    if isinstance(a, tuple):
        return f"bytes differ from the cached reference ({a[0]} {a[1]} vs {tuple(b.shape)} {b.dtype})"
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
    elif name == "tf32-off":
        # Around the vendor's forward only, so an upstream run in this process keeps the default.
        forward = vendor.model.forward

        def without_tf32(*args, **kwargs):
            torch.backends.cudnn.allow_tf32 = False
            try:
                return forward(*args, **kwargs)
            finally:
                torch.backends.cudnn.allow_tf32 = True
        vendor.model.forward = without_tf32
    elif name == "channels-last":
        original = preprocess.to_batch
        predictor.to_batch = lambda *a: original(*a).contiguous(memory_format=torch.channels_last)
    else:
        raise SystemExit(f"unknown falsifier {name!r}; known: "
                         f"{sorted(FALSIFIERS) + sorted(FALSIFIERS_CUDA)}")


def _opencv_decode(outputs, strides, threshold, iou_threshold, *, which):
    """The vendor's decode, written out, with one rewrite swapped in.

    OpenCV's ``(col + offset) * stride`` is not one of them: it rounds exactly as upstream's
    ``offset * stride + col * stride``, because every stride is a power of two and scaling by one
    is exact. Measured: no comparison moved.
    """
    from mozo.vendors.yunet_deploy import postprocess as pp

    cls_preds, bbox_preds, obj_preds, kps_preds = outputs
    priors = pp.grid_priors([tuple(c.shape[2:]) for c in cls_preds], strides, torch.float32,
                            cls_preds[0].device)
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


def run_upstream(upstream, postprocessor, batch_of, bgr: np.ndarray, shared: set[str],
                 device: torch.device) -> dict:
    """Upstream's stages for one image, on the host: the batch, every shared module's output in
    execution order, and the final rows at each threshold."""
    batch = batch_of(bgr).to(device)
    seen, handles = capture(upstream, shared)
    try:
        outputs = upstream(batch)
    finally:
        for handle in handles:
            handle.remove()
    rows = {}
    for threshold in THRESHOLDS:
        result = postprocessor(score_threshold=threshold)(outputs)[0]
        rows[threshold] = (result.boxes.cpu().numpy(), result.scores.cpu().numpy(),
                           result.keypoints.cpu().numpy())
    return {"batch": tensors(batch)[0], "modules": seen, "rows": rows}


def run_vendor(vendor, rgb: np.ndarray, shared: set[str]) -> dict:
    """The vendor's stages for one image, in the same shape as :func:`run_upstream`'s."""
    batch = sys.modules["mozo.vendors.yunet_deploy.predictor"].to_batch(
        rgb, vendor.spec.size_divisor, vendor.device)
    seen, handles = capture(vendor.model, shared)
    try:
        vendor.predict(rgb, THRESHOLDS[0])
    finally:
        for handle in handles:
            handle.remove()
    rows = {}
    for threshold in THRESHOLDS:
        faces = vendor.predict(rgb, threshold)
        rows[threshold] = (faces.boxes, faces.scores, faces.keypoints.reshape(-1, 10))
    return {"batch": tensors(batch)[0], "modules": seen, "rows": rows}


def repeats(a: dict, b: dict) -> bool:
    """Whether two runs of one side gave the same bytes at every stage."""
    return (equal(a["batch"], b["batch"]) and a["modules"].keys() == b["modules"].keys()
            and all(len(x) == len(y) and all(equal(u, v) for u, v in zip(x, y))
                    for x, y in zip(a["modules"].values(), b["modules"].values()))
            and all(equal(u, v) for t in THRESHOLDS for u, v in zip(a["rows"][t], b["rows"][t])))


def verify_variant(variant: str, args, pipeline, checkpoint: Path, mozo, frames, gate: Gate,
                   cached: dict | None, saved: dict) -> None:
    from mozo.vendors.yunet_deploy import Predictor, get_spec

    device = torch.device(args.device)
    vendor = Predictor(checkpoint, variant, device)
    upstream = shared = None
    if cached is None:
        build_yunet, batch_of, YuNetPostprocessor = pipeline
        filename, _, pinned, _ = CHECKPOINTS[variant]
        original = args.upstream / "weights" / filename
        if digest(original) != pinned:
            raise SystemExit(f"{original} is not the pinned upstream file (sha256 {pinned})")
        upstream = build_yunet(get_spec(variant).upstream)
        state = torch.load(original, map_location="cpu", weights_only=True)["state_dict"]
        upstream.load_state_dict(state, strict=True)
        upstream.eval().to(device)
        shared = {n for n, _ in upstream.named_modules()} & {n for n, _ in vendor.model.named_modules()}
    if args.falsify:
        falsify(args.falsify, vendor)

    compared = 0
    check_repeats = device.type == "cuda" and not args.falsify
    for label, bgr in frames:
        rgb = np.ascontiguousarray(bgr[..., ::-1])
        if cached is not None:
            up = cached[variant, label]
            names = set(up["modules"])
        else:
            up = run_upstream(upstream, YuNetPostprocessor, batch_of, bgr, shared, device)
            if check_repeats:
                gate.record(f"{variant}/{label}/upstream repeats itself",
                            repeats(up, run_upstream(upstream, YuNetPostprocessor, batch_of, bgr,
                                                     shared, device)))
            if args.reference_cache is not None:
                saved[variant, label] = {
                    "batch": fingerprint(up["batch"]), "rows": up["rows"],
                    "modules": {n: [fingerprint(t) for t in ts] for n, ts in up["modules"].items()}}
            names = shared
        mine = run_vendor(vendor, rgb, names)
        if check_repeats:
            gate.record(f"{variant}/{label}/vendor repeats itself",
                        repeats(mine, run_vendor(vendor, rgb, names)))

        gate.check(f"{variant}/{label}/batch", up["batch"], mine["batch"])
        # In upstream's execution order, so the first difference listed is the earliest.
        for name in [n for n in up["modules"] if n in mine["modules"]]:
            a, b = up["modules"][name], mine["modules"][name]
            if not gate.record(f"{variant}/{label}/module {name}", len(a) == len(b),
                               lambda: f"{len(a)} vs {len(b)} output tensors"):
                continue
            for index, (x, y) in enumerate(zip(a, b)):
                compared += 1
                gate.check(f"{variant}/{label}/module {name}[{index}]", x, y)

        for threshold in THRESHOLDS:
            boxes, scores, landmarks = up["rows"][threshold]
            v_boxes, v_scores, v_landmarks = mine["rows"][threshold]
            stage = f"{variant}/{label}/t={threshold}"
            gate.check(f"{stage}/boxes", boxes, v_boxes)
            gate.check(f"{stage}/scores", scores, v_scores)
            gate.check(f"{stage}/landmarks", landmarks, v_landmarks)
            if args.verbose:
                print(f"  {stage}: {len(v_scores)} faces")
            if mozo is not None:
                expected = mozo.detections(boxes, scores, landmarks.reshape(-1, 5, 2),
                                           mozo.models[variant]._labels).to_dict()
                actual = mozo.models[variant].predict(rgb, threshold=threshold).to_dict()
                gate.check(f"{stage}/mozo rows", np.array(len(expected)), np.array(len(actual)))
                gate.record(f"{stage}/mozo", expected == actual)

    print(f"  {variant}: {len(names)} shared module names, {compared} module outputs compared "
          f"over {len(frames)} images" + (", each side run twice and repeated itself"
                                          if check_repeats else ""))


def environment(device: torch.device) -> str:
    """torch, CUDA, cuDNN, the GPU, its driver and the settings in force: what a result holds for."""
    if device.type == "cpu":
        return f"torch {torch.__version__}, CPU, MKL_CBWR={os.environ.get('MKL_CBWR', '')}"
    driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader",
                             "-i", str(device.index or 0)], capture_output=True, text=True)
    return (f"torch {torch.__version__}, CUDA {torch.version.cuda}, cuDNN "
            f"{torch.backends.cudnn.version()}, {torch.cuda.get_device_name(device)}, driver "
            f"{driver.stdout.strip() or 'unknown'}; cudnn.allow_tf32 "
            f"{torch.backends.cudnn.allow_tf32}, matmul.allow_tf32 "
            f"{torch.backends.cuda.matmul.allow_tf32}, cudnn.benchmark "
            f"{torch.backends.cudnn.benchmark}, deterministic "
            f"{torch.are_deterministic_algorithms_enabled()}")


def falsify_all(argv: list[str], names: list[str], device: str) -> int:
    """Run every falsifier in a process of its own and report which were caught where.

    On CUDA, upstream's stages are first computed once, by an unperturbed run that must itself
    pass, into a reference cache every falsifier then reads: upstream never changes between them.
    A second unperturbed run must then pass against the cache, so a falsifier's failure is the
    perturbation and not the cache.
    """
    base = [a for a in argv if not a.startswith("--falsify") and a != "all"]
    with tempfile.TemporaryDirectory() as scratch:
        if device == "cuda" and not any(a.startswith("--reference-cache") for a in base):
            base += ["--reference-cache", str(Path(scratch) / "reference.pt")]
            for what in ("reference", "against the cache"):
                run = subprocess.run([sys.executable, "-I", __file__, *base], capture_output=True,
                                     text=True)
                lines = run.stdout.strip().splitlines() or [run.stderr[-2000:]]
                print(f"unperturbed, {what}: {lines[-1].strip()}")
                if run.returncode != 0:
                    print(run.stdout[-3000:], run.stderr[-3000:])
                    return 1
        missed = []
        for name in names:
            run = subprocess.run([sys.executable, "-I", __file__, *base, "--falsify", name],
                                 capture_output=True, text=True)
            verdict = [line for line in run.stdout.splitlines() if line.startswith("falsifier")]
            print(verdict[-1] if verdict else f"falsifier {name!r}: no verdict\n{run.stderr[-2000:]}")
            if run.returncode != 0:
                missed.append(name)
    print(f"\n{len(names) - len(missed)} of {len(names)} falsifiers caught")
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
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--falsify", default=None)
    parser.add_argument("--reference-cache", type=Path, default=None,
                        help="upstream's stages: read from this file, or computed and saved there")
    parser.add_argument("--mozo", action="store_true", help="also check YuNetPredictor")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.dtype != "fp32":
        parser.error("YuNet runs through torch in fp32 only: mozo publishes no torch half runtime "
                     "for it and upstream has no half-precision inference path")
    falsifiers = {**FALSIFIERS, **(FALSIFIERS_CUDA if args.device == "cuda" else {})}
    if args.falsify == "list":
        for name, what in falsifiers.items():
            print(f"{name:14s} {what}")
        return 0
    if args.falsify == "all":
        return falsify_all(sys.argv[1:], list(falsifiers), args.device)
    if args.falsify is not None and args.falsify not in falsifiers:
        parser.error(f"--falsify {args.falsify} does not apply on {args.device}; "
                     f"known: {', '.join(falsifiers)}")
    if args.device == "cpu" and os.environ.get("MKL_CBWR") != "AVX2,STRICT":
        print("warning: MKL_CBWR is not AVX2,STRICT; CPU results may differ between processes")

    cached = None
    if args.reference_cache is not None and args.reference_cache.exists():
        cached = torch.load(args.reference_cache, weights_only=False)
        if cached.pop("for") != (args.device, torch.__version__, args.count):
            raise SystemExit(f"{args.reference_cache} was made for another device, torch or count")

    commit = ((subprocess.run(["git", "-C", str(args.upstream), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "?")
              if (args.upstream / ".git").exists() else "?")
    device = torch.device(args.device)
    print(f"{environment(device)}; cv2 {cv2.__version__}, numpy {np.__version__}, upstream "
          f"{commit[:8]} (pinned {COMMIT[:8]})" + (", reference cached" if cached else ""))
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
            models={v: YuNetPredictor(v, device=args.device, runtime="torch-fp32")
                    for v in args.variants})

    def checkpoint_for(variant: str) -> Path:
        if args.repacked is not None:
            return args.repacked / f"{variant}.pth"
        from mozo.weights import resolve
        return resolve("yunet", variant, "torch-fp32")

    gate = Gate()
    pipeline = upstream_pipeline(args.upstream) if cached is None else None
    saved: dict = {}
    for variant in args.variants:
        verify_variant(variant, args, pipeline, checkpoint_for(variant), mozo, frames, gate,
                       cached, saved)
    if args.reference_cache is not None and cached is None and not args.falsify and not gate.failed:
        torch.save({**saved, "for": (args.device, torch.__version__, args.count)},
                   args.reference_cache)

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

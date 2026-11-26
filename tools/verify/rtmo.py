#!/usr/bin/env python3
"""Verify the vendored RTMO against MMPose, stage by stage, exactly.

    python -I tools/verify/rtmo.py --device cuda --mmpose /path/to/mmpose-v1.3.2 \\
        --checkpoints /path/to/upstream/pths --coco /path/to/val2017 [--mozo] [--variants small ...]
    python -I tools/verify/rtmo.py --device cuda ... --falsify all
    MKL_CBWR=AVX2,STRICT python -I tools/verify/rtmo.py ...      # the CPU gate (the default)

Runs in a scratch environment that has MMPose v1.3.2 installed (``mmengine``, ``mmcv-lite``,
``mmdet --no-deps``, ``mmpose --no-deps``); none of that is a mozo dependency. Three paths are
compared in one process on one device, with ``torch.equal`` / ``np.array_equal`` and no tolerance:

1. **upstream** -- ``mmpose.apis.init_model`` on the *original* checkpoint file, then
   ``inference_bottomup`` on the BGR frame: MMPose's own pipeline, not a re-creation of it.
2. **vendor** -- :class:`mozo.vendors.rtmo_deploy.Predictor` on the republished state dict.
3. **mozo** -- ``RTMOPredictor`` through the registry, the weights tree and PixelFlow, when the
   variant has been fetched into ``weights/`` (``MOZO_BASE_URL`` / the local tree decides).

Stages compared per image: the letterboxed ``uint8`` square, its centre and scale, the float batch,
**every named module's output** (the names are upstream's, so the two models are walked side by side
with forward hooks -- 400 to 700 modules per variant), the DCC's inputs and output, and the final
boxes, scores, joints and joint scores, at upstream's ``score_thr`` of 0.1 and at mozo's default 0.5.

**mmcv's compiled ops are stubbed.** ``mmpose.models`` imports EDPose, which imports
``mmcv.ops.MultiScaleDeformableAttention`` at module level, and ``mmcv-lite`` has no compiled ops.
RTMO never calls ``mmcv.ops``; the per-module comparison is what shows the stub reached nothing.

``--device cuda`` is the primary gate: upstream, the vendor and mozo run on the same GPU, in one
process, with torch's default CUDA settings -- the ones mozo's runtime leaves alone (cuDNN may use
TF32 for convolutions, matmuls do not; ``cudnn.benchmark`` and deterministic algorithms are off).
Upstream runs as ``init_model(..., device="cuda")`` + ``inference_bottomup`` run it: the letterbox
on the host, the ``uint8`` batch moved to the GPU and cast there, then the network, the decode,
NMS and the DCC on the GPU. The vendor makes its float batch on the host and moves it; every value
is an integer 0..255 either way. Every stage is copied to the host, which is exact, and compared
there. Each side first runs every frame twice and must repeat itself bit for bit, or the
comparison would mean nothing (VENDORING §2.7).

``--dtype`` is fp32 only. RTMO's CUDA runtimes through torch are ``torch-fp32`` alone: mozo
publishes no torch half runtime for it, its ``auto`` on CUDA is a TensorRT fp32 engine (measured by
parity and accuracy instead), and the DCC that runs after a graph runtime is torch fp32 -- the code
this gate checks. Upstream has no half-precision inference path either.

**On CPU, set ``MKL_CBWR=AVX2,STRICT``.** MKL picks sgemm kernels by buffer alignment, so two CPU
torch processes -- or two models in one -- can differ in the last bit without it (VENDORING §5).

``--falsify NAME`` perturbs one thing in the vendor and expects the gate to fail at the right
stage; ``--falsify list`` names them, ``--falsify all`` runs every one (each in a fresh process)
and fails unless each was caught. On CUDA three falsifiers only a GPU can catch are added (see
:data:`FALSIFIERS_CUDA`), and ``all`` first computes upstream's stages once, unperturbed, into a
reference cache (``--reference-cache``) that every falsifier reuses: upstream never changes
between them.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import hashlib
import os
import pickle
import random
import subprocess
import sys
import tempfile
import types
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
from tools.fetch.rtmo import CHECKPOINTS  # noqa: E402

#: mozo variant -> the upstream config it was trained with. The checkpoint files are
#: ``tools/fetch/rtmo.py``'s, the one place the upstream filenames live.
CONFIGS = {
    "tiny": "rtmo-t_8xb32-600e_body7-416x416.py",
    "small": "rtmo-s_8xb32-600e_body7-640x640.py",
    "medium": "rtmo-m_16xb16-600e_body7-640x640.py",
    "large": "rtmo-l_16xb16-600e_body7-640x640.py",
}

THRESHOLDS = (0.1, 0.5)

#: Perturbations of the vendor, each expected to be caught at the stage named.
FALSIFIERS = {
    "pad": "letterbox border 114 -> 113 (caught at the input square)",
    "swish": "backbone Swish -> nn.SiLU (caught at the first backbone convolution)",
    "neck_eps": "neck BN eps 1e-5 -> 1e-3 (caught at the neck's input projection)",
    "attn_pos": "position added to the value as well (caught at the encoder attention)",
    "softmax_eps": "DCC softmax epsilon 1e-8 -> 1e-7 (caught at the DCC output only)",
    "padding": "DCC window padding 1.25 -> 1.26 (caught at the DCC input, boxes unchanged)",
    "nms": "NMS threshold 0.65 -> 0.6 (caught at the final rows, network untouched)",
    "cells": "grid-cell centres moved by half a pixel (caught at the DCC and the boxes; every "
             "named network module exact)",
    "clamp": "boxes left unclamped (caught at the final boxes of the frames whose boxes run past "
             "the input square: the cut-off frame, and with OpenCV 5.1 the 37x23 one)",
}

#: Perturbations only a GPU run can catch, added on ``--device cuda``. A falsifier that makes the
#: vendor's batch on the GPU, as upstream does, is not among them: the batch holds integers 0..255,
#: exact on either side, so it is the same arithmetic and could never be caught.
FALSIFIERS_CUDA = {
    "tf32-off": "cuDNN TF32 off on the vendor's network alone (caught at the first convolution "
                "that uses it)",
    "channels-last": "the batch in channels-last memory, values equal (caught at the network: "
                     "cuDNN picks other kernels)",
    "dcc-host": "the DCC run on the host, as a CPU copy of the same module (caught at the DCC "
                "output; everything before it exact)",
}


def stub_mmcv_ops() -> None:
    """Give ``mmcv.ops`` a placeholder, so ``mmpose.models`` imports on ``mmcv-lite``."""
    ops = types.ModuleType("mmcv.ops")

    class _Absent:  # pragma: no cover - never instantiated by RTMO
        def __init__(self, *args, **kwargs):
            raise RuntimeError("mmcv.ops is stubbed: this path is not RTMO's")

    for name in ("MultiScaleDeformableAttention", "DeformConv2d", "batched_nms", "nms"):
        setattr(ops, name, _Absent)
    sys.modules["mmcv.ops"] = ops


def fixtures(coco: Path | None, count: int, seed: int = 0) -> list[tuple[str, np.ndarray]]:
    """BGR frames: the repo's photograph, awkward synthetic cases, and random COCO val2017."""
    frames = [(p.name, cv2.imread(str(p))) for p in photographs()]
    rng = np.random.default_rng(seed)
    example = frames[0][1]
    frames += [
        ("blank-640x480", np.full((480, 640, 3), 127, np.uint8)),
        ("noise-333x517", rng.integers(0, 256, (333, 517, 3), dtype=np.uint8)),
        ("panorama-2400x300", cv2.resize(example, (2400, 300))),
        ("tower-200x1500", cv2.resize(example, (200, 1500))),
        ("tiny-37x23", cv2.resize(example, (37, 23))),
        ("odd-641x479", cv2.resize(example, (641, 479))),
        ("huge-3000x2000", cv2.resize(example, (3000, 2000))),
        # A person cut by the frame's edge: the one fixture whose boxes run past the input square
        # before upstream clamps them, so the clamp is actually exercised.
        ("cut-1600x1281", np.ascontiguousarray(example[:, :1600])),
        ("grey", cv2.cvtColor(cv2.cvtColor(example, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)),
    ]
    if coco is not None and count:
        names = sorted(glob.glob(str(coco / "*.jpg")))
        for path in random.Random(seed).sample(names, count):
            frames.append((Path(path).name, cv2.imread(path)))
    return frames


def capture(model: torch.nn.Module, names: set[str]) -> tuple[dict, list]:
    """Record the output of every module named in *names* (first call each), via forward hooks,
    copied to the host as it is produced, in execution order."""
    seen: dict[str, object] = {}
    handles = []
    for name, module in model.named_modules():
        if not name or name not in names:  # the root returns data samples upstream, not tensors
            continue

        def hook(_module, _inputs, output, name=name):
            if name not in seen:
                seen[name] = tensors(output)
        handles.append(module.register_forward_hook(hook))
    return seen, handles


def tensors(value) -> list[torch.Tensor]:
    """Every tensor in *value*, copied to the host (exact): the stages are compared, and cached,
    on the CPU, and a copy cannot be overwritten by a later in-place op."""
    if torch.is_tensor(value):
        return [value.detach().to("cpu", copy=True)]
    if isinstance(value, (list, tuple)):
        return [t for item in value for t in tensors(item)]
    return []


class Fingerprint(tuple):
    """What the reference cache keeps of a large intermediate: shape, dtype and the sha256 of its
    bytes. The full module outputs of 4 variants x 26 frames run to many GB; the final rows and the
    DCC's (small) inputs and outputs are kept whole."""

    @classmethod
    def of(cls, value) -> "Fingerprint":
        array = (value.contiguous().numpy() if torch.is_tensor(value)
                 else np.ascontiguousarray(value))
        return cls((tuple(array.shape), str(value.dtype), hashlib.sha256(array.tobytes()).hexdigest()))


def equal(a, b) -> bool:
    if isinstance(a, Fingerprint):  # a cached reference: the bytes must be the same
        return a == Fingerprint.of(b)
    if torch.is_tensor(a):
        return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)


def delta(a, b) -> str:
    if isinstance(a, Fingerprint):
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
    """Apply one perturbation to the vendor modules (in this process only)."""
    # Through ``sys.modules``: the package re-exports names that shadow its submodules.
    network = sys.modules["mozo.vendors.rtmo_deploy.network"]
    postprocess = sys.modules["mozo.vendors.rtmo_deploy.postprocess"]
    predictor = sys.modules["mozo.vendors.rtmo_deploy.predictor"]
    if name == "pad":
        vendor.spec = dataclasses.replace(vendor.spec, pad_value=113)
    elif name == "swish":
        vendor.model.backbone.stem.conv.activate = torch.nn.SiLU()
    elif name == "neck_eps":
        for module in vendor.model.neck.input_proj:
            module.bn.eps = 1e-3
    elif name == "attn_pos":
        attn = vendor.model.neck.encoder[0].layers[0].self_attn

        def forward(query, pos, self=attn):
            q = (query + pos).transpose(0, 1)
            out = self.attn(query=q, key=(query + pos).transpose(0, 1),
                            value=(query + pos).transpose(0, 1))[0]
            return query + out.transpose(0, 1)
        attn.forward = forward
    elif name == "softmax_eps":
        def softmax(hms):
            hms = hms.clamp(min=-5e4, max=5e4)
            hms = hms - hms.max(dim=-1, keepdims=True).values
            exp = hms.exp()
            return exp / (exp.sum(dim=-1, keepdims=True) + 1e-7)
        network.DCC._softmax = staticmethod(softmax)
    elif name == "padding":
        vendor.spec = dataclasses.replace(vendor.spec, bbox_padding=1.26)
    elif name == "nms":
        vendor.spec = dataclasses.replace(vendor.spec, nms_threshold=0.6)
    elif name == "cells":
        original_priors = postprocess.grid_priors

        def shifted(*args, **kwargs):
            priors, strides = original_priors(*args, **kwargs)
            return priors + 0.5, strides
        network.grid_priors = predictor.grid_priors = shifted
    elif name == "clamp":
        predictor.clamp_boxes = lambda boxes, side: boxes
    elif name == "tf32-off":
        # Around the vendor's network only, so upstream in this process keeps the default.
        forward = vendor.model.forward

        def without_tf32(*args, **kwargs):
            torch.backends.cudnn.allow_tf32 = False
            try:
                return forward(*args, **kwargs)
            finally:
                torch.backends.cudnn.allow_tf32 = True
        vendor.model.forward = without_tf32
    elif name == "channels-last":
        original = predictor.to_batch
        predictor.to_batch = lambda canvas: original(canvas).contiguous(
            memory_format=torch.channels_last)
    elif name == "dcc-host":
        import copy
        on_host = copy.deepcopy(vendor.dcc).cpu()
        vendor.dcc.forward = lambda *inputs: on_host(*(t.cpu() for t in inputs)).to(vendor.device)
    else:
        raise SystemExit(f"unknown falsifier {name!r}; known: "
                         f"{sorted(FALSIFIERS) + sorted(FALSIFIERS_CUDA)}")


def load_upstream(variant: str, args, device: torch.device):
    """MMPose's model for *variant* on *device*, from the original checkpoint, with the DCC's
    calls recorded (it is called through ``forward_test``, not ``__call__``, so hooks cannot see
    it), and its test pipeline."""
    from mmengine.dataset import Compose
    from mmpose.apis import init_model

    config = args.mmpose / "configs/body_2d_keypoint/rtmo/body7" / CONFIGS[variant]
    checkpoint_name = CHECKPOINTS[variant][0]
    # The upstream file pickles numpy in its ``meta``, so it needs ``weights_only=False``; allowed
    # for this one call, on a file whose sha256 the fetch tool checked against the publisher's.
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    try:
        upstream = init_model(str(config), str(args.checkpoints / checkpoint_name),
                              device=str(device))
    finally:
        del os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"]

    dcc_calls: list = []
    original_forward_test = upstream.head.dcc.forward_test

    def recording_forward_test(pose_feats, bbox_cs, grids):
        out = original_forward_test(pose_feats, bbox_cs, grids)
        dcc_calls.append(tensors([pose_feats, bbox_cs, grids, out]))
        return out
    upstream.head.dcc.forward_test = recording_forward_test
    # Upstream's data pipeline, to compare the letterbox and the batch on their own.
    return upstream, Compose(upstream.cfg.test_dataloader.dataset.pipeline), dcc_calls


def run_upstream(upstream, pipeline, dcc_calls: list, bgr: np.ndarray, names: set[str]) -> dict:
    """Upstream's stages for one frame, on the host: the square, its centre and scale, the batch,
    every shared module's output in execution order (at the first threshold), and per threshold
    the DCC's inputs and output and the final rows."""
    from mmengine.dataset import pseudo_collate
    from mmpose.apis import inference_bottomup

    data = pipeline(dict(img=bgr.copy(), **upstream.dataset_meta))
    batch = upstream.data_preprocessor(pseudo_collate([data]))["inputs"]
    meta = data["data_samples"].metainfo
    out = {"square": data["inputs"].permute(1, 2, 0).numpy(), "center": meta["input_center"],
           "scale": meta["input_scale"], "batch": tensors(batch)[0], "modules": {}, "dcc": {},
           "rows": {}}
    for threshold in THRESHOLDS:
        upstream.test_cfg["score_thr"] = threshold
        seen, handles = capture(upstream, names) if threshold == THRESHOLDS[0] else ({}, [])
        dcc_calls.clear()
        try:
            result = inference_bottomup(upstream, bgr.copy())[0].pred_instances
        finally:
            for handle in handles:
                handle.remove()
        if threshold == THRESHOLDS[0]:
            out["modules"] = seen
        out["dcc"][threshold] = dcc_calls[0] if dcc_calls else None
        out["rows"][threshold] = (result.bboxes, result.bbox_scores, result.keypoints,
                                  result.keypoint_scores)
    return out


def run_vendor(vendor, rgb: np.ndarray, names: set[str]) -> dict:
    """The vendor's stages for one frame, in the shape :func:`run_upstream` returns."""
    batch, center, scale = vendor.preprocess(rgb)
    out = {"square": batch[0].permute(1, 2, 0).to(torch.uint8).numpy(), "center": center,
           "scale": scale, "batch": tensors(batch)[0], "modules": {}, "dcc": {}, "rows": {}}
    for threshold in THRESHOLDS:
        seen, handles = capture(vendor.model, names) if threshold == THRESHOLDS[0] else ({}, [])
        captured: list = []
        handles.append(vendor.dcc.register_forward_hook(
            lambda _m, inputs, output: captured.append(tensors([*inputs, output]))))
        try:
            people = vendor.predict(rgb, threshold)
        finally:
            for handle in handles:
                handle.remove()
        if threshold == THRESHOLDS[0]:
            out["modules"] = seen
        out["dcc"][threshold] = captured[0] if captured else None
        out["rows"][threshold] = (people.boxes, people.scores, people.keypoints, people.visibility)
    return out


def same_run(a: dict, b: dict) -> bool:
    """Whether two runs of one side gave the same bytes at every stage."""
    def same(x, y) -> bool:
        if x is None or y is None:
            return x is y
        if isinstance(x, (list, tuple)):
            return len(x) == len(y) and all(same(u, v) for u, v in zip(x, y))
        if isinstance(x, dict):
            return list(x) == list(y) and all(same(x[k], y[k]) for k in x)
        return equal(x, y)
    return same(a, b)


def fingerprinted(up: dict) -> dict:
    """What the reference cache keeps of one frame's upstream stages (see :class:`Fingerprint`)."""
    return {**up, "square": Fingerprint.of(up["square"]), "batch": Fingerprint.of(up["batch"]),
            "modules": {n: [Fingerprint.of(t) for t in ts] for n, ts in up["modules"].items()}}


def verify_variant(variant: str, args, checkpoint: Path, mozo, frames, gate: Gate,
                   cached: dict | None, saved: dict) -> None:
    from mozo.vendors.rtmo_deploy import Predictor

    device = torch.device(args.device)
    vendor = Predictor(checkpoint, variant, device)
    vendor_names = {n for n, _ in vendor.model.named_modules()}
    if cached is None:
        upstream, pipeline, dcc_calls = load_upstream(variant, args, device)
        names = {n for n, _ in upstream.named_modules()} & vendor_names
    if args.falsify:
        falsify(args.falsify, vendor)

    compared_modules = 0
    check_repeats = device.type == "cuda" and not args.falsify
    for label, bgr in frames:
        rgb = np.ascontiguousarray(bgr[..., ::-1])
        if cached is not None:
            up = cached[variant, label]
            names = set(up["modules"])
        else:
            up = run_upstream(upstream, pipeline, dcc_calls, bgr, names)
            if check_repeats:
                gate.record(f"{variant}/{label}/upstream repeats itself",
                            same_run(up, run_upstream(upstream, pipeline, dcc_calls, bgr, names)))
            if args.reference_cache is not None:
                saved[variant, label] = fingerprinted(up)
        mine = run_vendor(vendor, rgb, names)
        if check_repeats:
            gate.record(f"{variant}/{label}/vendor repeats itself",
                        same_run(mine, run_vendor(vendor, rgb, names)))

        for what in ("square", "center", "scale", "batch"):
            gate.check(f"{variant}/{label}/{what}", up[what], mine[what])
        # In upstream's execution order, so the first difference listed is the earliest.
        for name in [n for n in up["modules"] if n in mine["modules"]]:
            a, b = up["modules"][name], mine["modules"][name]
            if len(a) != len(b):
                gate.record(f"{variant}/{label}/module {name}", False,
                            lambda: f"{len(a)} vs {len(b)} output tensors")
                continue
            for index, (x, y) in enumerate(zip(a, b)):
                compared_modules += 1
                gate.check(f"{variant}/{label}/module {name}[{index}]", x, y)

        for threshold in THRESHOLDS:
            stage = f"{variant}/{label}/t={threshold}"
            calls, ours = up["dcc"][threshold], mine["dcc"][threshold]
            if calls is not None and gate.record(f"{stage}/dcc", ours is not None,
                                                 lambda: "upstream ran it, the vendor did not"):
                for what, x, y in zip(("pose", "window", "cells", "joints"), calls, ours):
                    gate.check(f"{stage}/dcc {what}", x, y)
            rows, our_rows = up["rows"][threshold], mine["rows"][threshold]
            for what, x, y in zip(("boxes", "scores", "keypoints", "keypoint scores"),
                                  rows, our_rows):
                gate.check(f"{stage}/{what}", x, y)
            if args.verbose:
                print(f"  {stage}: {len(our_rows[1])} people")
            if mozo is not None:
                verify_mozo(stage, rgb, threshold, rows, mozo[variant], gate)

    print(f"  {variant}: {len(names)} shared module names, {compared_modules} module outputs "
          f"compared over {len(frames)} frames"
          + (", each side run twice and repeated itself" if check_repeats else ""))


def verify_mozo(stage: str, rgb, threshold: float, rows, model, gate: Gate) -> None:
    """mozo's ``RTMOPredictor`` against upstream's rows, through PixelFlow on both sides."""
    import pixelflow as pf

    boxes, scores, keypoints, keypoint_scores = rows
    joints = np.concatenate([keypoints, keypoint_scores[..., None]], axis=-1)
    expected = pf.detections.from_arrays(
        boxes=boxes, scores=scores, class_ids=np.zeros(len(scores), dtype=np.int64),
        keypoints=joints, labels=model._labels).to_dict()
    actual = model.predict(rgb, threshold=threshold).to_dict()
    gate.check(f"{stage}/mozo rows", np.array(len(expected)), np.array(len(actual)))
    gate.record(f"{stage}/mozo", expected == actual)


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
            base += ["--reference-cache", str(Path(scratch) / "reference.pkl")]
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mmpose", type=Path, required=True, help="MMPose v1.3.2 checkout")
    parser.add_argument("--checkpoints", type=Path, required=True, help="the upstream .pth files")
    parser.add_argument("--repacked", type=Path, default=None,
                        help="republished state dicts as <variant>.pth (default: weights/ tree)")
    parser.add_argument("--coco", type=Path, default=None, help="COCO val2017 images")
    parser.add_argument("--count", type=int, default=16, help="random COCO images (seed 0)")
    parser.add_argument("--variants", nargs="*", default=list(CONFIGS))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--falsify", default=None)
    parser.add_argument("--reference-cache", type=Path, default=None,
                        help="upstream's stages: read from this file, or computed and saved there")
    parser.add_argument("--mozo", action="store_true", help="also check RTMOPredictor")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.dtype != "fp32":
        raise SystemExit(f"--dtype {args.dtype}: RTMO has no torch half-precision runtime on CUDA "
                         "(its only torch runtime is torch-fp32), so there is nothing to gate")
    falsifiers = {**FALSIFIERS, **(FALSIFIERS_CUDA if args.device == "cuda" else {})}
    if args.falsify == "list":
        for name, what in falsifiers.items():
            print(f"{name:14s} {what}")
        return 0
    if args.falsify == "all":
        return falsify_all(sys.argv[1:], list(falsifiers), args.device)
    if args.falsify and args.falsify not in falsifiers:
        raise SystemExit(f"unknown falsifier {args.falsify!r} on {args.device}; "
                         f"known: {sorted(falsifiers)}")
    device = torch.device(args.device)
    if device.type == "cpu" and os.environ.get("MKL_CBWR") != "AVX2,STRICT":
        print("warning: MKL_CBWR is not AVX2,STRICT; CPU results may differ between processes")

    stub_mmcv_ops()
    import mmpose
    print(f"mmpose {mmpose.__version__}, cv2 {cv2.__version__}, numpy {np.__version__}")
    settings = environment(device)
    print(settings)
    torch.set_grad_enabled(False)

    cached, saved = None, {}
    if args.reference_cache is not None and args.reference_cache.exists():
        with open(args.reference_cache, "rb") as handle:  # this script's own file
            cached = pickle.load(handle)
        print(f"upstream's stages read from {args.reference_cache}")

    frames = fixtures(args.coco, args.count)
    mozo = None
    if args.mozo:
        from mozo.adapters.rtmo import RTMOPredictor
        mozo = {v: RTMOPredictor(v, device=args.device, runtime="torch-fp32")
                for v in args.variants}

    def checkpoint_for(variant: str) -> Path:
        if args.repacked is not None:
            return args.repacked / f"{variant}.pth"
        from mozo.weights import resolve
        return resolve("rtmo", variant, "torch-fp32")

    gate = Gate()
    for variant in args.variants:
        verify_variant(variant, args, checkpoint_for(variant), mozo, frames, gate, cached, saved)
    if environment(device) != settings:  # nothing on either side may change a global setting
        gate.record("settings", False, lambda: f"changed during the run: {environment(device)}")

    if saved and not args.falsify:
        with open(args.reference_cache, "wb") as handle:
            pickle.dump(saved, handle)
        print(f"upstream's stages saved to {args.reference_cache}")

    print(f"\n{gate.passed} comparisons exact, {len(gate.failed)} differ")
    for line in gate.failed[:40]:
        print(f"  DIFF {line}")
    if args.falsify:
        first = gate.failed[0].split(": ")[0] if gate.failed else "-"
        print(f"falsifier {args.falsify!r}: {'CAUGHT' if gate.failed else 'NOT CAUGHT'}; "
              f"first difference at {first}")
        return 0 if gate.failed else 1
    return 1 if gate.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

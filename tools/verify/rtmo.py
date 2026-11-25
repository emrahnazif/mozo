#!/usr/bin/env python3
"""Verify the vendored RTMO against MMPose, stage by stage, exactly.

    MKL_CBWR=AVX2,STRICT python -I tools/verify/rtmo.py --mmpose /path/to/mmpose-v1.3.2 \\
        --checkpoints /path/to/upstream/pths --coco /path/to/val2017 [--variants small ...]

Runs in a scratch environment that has MMPose v1.3.2 installed (``mmengine``, ``mmcv-lite``,
``mmdet --no-deps``, ``mmpose --no-deps``); none of that is a mozo dependency. Three paths are
compared on CPU, with ``torch.equal`` / ``np.array_equal`` and no tolerance:

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

**Set ``MKL_CBWR=AVX2,STRICT``.** MKL picks sgemm kernels by buffer alignment, so two CPU torch
processes -- or two models in one -- can differ in the last bit without it (VENDORING §5).

``--falsify NAME`` perturbs one thing in the vendor and expects the gate to fail at the right
stage; ``--falsify list`` names them.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import os
import random
import sys
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
    "clamp": "boxes left unclamped (caught at the final boxes of the cut-off frame only)",
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
    """Record the output of every module named in *names* (first call each), via forward hooks."""
    seen: dict[str, object] = {}
    handles = []
    for name, module in model.named_modules():
        if not name or name not in names:  # the root returns data samples upstream, not tensors
            continue

        def hook(_module, _inputs, output, name=name):
            if name not in seen:
                seen[name] = output
        handles.append(module.register_forward_hook(hook))
    return seen, handles


def tensors(value) -> list[torch.Tensor]:
    if torch.is_tensor(value):
        return [value]
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
    else:
        raise SystemExit(f"unknown falsifier {name!r}; known: {sorted(FALSIFIERS)}")


def verify_variant(variant: str, args, checkpoint: Path, mozo, frames, gate: Gate) -> None:
    from mmpose.apis import inference_bottomup, init_model

    from mozo.vendors.rtmo_deploy import Predictor

    config = args.mmpose / "configs/body_2d_keypoint/rtmo/body7" / CONFIGS[variant]
    checkpoint_name = CHECKPOINTS[variant][0]
    # The upstream file pickles numpy in its ``meta``, so it needs ``weights_only=False``; allowed
    # for this one call, on a file whose sha256 the fetch tool checked against the publisher's.
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    try:
        upstream = init_model(str(config), str(args.checkpoints / checkpoint_name), device="cpu")
    finally:
        del os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"]
    vendor = Predictor(checkpoint, variant, "cpu")
    if args.falsify:
        falsify(args.falsify, vendor)

    # The DCC is called through ``forward_test``, not ``__call__``, so hooks cannot see it.
    dcc_calls: list = []
    original_forward_test = upstream.head.dcc.forward_test

    def recording_forward_test(pose_feats, bbox_cs, grids):
        out = original_forward_test(pose_feats, bbox_cs, grids)
        dcc_calls.append((pose_feats.clone(), bbox_cs.clone(), grids.clone(), out.clone()))
        return out
    upstream.head.dcc.forward_test = recording_forward_test

    # Upstream's data pipeline, to compare the letterbox and the batch on their own.
    from mmengine.dataset import Compose, pseudo_collate
    pipeline = Compose(upstream.cfg.test_dataloader.dataset.pipeline)

    upstream_names = {n for n, _ in upstream.named_modules()}
    vendor_names = {n for n, _ in vendor.model.named_modules()}
    shared = upstream_names & vendor_names
    compared_modules = 0

    for label, bgr in frames:
        rgb = np.ascontiguousarray(bgr[..., ::-1])
        data = pipeline(dict(img=bgr.copy(), **upstream.dataset_meta))
        batch_up = upstream.data_preprocessor(pseudo_collate([data]))["inputs"]
        batch_v, center, scale = vendor.preprocess(rgb)
        meta = data["data_samples"].metainfo
        gate.check(f"{variant}/{label}/square", data["inputs"].permute(1, 2, 0).numpy(),
                   batch_v[0].permute(1, 2, 0).to(torch.uint8).numpy())
        gate.check(f"{variant}/{label}/center", meta["input_center"], center)
        gate.check(f"{variant}/{label}/scale", meta["input_scale"], scale)
        gate.check(f"{variant}/{label}/batch", batch_up, batch_v)

        for threshold in THRESHOLDS:
            upstream.test_cfg["score_thr"] = threshold
            up_seen, up_handles = capture(upstream, shared)
            v_seen, v_handles = capture(vendor.model, shared)
            dcc_calls.clear()
            captured: dict = {}

            def dcc_hook(_m, inputs, output, captured=captured):
                captured["in"], captured["out"] = inputs, output
            v_handles.append(vendor.dcc.register_forward_hook(dcc_hook))
            try:
                result = inference_bottomup(upstream, bgr.copy())[0].pred_instances
                people = vendor.predict(rgb, threshold)
            finally:
                for handle in up_handles + v_handles:
                    handle.remove()

            if threshold == THRESHOLDS[0]:
                # In upstream's execution order, so the first difference listed is the earliest.
                for name in [n for n in up_seen if n in v_seen]:
                    a, b = tensors(up_seen[name]), tensors(v_seen[name])
                    if len(a) != len(b):
                        gate.record(f"{variant}/{label}/module {name}", False,
                                    lambda: f"{len(a)} vs {len(b)} output tensors")
                        continue
                    for index, (x, y) in enumerate(zip(a, b)):
                        compared_modules += 1
                        gate.check(f"{variant}/{label}/module {name}[{index}]", x, y)

            stage = f"{variant}/{label}/t={threshold}"
            if dcc_calls:
                pose_feats, bbox_cs, grids, keypoints = dcc_calls[0]
                if gate.record(f"{stage}/dcc", "in" in captured,
                               lambda: "upstream ran it, the vendor did not"):
                    gate.check(f"{stage}/dcc pose", pose_feats, captured["in"][0])
                    gate.check(f"{stage}/dcc window", bbox_cs, captured["in"][1])
                    gate.check(f"{stage}/dcc cells", grids, captured["in"][2])
                    gate.check(f"{stage}/dcc joints", keypoints, captured["out"])

            gate.check(f"{stage}/boxes", result.bboxes, people.boxes)
            gate.check(f"{stage}/scores", result.bbox_scores, people.scores)
            gate.check(f"{stage}/keypoints", result.keypoints, people.keypoints)
            gate.check(f"{stage}/keypoint scores", result.keypoint_scores, people.visibility)
            if args.verbose:
                print(f"  {stage}: {len(people.scores)} people")

        if mozo is not None:
            verify_mozo(variant, label, rgb, upstream, bgr, mozo, gate)

    print(f"  {variant}: {len(shared)} shared module names, {compared_modules} module outputs "
          f"compared over {len(frames)} frames")


def verify_mozo(variant, label, rgb, upstream, bgr, model, gate: Gate) -> None:
    """mozo's ``RTMOPredictor`` against upstream, through PixelFlow on both sides."""
    import pixelflow as pf
    from mmpose.apis import inference_bottomup

    for threshold in THRESHOLDS:
        upstream.test_cfg["score_thr"] = threshold
        result = inference_bottomup(upstream, bgr.copy())[0].pred_instances
        joints = np.concatenate([result.keypoints, result.keypoint_scores[..., None]], axis=-1)
        expected = pf.detections.from_arrays(
            boxes=result.bboxes, scores=result.bbox_scores,
            class_ids=np.zeros(len(result.bbox_scores), dtype=np.int64),
            keypoints=joints, labels=model[variant]._labels).to_dict()
        actual = model[variant].predict(rgb, threshold=threshold).to_dict()
        gate.check(f"{variant}/{label}/t={threshold}/mozo rows", np.array(len(expected)),
                   np.array(len(actual)))
        gate.record(f"{variant}/{label}/t={threshold}/mozo", expected == actual)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mmpose", type=Path, required=True, help="MMPose v1.3.2 checkout")
    parser.add_argument("--checkpoints", type=Path, required=True, help="the upstream .pth files")
    parser.add_argument("--repacked", type=Path, default=None,
                        help="republished state dicts as <variant>.pth (default: weights/ tree)")
    parser.add_argument("--coco", type=Path, default=None, help="COCO val2017 images")
    parser.add_argument("--count", type=int, default=16, help="random COCO images (seed 0)")
    parser.add_argument("--variants", nargs="*", default=list(CONFIGS))
    parser.add_argument("--falsify", default=None)
    parser.add_argument("--mozo", action="store_true", help="also check RTMOPredictor")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.falsify == "list":
        for name, what in FALSIFIERS.items():
            print(f"{name:12s} {what}")
        return 0
    if os.environ.get("MKL_CBWR") != "AVX2,STRICT":
        print("warning: MKL_CBWR is not AVX2,STRICT; CPU results may differ between processes")

    stub_mmcv_ops()
    import mmpose
    print(f"torch {torch.__version__}, mmpose {mmpose.__version__}, cv2 {cv2.__version__}, "
          f"numpy {np.__version__}")
    torch.set_grad_enabled(False)

    frames = fixtures(args.coco, args.count)
    mozo = None
    if args.mozo:
        from mozo.adapters.rtmo import RTMOPredictor
        mozo = {v: RTMOPredictor(v, device="cpu", runtime="torch-fp32") for v in args.variants}

    def checkpoint_for(variant: str) -> Path:
        if args.repacked is not None:
            return args.repacked / f"{variant}.pth"
        from mozo.weights import resolve
        return resolve("rtmo", variant, "torch-fp32")

    gate = Gate()
    for variant in args.variants:
        verify_variant(variant, args, checkpoint_for(variant), mozo, frames, gate)

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

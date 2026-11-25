#!/usr/bin/env python3
"""Check the vendored Mayaku models against ``mayaku`` itself, stage by stage, and mozo
against the vendor.

    python tools/verify/mayaku.py                         # every variant, the fixtures
    python tools/verify/mayaku.py --variant street-seg-nano photo.jpg other.jpg
    python tools/verify/mayaku.py --falsify all           # prove the gate can fail
    python tools/verify/mayaku.py --no-reference          # vendor vs mozo only

Three paths over the same weights and the same decoded pixels:

1. **upstream**, ``mayaku`` 3.1.1: ``Predictor.from_checkpoint`` (which builds the trained model,
   loads it strictly and fuses it with ``for_deploy``) called with the BGR image and ``conf=``;
2. **the vendor**, ``mozo.vendors.mayaku_deploy``, called with the RGB image;
3. **mozo**, ``ModelManager().get_model(...)`` with ``runtime="torch-fp32"``: registry, weights,
   adapter, PixelFlow.

Upstream against the vendor is compared at every stage -- the letterboxed canvas, the network's
input tensor, each raw output map, the detections in canvas pixels, and the final boxes, scores,
class ids and masks -- with ``torch.equal``. No tolerance. The vendor against mozo is compared
after both go through PixelFlow's rounding (``tests/conftest.py::as_pixelflow_reports``), exactly,
masks included.

It needs ``mayaku==3.1.1`` importable (``pip install --no-deps mayaku==3.1.1`` plus ``pydantic``,
``pyyaml``, ``typer``), so it does not run in the test suite. CPU only: that is where the
comparison is defined. Upstream decodes on the GPU when its model is on CUDA; the vendor always
decodes on the CPU.

``--falsify NAME`` perturbs one thing in the vendor and expects the gate to fail, at the stage
that thing reaches and at no earlier one. See :data:`FALSIFY`.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
# This file is called mayaku.py, so its own directory would shadow the reference package it checks.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault("MOZO_BASE_URL", f"file://{ROOT / 'weights'}")

import mozo  # noqa: E402
from common import fixtures  # noqa: E402
from conftest import as_pixelflow_reports  # noqa: E402
from mozo.image import load_image  # noqa: E402
from mozo.vendors.mayaku_deploy import SPECS, Detector, detect, decode as vdecode  # noqa: E402
from mozo.vendors.mayaku_deploy import image as vimage, mask as vmask  # noqa: E402
from mozo.vendors.mayaku_deploy import blocks as vblocks  # noqa: E402
from mozo.weights import resolve  # noqa: E402

FAMILY = "mayaku"
VARIANTS = tuple(SPECS)
#: The checkpoint's own threshold, and a low one where marginal detections are compared too.
THRESHOLDS = (0.4, 0.05)
STAGES = ("canvas", "input", "maps", "canvas dets", "boxes", "scores", "classes", "masks")


def _synthetic(photo: np.ndarray) -> dict[str, np.ndarray]:
    """The awkward cases: nothing to find, a tiny image, extreme aspect ratios both ways, an image
    already at the canvas size (no resize at all), and noise at an odd size."""
    rng = np.random.default_rng(0)
    return {
        "blank": np.full((480, 640, 3), 114, np.uint8),
        "tiny": np.ascontiguousarray(photo[::97, ::61]),
        "wide": np.ascontiguousarray(photo[500:620]),
        "tall": np.ascontiguousarray(photo[:, 900:1000]),
        "exact-800": np.ascontiguousarray(photo[:800, :800]),
        "noise": rng.integers(0, 256, (333, 517, 3), dtype=np.uint8),
    }


def _upstream_stages(predictor, bgr: np.ndarray, conf: float) -> dict:
    from mayaku.inference.decode import postprocess, predict_decode
    from mayaku.inference.preprocess import letterbox_batch, network_input

    x, metas = letterbox_batch([bgr], tuple(predictor.sidecar["canvas_hw"]))
    batch = network_input(x, torch.device("cpu"))
    with torch.no_grad():
        maps = predictor.model(batch)
    d = dataclasses.replace(predict_decode(predictor.sidecar), conf=conf)
    dets = postprocess(maps, len(predictor.class_names), predictor.sidecar["decode"]["reg_max"], d)[0][0]
    final = predictor(bgr, conf=conf)
    return {"canvas": np.ascontiguousarray(x[0][..., ::-1]), "input": batch, "maps": maps,
            "canvas dets": dets, "boxes": final.boxes, "scores": final.scores,
            "classes": final.labels, "masks": final.masks}


def _vendor_stages(detector: Detector, rgb: np.ndarray, conf: float) -> dict:
    canvas, _ = vimage.place(rgb, detector.spec.canvas)
    batch = vimage.planar(canvas)
    maps = detector.forward(batch)
    spec = detector.spec
    dets = vdecode.postprocess(maps, spec.num_classes, spec.cfg.reg_max, conf, spec.iou, spec.topk,
                               spec.max_det)[0]
    # The maps already computed, handed to the full path, so the network runs once per image.
    boxes, scores, classes, masks = detect(rgb, lambda _batch: maps, spec, conf)
    return {"canvas": canvas, "input": batch, "maps": maps, "canvas dets": dets,
            "boxes": boxes, "scores": scores, "classes": classes, "masks": masks}


def _equal(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.shape == b.shape and np.array_equal(a, b)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a.shape == b.shape and torch.equal(a, b)


def _mozo_matches(model, rgb: np.ndarray, conf: float, vendor: dict) -> bool:
    found = model.predict(rgb, threshold=conf)
    boxes, scores = as_pixelflow_reports(vendor["boxes"].numpy(), vendor["scores"].numpy(),
                                         vendor["classes"].numpy())
    rows = found.to_dict()
    got_boxes = np.array([r["bbox"] for r in rows], np.float64).reshape(-1, 4)
    got_scores = np.array([r["confidence"] for r in rows], np.float64)
    ok = (np.array_equal(boxes, got_boxes) and np.array_equal(scores, got_scores)
          and [r["class_id"] for r in rows] == vendor["classes"].tolist())
    if vendor["masks"] is not None:
        ok = ok and all(np.array_equal(d.masks[0], m) for d, m in zip(found, vendor["masks"].numpy()))
    return ok


#: Each perturbation, and the first stage it must fail at. A gate that fails earlier than the
#: stage a change reaches is comparing something else; one that never fails is not a gate.
FALSIFY = {
    "pad": ("canvas", lambda: setattr(vimage, "PAD", 115)),
    "rescale": ("input", lambda: setattr(vimage, "planar", _multiplied_planar)),
    "bn-eps": ("maps", lambda: setattr(vblocks, "fuse_conv_bn_weights", _eps_nudged)),
    "nms-iou": ("canvas dets", lambda: setattr(vdecode, "batched_nms", _nms_at(0.51))),
    "coord-scale": ("masks", lambda: setattr(vmask, "COORD_SCALE", 8.0001)),
}


def _multiplied_planar(canvas):
    return torch.from_numpy((canvas[None].transpose(0, 3, 1, 2) * np.float32(1 / 255))
                            .astype(np.float32))


def _eps_nudged(w, b, mean, var, eps, gamma, beta, _original=vblocks.fuse_conv_bn_weights):
    return _original(w, b, mean, var, eps * 1.001, gamma, beta)


def _nms_at(iou, _original=vdecode.batched_nms):
    return lambda boxes, scores, idxs, _iou: _original(boxes, scores, idxs, iou)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="*", type=Path)
    parser.add_argument("--variant", choices=VARIANTS, action="append")
    parser.add_argument("--falsify", choices=(*FALSIFY, "all"))
    parser.add_argument("--no-reference", action="store_true")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", message="predictions use conf")

    if args.falsify == "all":
        import subprocess

        failed = []
        for name in FALSIFY:
            extra = [f"--variant={v}" for v in (args.variant or ["street-seg-nano"])]
            done = subprocess.run([sys.executable, __file__, "--falsify", name, *extra,
                                   *map(str, args.images)], capture_output=True, text=True)
            verdict = done.stdout.strip().splitlines()[-1] if done.stdout.strip() else done.stderr
            print(f"{name:<12} {verdict}")
            failed += [] if done.returncode == 0 else [name]
        return 1 if failed else 0

    reference = not args.no_reference
    if reference:
        import mayaku
        from mayaku.inference.predictor import Predictor

        print(f"mayaku {mayaku.__version__}, torch {torch.__version__}, CPU")
        if mayaku.__version__ != "3.1.1":
            raise SystemExit("the reference is mayaku 3.1.1")

    expect = None
    if args.falsify:
        expect, apply = FALSIFY[args.falsify]
        apply()
        print(f"falsify {args.falsify}: expecting the first failure at '{expect}'")

    paths = args.images or fixtures()
    images = {p.name: load_image(str(p)) for p in paths}
    images.update(_synthetic(next(iter(images.values()))))
    first_failures, compared = set(), 0
    for variant in args.variant or VARIANTS:
        checkpoint = resolve(FAMILY, variant, "torch-fp32")
        detector = Detector(checkpoint, variant)
        upstream = Predictor.from_checkpoint(checkpoint, device="cpu") if reference else None
        served = None if args.falsify else mozo.ModelManager().get_model(
            FAMILY, variant, device="cpu", runtime="torch-fp32")
        for conf in THRESHOLDS:
            for name, rgb in images.items():
                vendor = _vendor_stages(detector, rgb, conf)
                failed = []
                if upstream is not None:
                    want = _upstream_stages(upstream, np.ascontiguousarray(rgb[..., ::-1]), conf)
                    failed = [s for s in STAGES if not _equal(want[s], vendor[s])]
                if served is not None and not _mozo_matches(served, rgb, conf, vendor):
                    failed.append("mozo")
                compared += len(vendor["scores"])
                if failed:
                    first_failures.add(failed[0])
                print(f"  {'FAIL' if failed else 'ok':<4} {variant:<{max(map(len, VARIANTS))}} conf {conf:<4} {name:<18} "
                      f"{len(vendor['scores']):>4} detections"
                      + (f"   differs at: {', '.join(failed)}" if failed else ""))
    if args.falsify:
        if first_failures == {expect}:
            print(f"caught: failed first at '{expect}', as expected")
            return 0
        print(f"NOT CAUGHT as expected: first failures {sorted(first_failures) or 'none'}")
        return 1
    if first_failures:
        print(f"\nDISAGREEMENT, first at: {sorted(first_failures)}")
        return 1
    print(f"\nidentical at every stage: {compared} detections compared")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

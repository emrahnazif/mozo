#!/usr/bin/env python3
"""Check the vendored Mayaku models against ``mayaku`` itself, stage by stage, and mozo
against the vendor.

    python tools/verify/mayaku.py                         # every variant, the fixtures
    python tools/verify/mayaku.py --variant street-seg-nano photo.jpg other.jpg
    python tools/verify/mayaku.py --falsify all           # prove the gate can fail
    python tools/verify/mayaku.py --no-reference          # vendor vs mozo only
    python tools/verify/mayaku.py --device cuda           # the primary gate: GPU against GPU
    python tools/verify/mayaku.py --device cuda --falsify all

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
``pyyaml``, ``typer``), so it does not run in the test suite.

``--device cuda`` is the primary gate: upstream, the vendor and mozo all run their network on the
same GPU, in one process, with torch's default CUDA settings -- the ones mozo's runtime leaves
alone (cuDNN may use TF32 for convolutions, ``cudnn.benchmark`` and deterministic algorithms are
off). Every stage is compared on the CPU after an exact copy. Each side is first run twice and must
repeat itself bit for bit, or the comparison would mean nothing. Both make the network's input on
the GPU and decode there, as upstream does on CUDA; ``tf32-off`` is the falsifier only a GPU run
has.

``--dtype`` is fp32 only. Mayaku's CUDA runtimes through torch are ``torch-fp32`` alone (its
fp16 is a TensorRT engine, measured by parity and accuracy instead), and upstream has no half
precision inference path to hold one against.

``--falsify NAME`` perturbs one thing in the vendor and expects the gate to fail, at the stage
that thing reaches and at no earlier one. See :data:`FALSIFY`.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import subprocess
import sys
import tempfile
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
from mozo.vendors.mayaku_deploy import blocks as vblocks, model as vmodel  # noqa: E402
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


def _cpu(value):
    """*value* with every tensor copied to the CPU, which is exact, so stages compare anywhere."""
    if isinstance(value, torch.Tensor):
        return value.cpu()
    if isinstance(value, (list, tuple)):
        return [_cpu(v) for v in value]
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    return value


def _upstream_stages(predictor, bgr: np.ndarray, conf: float) -> dict:
    from mayaku.inference.decode import postprocess, predict_decode
    from mayaku.inference.preprocess import letterbox_batch, network_input

    x, metas = letterbox_batch([bgr], tuple(predictor.sidecar["canvas_hw"]))
    batch = network_input(x, predictor.device)
    with torch.no_grad():
        maps = predictor.model(batch)
    d = dataclasses.replace(predict_decode(predictor.sidecar), conf=conf)
    dets = postprocess(maps, len(predictor.class_names), predictor.sidecar["decode"]["reg_max"], d)[0][0]
    final = predictor(bgr, conf=conf)
    return _cpu({"canvas": np.ascontiguousarray(x[0][..., ::-1]), "input": batch, "maps": maps,
                 "canvas dets": dets, "boxes": final.boxes, "scores": final.scores,
                 "classes": final.labels, "masks": final.masks})


def _vendor_stages(detector: Detector, rgb: np.ndarray, conf: float) -> dict:
    canvas, _ = vimage.place(rgb, detector.spec.canvas)
    batch = vimage.network_input(canvas, detector.device)
    maps = detector.forward(batch)
    spec = detector.spec
    dets = vdecode.postprocess(maps, spec.num_classes, spec.cfg.reg_max, conf, spec.iou, spec.topk,
                               spec.max_det)[0]
    # The maps already computed, handed to the full path, so the network runs once per image.
    boxes, scores, classes, masks = detect(rgb, lambda _batch: maps, spec, conf, detector.device)
    return _cpu({"canvas": canvas, "input": batch, "maps": maps, "canvas dets": dets,
                 "boxes": boxes, "scores": scores, "classes": classes, "masks": masks})


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
#: On CUDA, ``rescale`` cannot be caught and is replaced: the vendor makes the input on the device
#: there, where torch's division by a scalar already is a multiplication by its reciprocal, so the
#: perturbation is the same arithmetic. ``host-input`` makes it on the host instead, dividing
#: exactly; ``tf32-off`` changes a CUDA setting on the vendor's side alone (the reference comes
#: from a cached run, so only the vendor sees it).
FALSIFY_CUDA = {
    "host-input": ("input", lambda: _patch_input(lambda c, dev: vimage.planar(c).to(dev))),
    "tf32-off": ("maps", lambda: setattr(torch.backends.cudnn, "allow_tf32", False)),
}


def _patch_input(fn):
    setattr(vimage, "network_input", fn)
    setattr(vmodel, "network_input", fn)


def _multiplied_planar(canvas):
    return torch.from_numpy((canvas[None].transpose(0, 3, 1, 2) * np.float32(1 / 255))
                            .astype(np.float32))


def _eps_nudged(w, b, mean, var, eps, gamma, beta, _original=vblocks.fuse_conv_bn_weights):
    return _original(w, b, mean, var, eps * 1.001, gamma, beta)


def _nms_at(iou, _original=vdecode.batched_nms):
    return lambda boxes, scores, idxs, _iou: _original(boxes, scores, idxs, iou)


def _environment(device: torch.device) -> str:
    """torch, CUDA, cuDNN, the GPU and its driver: what a GPU result holds for."""
    if device.type == "cpu":
        return f"torch {torch.__version__}, CPU"
    driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader",
                             "-i", str(device.index or 0)], capture_output=True, text=True)
    return (f"torch {torch.__version__}, CUDA {torch.version.cuda}, cuDNN "
            f"{torch.backends.cudnn.version()}, {torch.cuda.get_device_name(device)}, driver "
            f"{driver.stdout.strip() or 'unknown'}; cudnn.allow_tf32 "
            f"{torch.backends.cudnn.allow_tf32}, matmul.allow_tf32 "
            f"{torch.backends.cuda.matmul.allow_tf32}, cudnn.benchmark "
            f"{torch.backends.cudnn.benchmark}, deterministic "
            f"{torch.are_deterministic_algorithms_enabled()}")


def _falsify_all(args, falsifiers: dict) -> int:
    """Run every falsifier in its own process against one cached reference, computed (and gated
    unperturbed) once first: upstream never changes between falsifiers."""
    common = [f"--device={args.device}", f"--dtype={args.dtype}",
              *[f"--variant={v}" for v in (args.variant or ["street-seg-nano"])],
              *map(str, args.images)]
    with tempfile.TemporaryDirectory() as scratch:
        cache = str(Path(scratch) / "reference.pt")
        failed = []
        for name in (None, *falsifiers):
            what = ["--falsify", name, "--reference-cache", cache] if name else [
                "--reference-cache", cache]
            done = subprocess.run([sys.executable, __file__, *what, *common],
                                  capture_output=True, text=True)
            verdict = done.stdout.strip().splitlines()[-1] if done.stdout.strip() else done.stderr
            print(f"{name or '(none)':<12} {verdict}")
            failed += [] if done.returncode == 0 else [name or "(none)"]
            if name is None and failed:
                return 1
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="*", type=Path)
    parser.add_argument("--variant", choices=VARIANTS, action="append")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--falsify", choices=(*FALSIFY, *FALSIFY_CUDA, "all"))
    parser.add_argument("--no-reference", action="store_true")
    parser.add_argument("--reference-cache", type=Path,
                        help="upstream's stages: read from this file, or computed and saved there")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", message="predictions use conf")
    if args.dtype != "fp32":
        parser.error("Mayaku runs through torch in fp32 only: mozo publishes no torch half runtime "
                     "for it (fp16 is a TensorRT engine) and upstream has no half inference path")
    device = torch.device(args.device)
    falsifiers = FALSIFY if device.type == "cpu" else {
        **{k: v for k, v in FALSIFY.items() if k != "rescale"}, **FALSIFY_CUDA}
    if args.falsify not in (*falsifiers, "all", None):
        parser.error(f"--falsify {args.falsify} does not apply on {device.type}")

    if args.falsify == "all":
        return _falsify_all(args, falsifiers)

    reference = not args.no_reference
    cached = None
    if args.reference_cache and args.reference_cache.exists():
        cached = torch.load(args.reference_cache, weights_only=False)
        if cached.pop("for") != (args.device, torch.__version__):
            raise SystemExit(f"{args.reference_cache} was made for another device or torch")
    if reference and cached is None:
        import mayaku
        from mayaku.inference.predictor import Predictor

        if mayaku.__version__ != "3.1.1":
            raise SystemExit("the reference is mayaku 3.1.1")
    print(("mayaku 3.1.1 (cached), " if cached is not None else
           "mayaku 3.1.1, " if reference else "") + _environment(device))

    expect = None
    if args.falsify:
        expect, apply = falsifiers[args.falsify]
        apply()
        print(f"falsify {args.falsify}: expecting the first failure at '{expect}'")

    paths = args.images or fixtures()
    images = {p.name: load_image(str(p)) for p in paths}
    images.update(_synthetic(next(iter(images.values()))))
    first_failures, compared, saved = set(), 0, {}
    for variant in args.variant or VARIANTS:
        checkpoint = resolve(FAMILY, variant, "torch-fp32")
        detector = Detector(checkpoint, variant, device)
        upstream = Predictor.from_checkpoint(checkpoint, device=str(device)) \
            if reference and cached is None else None
        served = None if args.falsify else mozo.ModelManager().get_model(
            FAMILY, variant, device=str(device), runtime="torch-fp32")
        for conf in THRESHOLDS:
            for name, rgb in images.items():
                key = (variant, conf, name)
                vendor = _vendor_stages(detector, rgb, conf)
                failed = []
                if device.type == "cuda" and not args.falsify:
                    # Determinism first: each side must repeat itself before they are compared.
                    again = _vendor_stages(detector, rgb, conf)
                    failed += ["vendor repeat"] if not all(
                        _equal(vendor[s], again[s]) for s in STAGES) else []
                want = None
                if cached is not None:
                    want = cached[key]
                elif upstream is not None:
                    bgr = np.ascontiguousarray(rgb[..., ::-1])
                    want = _upstream_stages(upstream, bgr, conf)
                    if device.type == "cuda":
                        again = _upstream_stages(upstream, bgr, conf)
                        failed += ["upstream repeat"] if not all(
                            _equal(want[s], again[s]) for s in STAGES) else []
                    saved[key] = want
                if want is not None:
                    failed += [s for s in STAGES if not _equal(want[s], vendor[s])]
                if served is not None and not _mozo_matches(served, rgb, conf, vendor):
                    failed.append("mozo")
                compared += len(vendor["scores"])
                if failed:
                    first_failures.add(failed[0])
                print(f"  {'FAIL' if failed else 'ok':<4} {variant:<{max(map(len, VARIANTS))}} conf {conf:<4} {name:<18} "
                      f"{len(vendor['scores']):>4} detections"
                      + (f"   differs at: {', '.join(failed)}" if failed else ""))
    if args.reference_cache and cached is None and saved:
        torch.save({**saved, "for": (args.device, torch.__version__)}, args.reference_cache)
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

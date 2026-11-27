#!/usr/bin/env python3
"""Export Mayaku checkpoints to the ONNX graphs mozo publishes, checked before they are written.

    python tools/export/mayaku.py street-nano street-seg-nano street-medium street-seg-medium
    python tools/export/mayaku.py street-nano street-medium --fp16      # the fp16 graph, on CUDA
    python tools/export/mayaku.py street-nano street-medium --coreml    # CoreML fp32 + fp16, on a Mac

Bootstrap tooling; never ships. ``onnx-fp32`` is the fused deploy graph at the street models' one
canvas, ``1x3x800x800``, opset 17, outputs named as upstream names them (``cls0, box0, ... ker2``).
It contains five op types and nothing else -- Conv, Relu, Add, nearest Resize, MaxPool -- and this
refuses a graph that contains anything more, the same contract upstream asserts on its own exports.

The shape is fixed, not dynamic, because the graph is not shape-agnostic: the spatial pyramid
pooling's global-context term runs ``max(h, w) - 1`` max-pools, so the trace unrolls 24 of them for
a 25x25 stride-32 map and would be wrong at any other canvas. mozo only ever runs the 800 canvas.

``onnx-fp16`` (``--fp16``) is exported on CUDA under torch's fp16 autocast, which casts every
convolution's inputs to half and keeps the graph's input fp32: the types TensorRT 11's strongly
typed builder then honours. It is published for TensorRT only.

Each graph is run through ONNX Runtime on the CPU and compared, through the vendor's own decode,
against the torch module it came from, on real photographs. fp32 must give the same detections:
same count, boxes within 0.01 px, scores within 0.001, masks overlapping at least 0.999. fp16 must
keep every confident detection (see :func:`_changes`).

``coreml-fp32`` and ``coreml-fp16`` (``--coreml``) are ML Program packages traced from the same
fused network, zipped, with a float tensor input named ``images`` -- the batch the vendor already
makes, so no scaling op runs on the CPU ahead of the Neural Engine -- and the graph's output names.
Each is run through Core ML (compute units ALL, as :class:`mozo.runtimes.CoreMLRunner` runs it) and
held to the same checks: fp32 to the ONNX tolerances except masks, at least 0.998 (Core ML's fp32
GPU kernels sum in their own order and flip border pixels: worst 0.9985 on 301 photographs, M1
Max), fp16 to no confident change. Mayaku's own Core ML export takes an image input instead, which
:class:`~mozo.runtimes.CoreMLRunner` does not feed.
"""

from __future__ import annotations

import collections
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import torch
from torchvision.ops import box_iou

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from common import fixtures, variant_parser  # noqa: E402
from tools.fetch.mayaku import REVISION  # noqa: E402
from mozo.image import load_image  # noqa: E402
from mozo.runtimes import OnnxRunner  # noqa: E402
from mozo.vendors.mayaku_deploy import Detector, detect, output_names  # noqa: E402

FAMILY = "mayaku"
OPSET, FP16_OPSET = 17, 18   # opset 17's TorchScript path cannot export autocast's casts
DEPLOY_OPS = {"Conv", "Relu", "Add", "Resize", "MaxPool"}
SCAFFOLDING = {"Constant", "Cast"}   # Resize's scales, and autocast's casts in the fp16 graph

#: Thresholds of the comparison. Low, so marginal detections -- where two executors diverge
#: first -- are compared too.
CONF = 0.05
BOX_TOLERANCE, SCORE_TOLERANCE, MASK_IOU = 1e-2, 1e-3, 0.999
COREML_MASK_IOU = 0.998


def _export(detector: Detector, destination: Path, fp16: bool) -> collections.Counter:
    """Trace the fused network to *destination* and refuse anything outside the five-op contract."""
    network, x = detector.network, torch.zeros(1, 3, *detector.spec.canvas)
    kwargs = dict(input_names=["images"], output_names=output_names(detector.spec.seg),
                  dynamo=False)
    with torch.no_grad():
        if fp16:
            with torch.autocast("cuda", torch.float16):
                torch.onnx.export(network.cuda(), x.cuda(), destination, opset_version=FP16_OPSET,
                                  **kwargs)
            network.cpu()
        else:
            torch.onnx.export(network, x, destination, opset_version=OPSET, **kwargs)
    graph = onnx.load(destination)
    ops = collections.Counter(node.op_type for node in graph.graph.node)
    extra = set(ops) - DEPLOY_OPS - SCAFFOLDING
    if extra:
        raise SystemExit(f"{destination.name}: ops outside the five-op contract: {sorted(extra)}")
    for node in graph.graph.node:
        if node.op_type == "Resize":
            mode = next(a.s for a in node.attribute if a.name == "mode")
            if mode != b"nearest":
                raise SystemExit(f"{destination.name}: a {mode!r} Resize; the contract is nearest")
    return collections.Counter({k: v for k, v in ops.items() if k in DEPLOY_OPS})


def _export_coreml(traced: torch.jit.ScriptModule, spec, package: Path, fp16: bool) -> None:
    """Convert the traced network to an ML Program saved as the directory *package*."""
    import coremltools as ct

    model = ct.convert(
        traced, inputs=[ct.TensorType(name="images", shape=(1, 3, *spec.canvas))],
        outputs=[ct.TensorType(name=n) for n in output_names(spec.seg)],
        convert_to="mlprogram", minimum_deployment_target=ct.target.macOS13,
        compute_precision=ct.precision.FLOAT16 if fp16 else ct.precision.FLOAT32)
    model.save(str(package))


def _graph_forward(runner, seg: bool):
    """A forward over any graph runner: outputs matched by name, as the adapter matches them."""
    def forward(batch):
        found = dict(zip(runner.outputs, runner(batch.numpy())))
        return [torch.from_numpy(np.asarray(found[name], np.float32)) for name in output_names(seg)]
    return forward


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def _changes(want, got, high=0.52, low=0.5) -> int:
    """Confident detections one side has and the other lacks: a detection above *high* with no
    same-class partner above *low* at IoU over 0.5. The margin keeps a score moving across a
    single threshold from counting as a change."""
    n = 0
    for (bx, sx, cx), (by, sy, cy) in ((want[:3], got[:3]), (got[:3], want[:3])):
        for k in torch.nonzero(sx > high).flatten().tolist():
            partners = (sy > low) & (cy == cx[k])
            if not bool(partners.any()) or float(box_iou(bx[k:k + 1], by[partners]).max()) <= 0.5:
                n += 1
    return n


def _references(detector: Detector) -> list[tuple[str, np.ndarray, tuple]]:
    """Each fixture photograph with the torch module's own answer, computed once per variant."""
    found = []
    for image_path in fixtures():
        image = load_image(str(image_path))
        with torch.no_grad():
            found.append((image_path.name, image, detect(image, detector.forward, detector.spec, CONF)))
    return found


def _check(detector: Detector, runner, references: list, fp16: bool,
           mask_iou: float = MASK_IOU) -> None:
    """Compare *runner* against the torch module through the vendor's decode, image by image."""
    forward = _graph_forward(runner, detector.spec.seg)
    problems = []
    for name, image, want in references:
        with torch.no_grad():
            got = detect(image, forward, detector.spec, CONF)
        if fp16:
            n = _changes(want, got)
            print(f"    {name:<20} {len(want[1])} / {len(got[1])} detections, {n} confident changes")
            if n:
                problems.append(f"{name}: {n} confident detections changed in fp16")
            continue
        if len(want[1]) != len(got[1]) or not torch.equal(want[2], got[2]):
            problems.append(f"{name}: {len(want[1])} detections from torch, {len(got[1])} from "
                            "the graph, or their classes differ")
            continue
        box = float((want[0] - got[0]).abs().max()) if len(want[1]) else 0.0
        score = float((want[1] - got[1]).abs().max()) if len(want[1]) else 0.0
        worst = min((_iou(a, b) for a, b in zip(want[3].numpy(), got[3].numpy())), default=1.0) \
            if want[3] is not None else 1.0
        print(f"    {name:<20} {len(want[1])} detections, boxes {box:.2g} px, scores {score:.2g}, "
              f"worst mask IoU {worst:.6f}")
        if box > BOX_TOLERANCE or score > SCORE_TOLERANCE or worst < mask_iou:
            problems.append(f"{name}: boxes {box:g} px, scores {score:g}, mask IoU {worst:g}")
    if problems:
        raise SystemExit("not published, the graph disagrees with torch:\n  " + "\n  ".join(problems))


def _detector(variant: str, revision: str, weights_dir: Path) -> tuple[Detector, Path]:
    """The variant's checkpoint loaded, and the revision directory its artifacts are written to."""
    directory = weights_dir / FAMILY / variant / revision
    checkpoint = directory / "torch-fp32.pth"
    if not checkpoint.is_file():
        raise SystemExit(f"{checkpoint} is missing. Run tools/fetch/{FAMILY}.py {variant} first.")
    return Detector(checkpoint, variant), directory


def _wrote(path: Path) -> None:
    print(f"  wrote {path.name}, {path.stat().st_size / 1e6:.1f} MB")


def export_coreml(variant: str, revision: str, weights_dir: Path) -> None:
    from mozo.runtimes import CoreMLRunner

    detector, directory = _detector(variant, revision, weights_dir)
    references = _references(detector)
    with torch.no_grad():
        traced = torch.jit.trace(detector.network, torch.zeros(1, 3, *detector.spec.canvas))
    for fp16 in (False, True):
        key = "coreml-fp16" if fp16 else "coreml-fp32"
        print(f"{FAMILY}/{variant} -> {key}")
        with tempfile.TemporaryDirectory(dir=directory) as staging:
            package = Path(staging) / f"{key}.mlpackage"
            _export_coreml(traced, detector.spec, package, fp16)
            _check(detector, CoreMLRunner(package), references, fp16, COREML_MASK_IOU)
            archive = shutil.make_archive(str(Path(staging) / key), "zip", root_dir=package)
            Path(archive).replace(directory / f"{key}.zip")
        _wrote(directory / f"{key}.zip")


def export_variant(variant: str, revision: str, weights_dir: Path, fp16: bool) -> None:
    if fp16 and not torch.cuda.is_available():
        raise SystemExit("the fp16 graph is exported on CUDA, under fp16 autocast")
    detector, directory = _detector(variant, revision, weights_dir)
    key = "onnx-fp16" if fp16 else "onnx-fp32"
    print(f"{FAMILY}/{variant} -> {key}")
    with tempfile.TemporaryDirectory(dir=directory) as staging:
        staged = Path(staging) / f"{key}.onnx"
        ops = _export(detector, staged, fp16)
        print(f"  {dict(ops)}")
        _check(detector, OnnxRunner(staged, device="cpu"), _references(detector), fp16)
        staged.replace(directory / f"{key}.onnx")
    _wrote(directory / f"{key}.onnx")


def main() -> int:
    parser = variant_parser(__doc__, ROOT / "weights", required=True, revision=REVISION)
    parser.add_argument("--fp16", action="store_true", help="export the fp16 graph instead")
    parser.add_argument("--coreml", action="store_true",
                        help="export the CoreML fp32 and fp16 packages instead (macOS)")
    args = parser.parse_args()
    for variant in args.variants:
        if args.coreml:
            export_coreml(variant, args.revision, args.weights_dir)
        else:
            export_variant(variant, args.revision, args.weights_dir, args.fp16)
    print("\nRun tools/generate_manifest.py next.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

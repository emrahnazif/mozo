#!/usr/bin/env python3
"""Export DEIM checkpoints to the ONNX graphs mozo publishes, checked before they are written.

    python tools/export/deim.py nano small medium large xlarge xlarge-obj365
    python tools/export/deim.py nano --fp16          # the fp16 graph, on CUDA

Bootstrap tooling; never ships. ``onnx-fp32`` is the deploy-form network (``deim_deploy.load``:
encoder conv+BN fused, deploy-form weighting function) at its one input size, ``1x3x640x640``,
opset 17, TorchScript exporter, with outputs named after the keys the vendor's decode reads
(``pred_logits``, ``pred_boxes``). Preprocessing and the top-k decode stay outside the graph, in
the vendor, on every runtime. Upstream's own exporter puts the post-processor inside the graph
with a second ``orig_target_sizes`` input; mozo does not, so the decode cannot drift between
runtimes.

The shape is fixed: the encoder's position embedding and the decoder's anchors are computed for
640x640 and baked in, as they are in upstream's eval path, so the graph is only right at 640.

``onnx-fp16`` (``--fp16``) is exported on CUDA under torch's fp16 autocast, which casts matrix
products and convolutions to half and keeps the graph's input fp32: the types TensorRT 11's
strongly typed builder then honours. Opset 18, because opset 17's TorchScript path cannot export
autocast's casts. Published for TensorRT only.

Each graph is run through ONNX Runtime on the CPU and compared, through the vendor's decode,
against the torch module it came from, on the fixture photographs. fp32 must give the same
detections above 0.3, each matched to the graph's same-class detection of highest IoU: boxes within
0.01 px, scores within 0.001, none unmatched. fp16 must keep every
confident detection (see :func:`_changes`).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
from torchvision.ops import box_iou

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from common import fixtures, variant_parser  # noqa: E402
from mozo.image import load_image  # noqa: E402
from mozo.runtimes import OnnxRunner  # noqa: E402
from mozo.vendors.deim_deploy import get_spec, load, postprocess, preprocess  # noqa: E402

FAMILY = "deim"
REVISION = "2026-11-25"
OPSET, FP16_OPSET = 17, 18
OUTPUTS = ["pred_logits", "pred_boxes"]
CONF = 0.05
BOX_TOLERANCE, SCORE_TOLERANCE = 1e-2, 1e-3


def _export(model, destination: Path, size: tuple[int, int], fp16: bool) -> None:
    x = torch.zeros(1, 3, *size)
    kwargs = dict(input_names=["images"], output_names=OUTPUTS, dynamo=False)
    with torch.no_grad():
        if fp16:
            with torch.autocast("cuda", torch.float16):
                torch.onnx.export(model.cuda(), x.cuda(), destination, opset_version=FP16_OPSET,
                                  **kwargs)
            model.cpu()
        else:
            torch.onnx.export(model, x, destination, opset_version=OPSET, **kwargs)


def _detect(forward, image, size):
    batch, original = preprocess(image, size)
    labels, boxes, scores = postprocess(forward(batch), original)
    keep = scores[0] > CONF
    return labels[0][keep], boxes[0][keep], scores[0][keep]


def _changes(want, got, high=0.52, low=0.5) -> int:
    """Confident detections one side has and the other lacks: a detection above *high* with no
    same-class partner above *low* at IoU over 0.5. The margin keeps a score moving across a
    single threshold from counting as a change."""
    n = 0
    for (lx, bx, sx), (ly, by, sy) in ((want, got), (got, want)):
        for k in torch.nonzero(sx > high).flatten().tolist():
            partners = (sy > low) & (ly == lx[k])
            if not bool(partners.any()) or float(box_iou(bx[k:k + 1], by[partners]).max()) <= 0.5:
                n += 1
    return n


def _matched(want, got, floor=0.3):
    """Each torch detection above *floor* against the graph's same-class detection of highest
    IoU: the largest box and score difference, and how many found no partner at all.

    Matched rather than compared by rank: the decode sorts 300 (query, class) pairs by score, and
    two near-tied scores that trade places under a 1e-5 change would read as a box hundreds of
    pixels off when nothing moved."""
    box = score = 0.0
    missing = 0
    for k in torch.nonzero(want[2] > floor).flatten().tolist():
        same = torch.nonzero(got[0] == want[0][k]).flatten()
        if not len(same):
            missing += 1
            continue
        j = same[box_iou(want[1][k:k + 1], got[1][same])[0].argmax()]
        box = max(box, float((want[1][k] - got[1][j]).abs().max()))
        score = max(score, float((want[2][k] - got[2][j]).abs()))
    return box, score, missing


def _check(model, path: Path, size, fp16: bool) -> None:
    """Compare the graph against the torch module through the vendor's decode, image by image."""
    runner = OnnxRunner(path, device="cpu")

    def graph(batch):
        found = dict(zip(runner.outputs, runner(batch.numpy())))
        return {name: torch.from_numpy(found[name]).float() for name in OUTPUTS}

    problems = []
    for image_path in fixtures():
        image = load_image(str(image_path))
        with torch.no_grad():
            want = _detect(model, image, size)
        got = _detect(graph, image, size)
        name = image_path.name
        if fp16:
            n = _changes(want, got)
            print(f"    {name:<20} {len(want[2])} / {len(got[2])} detections, {n} confident changes")
            if n:
                problems.append(f"{name}: {n} confident detections changed in fp16")
            continue
        box, score, missing = _matched(want, got)
        print(f"    {name:<20} {len(want[0])} detections, boxes {box:.2g} px, scores {score:.2g}, "
              f"{missing} unmatched")
        if missing or box > BOX_TOLERANCE or score > SCORE_TOLERANCE:
            problems.append(f"{name}: boxes {box:g} px, scores {score:g}, {missing} unmatched")
    if problems:
        raise SystemExit("not published, the graph disagrees with torch:\n  " + "\n  ".join(problems))


def export_variant(variant: str, revision: str, weights_dir: Path, fp16: bool) -> None:
    directory = weights_dir / FAMILY / variant / revision
    checkpoint = directory / "torch-fp32.pth"
    if not checkpoint.is_file():
        raise SystemExit(f"{checkpoint} is missing. Run tools/fetch/{FAMILY}.py {variant} first.")
    if fp16 and not torch.cuda.is_available():
        raise SystemExit("the fp16 graph is exported on CUDA, under fp16 autocast")
    model, size = load(variant, checkpoint), get_spec(variant).size
    key = "onnx-fp16" if fp16 else "onnx-fp32"
    print(f"{FAMILY}/{variant} -> {key}")
    with tempfile.TemporaryDirectory(dir=directory) as staging:
        staged = Path(staging) / f"{key}.onnx"
        _export(model, staged, size, fp16)
        _check(model, staged, size, fp16)
        staged.replace(directory / f"{key}.onnx")
    print(f"  wrote {key}.onnx, {(directory / f'{key}.onnx').stat().st_size / 1e6:.1f} MB")


def main() -> int:
    parser = variant_parser(__doc__, ROOT / "weights", required=True, revision=REVISION)
    parser.add_argument("--fp16", action="store_true", help="export the fp16 graph instead")
    args = parser.parse_args()
    for variant in args.variants:
        export_variant(variant, args.revision, args.weights_dir, args.fp16)
    print("\nRun tools/generate_manifest.py next.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Export OmDet-Turbo's image path to ONNX and drop the graphs into the local ``weights/`` tree.

Bootstrap tooling; runs once, on a machine you control, and never ships.

    python tools/export/omdet_turbo.py                    # onnx-fp32, and onnx-fp16 on CUDA
    python tools/export/omdet_turbo.py --coco /data/coco  # check on COCO images too

The graph is the image path only -- pixels plus the text features in, boxes and logits out
(``omdet_turbo_deploy/export.py``) -- because the text tower runs once per vocabulary. Users get
the graph and the checkpoint beside it, for the text tower.

``onnx-fp32``  exported on the CPU. The source TensorRT fp32 engines are built from, and what
               ONNX Runtime runs. Checked through ONNX Runtime on the CPU against the torch
               model's own detections: same count, same classes, boxes and scores within
               :data:`BOX_TOLERANCE_PX` and :data:`SCORE_TOLERANCE`.
``onnx-fp16``  exported on CUDA under fp16 autocast, so a strongly typed TensorRT engine runs
               autocast's precision split. Checked through ONNX Runtime on the CPU: it may be no
               farther from fp32 than torch autocast itself is, times :data:`FP16_MARGIN`.

An export that fails its check is not written.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from common import fixtures, require_one_file  # noqa: E402
from mozo.image import load_image  # noqa: E402
from mozo.adapters.omdet_turbo import onnx_forward  # noqa: E402
from mozo.runtimes import make_runner  # noqa: E402
from mozo.vendors.omdet_turbo_deploy import SPECS, THRESHOLD, Detector  # noqa: E402
from mozo.vendors.omdet_turbo_deploy.checkpoint import build  # noqa: E402
from mozo.vendors.omdet_turbo_deploy.export import export  # noqa: E402
from mozo.weights import resolve  # noqa: E402

#: Measured: torch and the fp32 graph through ONNX Runtime agree to 0.0004 px and 2e-06 on the
#: fixture's detections. Wide enough for float reordering, two orders below a real change.
BOX_TOLERANCE_PX = 0.2
SCORE_TOLERANCE = 0.001

#: How much farther from fp32 the fp16 graph may be than torch autocast is, with floors for when
#: autocast lands almost exactly.
FP16_MARGIN = 2.0
FP16_BOX_FLOOR_PX = 2.0
FP16_SCORE_FLOOR = 0.01

VOCABULARIES = (["person", "car", "dog"], ["cat", "remote", "bicycle", "potted plant"])


def _images(coco: Path | None, count: int) -> list[tuple[str, np.ndarray]]:
    found = [(path.name, load_image(str(path))) for path in fixtures()]
    if coco is not None:
        files = sorted((coco / "val2017").glob("*.jpg"))
        found += [(path.name, load_image(str(path)))
                  for path in random.Random(0).sample(files, count)]
    return found


def _drift(want, got) -> tuple[int, float, float]:
    """Unmatched detections both ways (same class, IoU > 0.5), and the largest box and score
    moves among the matched."""
    from torchvision.ops import box_iou

    (wb, ws, wl), (gb, gs, gl) = want, got
    if not len(ws) or not len(gs):
        return len(ws) + len(gs), 0.0, 0.0
    overlap = box_iou(wb, gb) * (wl[:, None] == gl[None])
    best, match = overlap.max(1)
    kept = best > 0.5
    unmatched = int((~kept).sum()) + max(0, len(gs) - int(kept.sum()))
    if not kept.any():
        return unmatched, 0.0, 0.0
    return (unmatched, float((wb[kept] - gb[match[kept]]).abs().max()),
            float((ws[kept] - gs[match[kept]]).abs().max()))


def _stage(image, destination: Path, images, fp16: bool, check) -> None:
    """Export *image* to a scratch file, run *check* on every image and vocabulary against the
    graph through ONNX Runtime on the CPU, and only then place it at *destination*."""
    with tempfile.TemporaryDirectory() as scratch:
        staged = Path(scratch) / destination.name
        export(image, staged, fp16=fp16)
        require_one_file(staged)
        graph = onnx_forward(make_runner(staged, destination.stem, device="cpu"))
        for name, pixels in images:
            for vocabulary in VOCABULARIES:
                print(f"  {name[:22]:<22} {vocabulary[0]:<6} " + check(name, pixels, vocabulary,
                                                                     graph))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(staged, destination)
    print(f"  {destination.stem} {destination.stat().st_size / 1e6:.1f} MB")


def export_fp32(reference: Detector, destination: Path, images) -> None:
    def check(name, pixels, vocabulary, graph):
        want = reference(pixels, vocabulary, THRESHOLD)
        # Matched by class and IoU, so a near-tie ordered differently is not a failure.
        lost, box, score = _drift(want, reference.with_forward(graph)(pixels, vocabulary,
                                                                     THRESHOLD))
        if lost or box > BOX_TOLERANCE_PX or score > SCORE_TOLERANCE:
            raise SystemExit(f"{name}: {lost} detections unmatched, boxes moved {box:.4f} px, "
                             f"scores {score:.6f}. Not writing it.")
        return f"{len(want[0]):>3} detections   boxes {box:.4f} px   scores {score:.2e}"

    _stage(reference.image, destination, images, False, check)


def export_fp16(reference: Detector, checkpoint: Path, destination: Path, images) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("the fp16 graph is exported on CUDA, under fp16 autocast")
    # The autocast baseline is compared in IEEE fp32 outside its fp16 casts, as rfdetr's is.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    half, _ = build(SPECS["tiny"], checkpoint, device="cuda")

    def autocast(pixels, classes, task):
        with torch.autocast("cuda", torch.float16):
            return half(pixels.cuda(), classes.cuda(), task.cuda())

    torch_fp16 = reference.with_forward(autocast)

    def check(name, pixels, vocabulary, graph):
        want = reference(pixels, vocabulary, THRESHOLD)
        ac_lost, ac_box, ac_score = _drift(want, torch_fp16(pixels, vocabulary, THRESHOLD))
        lost, box, score = _drift(want, reference.with_forward(graph)(pixels, vocabulary,
                                                                     THRESHOLD))
        box_limit = max(FP16_MARGIN * ac_box, FP16_BOX_FLOOR_PX)
        score_limit = max(FP16_MARGIN * ac_score, FP16_SCORE_FLOOR)
        if lost > ac_lost or box > box_limit or score > score_limit:
            raise SystemExit(
                f"{name}: the fp16 graph is farther from fp32 than torch autocast -- "
                f"{lost} unmatched against {ac_lost}, boxes {box:.2f} px (limit "
                f"{box_limit:.2f}), scores {score:.4f} (limit {score_limit:.4f}). Not writing it.")
        return (f"{len(want[0]):>3} detections   unmatched {lost} ({ac_lost})   boxes {box:.2f} px "
                f"({ac_box:.2f})   scores {score:.4f} ({ac_score:.4f})   [autocast in brackets]")

    _stage(half, destination, images, True, check)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--revision", default=None, help="published revision (default: latest)")
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "weights")
    parser.add_argument("--coco", type=Path, help="a COCO root holding val2017/, to check on too")
    parser.add_argument("--images", type=int, default=10, help="COCO images to check on")
    parser.add_argument("--only", choices=("fp32", "fp16"), help="export one graph only")
    args = parser.parse_args()

    checkpoint = Path(resolve("omdet_turbo", "tiny", "torch-fp32", revision=args.revision))
    revision_dir = args.weights_dir / "omdet_turbo" / "tiny" / checkpoint.parent.name
    images = _images(args.coco, args.images)
    # Detections compared at upstream's default threshold, on one CPU reference.
    reference = Detector(checkpoint, SPECS["tiny"])
    if args.only != "fp16":
        print("=== omdet_turbo/tiny onnx-fp32")
        export_fp32(reference, revision_dir / "onnx-fp32.onnx", images)
    if args.only != "fp32":
        print("=== omdet_turbo/tiny onnx-fp16")
        export_fp16(reference, checkpoint, revision_dir / "onnx-fp16.onnx", images)
    print("\nRun tools/generate_manifest.py to pick these up.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

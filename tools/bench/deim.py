#!/usr/bin/env python3
"""Measure DEIM's runtimes against each other: speed, COCO AP, and what changes against torch CPU.

    python tools/bench/deim.py --coco /data/coco --variant small
    python tools/bench/deim.py --coco /data/coco --count 100        # every variant, 100 images

``--coco`` is a folder holding ``val2017/`` and ``annotations/instances_val2017.json``. Bootstrap
tooling; never ships. Needs ``pycocotools``; ``onnxruntime`` (or ``onnxruntime-gpu`` for the CUDA
row) and TensorRT 11+ for their rows, which are skipped when absent.

Every runtime is fed the vendor's preprocessing and decoded by the vendor's post-processor, so
the rows differ only in the forward pass. Per runtime:

- **ms**: the forward pass alone on one 640x640 image, median, the device synchronised;
- **AP**: COCO box AP@[.50:.95] over the sampled images, all 300 detections per image (upstream's
  evaluation protocol on a subset, so not comparable with the README's full-val numbers);
- **changes**: detections above 0.52 on one side with no same-class partner above 0.5 at IoU over
  0.5 on the other, against torch fp32 on the CPU (the path ``tools/verify/deim.py`` holds equal
  to upstream). The margin keeps a score crossing one threshold from counting.

CUDA rows run with TF32 off, so torch and the engines are IEEE fp32 where they say fp32.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import box_iou

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MOZO_BASE_URL", f"file://{ROOT / 'weights'}")

from mozo.image import load_image  # noqa: E402
from mozo.registry import DERIVES  # noqa: E402
from mozo.vendors.deim_deploy import SPECS, load, postprocess, preprocess  # noqa: E402
from mozo.weights import parts, resolve  # noqa: E402

FAMILY = "deim"
OUTPUTS = ("pred_logits", "pred_boxes")


def sample(coco: Path, count: int) -> list[Path]:
    files = sorted((coco / "val2017").glob("*.jpg"))
    rng = np.random.default_rng(0)
    return [files[i] for i in sorted(rng.choice(len(files), count, replace=False))]


def median_ms(forward, batch, runs: int, cuda: bool) -> float:
    for _ in range(3):
        forward(batch)
    times = []
    for _ in range(runs):
        if cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        forward(batch)
        if cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1e3)
    return statistics.median(times)


def detections(forward, batches):
    found = []
    for batch, size in batches:
        raw = {k: v.float().cpu() for k, v in forward(batch).items()}
        found.append([t[0] for t in postprocess(raw, size)])
    return found


def coco_ap(coco: Path, ids: list[int], found) -> float:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        truth = COCO(str(coco / "annotations" / "instances_val2017.json"))
        categories = sorted(truth.getCatIds())
        rows = [{"image_id": i, "category_id": categories[int(label)],
                 "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score)}
                for i, (labels, boxes, scores) in zip(ids, found)
                for label, (x1, y1, x2, y2), score in zip(labels, boxes.tolist(), scores)]
        evaluation = COCOeval(truth, truth.loadRes(rows), "bbox")
        evaluation.params.imgIds = ids
        evaluation.evaluate()
        evaluation.accumulate()
        evaluation.summarize()
    return round(evaluation.stats[0] * 100, 2)


def changes(want, got, high=0.52, low=0.5) -> int:
    n = 0
    for a, b in zip(want, got):
        for (lx, bx, sx), (ly, by, sy) in ((a, b), (b, a)):
            for k in torch.nonzero(sx > high).flatten().tolist():
                partners = (sy > low) & (ly == lx[k])
                if not bool(partners.any()) or float(box_iou(bx[k:k + 1], by[partners]).max()) <= 0.5:
                    n += 1
    return n


def runtimes(variant: str):
    """(name, forward, device) for every runtime this machine can run, torch CPU first."""
    checkpoint = resolve(FAMILY, variant, "torch-fp32")
    model = load(variant, checkpoint)
    yield "torch-cpu", lambda b: model(b), "cpu"
    try:
        import onnxruntime as ort

        graph = str(resolve(FAMILY, variant, "onnx-fp32"))
        cpu = ort.InferenceSession(graph, providers=["CPUExecutionProvider"])
        yield "onnx-fp32 cpu", lambda b: dict(zip(OUTPUTS, map(torch.from_numpy, cpu.run(list(OUTPUTS), {"images": b.numpy()})))), "cpu"
        if torch.cuda.is_available() and "CUDAExecutionProvider" in ort.get_available_providers():
            gpu = ort.InferenceSession(graph, providers=[("CUDAExecutionProvider", {"use_tf32": 0})])
            yield "onnx-fp32 cuda", lambda b: dict(zip(OUTPUTS, map(torch.from_numpy, gpu.run(list(OUTPUTS), {"images": b.numpy()})))), "cuda"
    except ImportError:
        pass
    if not torch.cuda.is_available():
        return
    on_gpu = load(variant, checkpoint).cuda()
    yield "torch-cuda", lambda b: on_gpu(b.cuda()), "cuda"
    from mozo.runtimes import TensorRTRunner, _tensorrt_unusable

    if _tensorrt_unusable() is None:
        for key in ("tensorrt-fp32", "tensorrt-fp16"):
            runner = TensorRTRunner(parts(FAMILY, variant, DERIVES[FAMILY][key])[""], "cuda")
            yield key, (lambda r: lambda b: dict(zip(r.outputs, r(b.cuda()))))(runner), "cuda"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--variant", action="append", choices=list(SPECS))
    parser.add_argument("--count", type=int, default=500)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    files = sample(args.coco, args.count)
    ids = [int(f.stem) for f in files]
    batches = [preprocess(load_image(str(f))) for f in files]
    for variant in args.variant or list(SPECS):
        reference = None
        for name, forward, device in runtimes(variant):
            found = detections(forward, batches)
            reference = reference or found
            batch = batches[0][0].cuda() if device == "cuda" else batches[0][0]
            row = {"ms": round(median_ms(forward, batch, 50 if device == "cuda" else 10, device == "cuda"), 2),
                   "ap": coco_ap(args.coco, ids, found), "changes": changes(reference, found)}
            print(json.dumps({"variant": variant, "runtime": name, **row}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

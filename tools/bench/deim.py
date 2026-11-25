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
import copy
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault("MOZO_BASE_URL", f"file://{ROOT / 'weights'}")

from common import median_ms  # noqa: E402
from export.deim import confident_changes  # noqa: E402
from mozo.image import load_image  # noqa: E402
from mozo.registry import DERIVES  # noqa: E402
from mozo.vendors.deim_deploy import OUTPUTS, SPECS, load, postprocess, preprocess  # noqa: E402
from mozo.weights import parts, resolve  # noqa: E402

FAMILY = "deim"


def sample(coco: Path, count: int) -> list[Path]:
    files = sorted((coco / "val2017").glob("*.jpg"))
    rng = np.random.default_rng(0)
    return [files[i] for i in sorted(rng.choice(len(files), count, replace=False))]


def detections(forward, batches):
    found = []
    for batch, size in batches:
        raw = {k: v.float().cpu() for k, v in forward(batch).items()}
        found.append([t[0] for t in postprocess(raw, size)])
    return found


def coco_ap(truth, ids: list[int], found) -> float:
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
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


def changes(want, got) -> int:
    """:func:`export.deim.confident_changes` (the rule the fp16 graph is published under), summed
    over images."""
    return sum(confident_changes(a, b) for a, b in zip(want, got))


def _ort(session):
    """An ONNX Runtime session as a forward: a CPU batch in, named torch outputs back."""
    return lambda b: dict(zip(OUTPUTS, map(torch.from_numpy, session.run(list(OUTPUTS), {"images": b.cpu().numpy()}))))


def runtimes(variant: str):
    """(name, forward, device) for every runtime this machine can run, torch CPU first."""
    checkpoint = resolve(FAMILY, variant, "torch-fp32")
    model = load(variant, checkpoint)
    yield "torch-cpu", lambda b: model(b), "cpu"
    try:
        import onnxruntime as ort

        graph = str(resolve(FAMILY, variant, "onnx-fp32"))
        cpu = ort.InferenceSession(graph, providers=["CPUExecutionProvider"])
        yield "onnx-fp32 cpu", _ort(cpu), "cpu"
        if torch.cuda.is_available() and "CUDAExecutionProvider" in ort.get_available_providers():
            gpu = ort.InferenceSession(graph, providers=[("CUDAExecutionProvider", {"use_tf32": 0})])
            # Fed numpy on the host, so timed from a CPU batch like the CPU rows.
            yield "onnx-fp32 cuda", _ort(gpu), "cpu"
    except ImportError:
        pass
    if not torch.cuda.is_available():
        return
    on_gpu = copy.deepcopy(model).cuda()
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
    from pycocotools.coco import COCO

    with contextlib.redirect_stdout(io.StringIO()):
        truth = COCO(str(args.coco / "annotations" / "instances_val2017.json"))
    files = sample(args.coco, args.count)
    ids = [int(f.stem) for f in files]
    batches = [preprocess(load_image(str(f))) for f in files]
    for variant in args.variant or list(SPECS):
        reference = None
        for name, forward, device in runtimes(variant):
            found = detections(forward, batches)
            reference = reference or found
            batch = batches[0][0].to(device)
            # ``median_ms`` synchronises *device*; the ONNX CUDA row syncs itself (host outputs).
            ms = median_ms(lambda: forward(batch), device, runs=50 if device == "cuda" else 10)
            row = {"ms": round(ms, 2), "ap": coco_ap(truth, ids, found),
                   "changes": changes(reference, found)}
            print(json.dumps({"variant": variant, "runtime": name, **row}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

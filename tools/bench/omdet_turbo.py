#!/usr/bin/env python3
"""Measure OmDet-Turbo's runtimes: COCO AP, agreement with torch fp32, and latency.

Bootstrap tooling; never ships. What ``DERIVES["omdet_turbo"]`` and the family page's runtime
table rest on.

    python tools/bench/omdet_turbo.py --coco /data/coco                       # CUDA runtimes
    python tools/bench/omdet_turbo.py --coco /data/coco --device cpu --n 100 \\
        --runtimes torch-fp32 onnx-fp32

``--coco`` holds ``val2017/`` and ``annotations/instances_val2017.json``. Images are a seeded
shuffle of val2017. Each runtime is built through the public adapter, so what is measured is
what ships; ``floor`` (fp32 with its input rounded to fp16) and ``autocast`` (torch fp16
autocast, which mozo does not offer) are measured beside them for the record.

``AP``       COCO box AP with the 80 class names as the vocabulary (the task sentence truncates at
             77 tokens, as upstream's does), up to 100 detections per image above 0.001.
``changes``  Against the first runtime listed: a detection above 0.32 on one side with no IoU>0.5
             same-class partner above 0.3 on the other, counted both ways, out of the reference's
             detections above 0.3.
``latency``  Medians on the first image: ``forward`` the image path alone with the text cached,
             ``predict`` a whole detector call from an array, with 3 prompts and with the 80 COCO names.

Run CPU runtimes one per process: torch's and ONNX Runtime's thread pools slow each other down.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from common import median_ms  # noqa: E402
from mozo.adapters.omdet_turbo import OmDetTurboPredictor  # noqa: E402
from mozo.image import load_image  # noqa: E402
from mozo.vendors.omdet_turbo_deploy.postprocess import postprocess  # noqa: E402
from mozo.vendors.omdet_turbo_deploy.preprocess import preprocess  # noqa: E402

THRESHOLD, MARGIN = 0.3, 0.02
FEW = ["person", "car", "dog"]


def _build(name: str, device: str):
    """The loaded detector for *name*: a shipped runtime through the public adapter, or one of the
    two record-only variants on the torch image path."""
    if name not in ("floor", "autocast"):
        return OmDetTurboPredictor("tiny", device, runtime=name)._detector
    detector = OmDetTurboPredictor("tiny", device, runtime="torch-fp32")._detector
    model = detector.image
    if name == "floor":
        return detector.with_forward(lambda p, c, t: model(p.half().float(), c, t))

    def autocast(p, c, t):
        with torch.autocast(device, torch.float16):
            return model(p, c, t)
    return detector.with_forward(autocast)


def _unmatched(src, dst) -> int:
    from torchvision.ops import box_iou

    (sb, ss, sl), (db, ds, dl) = src, dst
    mine, theirs = ss > THRESHOLD + MARGIN, ds > THRESHOLD
    if not mine.any() or not theirs.any():
        return int(mine.sum())
    overlap = box_iou(sb[mine], db[theirs]) * (sl[mine][:, None] == dl[theirs][None])
    return int((overlap.max(1).values <= 0.5).sum())


def measure(coco: Path, device: str, names: list[str], n: int, seed: int) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(coco / "annotations" / "instances_val2017.json"))
    cats = gt.loadCats(gt.getCatIds())
    vocabulary, cat_ids = [c["name"] for c in cats], [c["id"] for c in cats]
    ids = sorted(gt.getImgIds())
    random.Random(seed).shuffle(ids)
    ids = ids[:n]

    def path(img_id):
        return str(coco / "val2017" / gt.loadImgs(img_id)[0]["file_name"])

    report = {"device": device, "images": n, "seed": seed, "torch": torch.__version__,
              "runtimes": {}}
    if device == "cuda":
        report["gpu"] = torch.cuda.get_device_name()
    raw: dict[str, dict] = {}
    first = load_image(path(ids[0]))
    for name in names:
        start = time.perf_counter()
        detector = _build(name, device)
        loaded = time.perf_counter() - start
        runs = 5 if device == "cpu" else 30
        latency = {"load_s": loaded}
        with torch.inference_mode():
            pixels = preprocess(first).to(device)
            for label, prompts in (("few", FEW), ("coco", vocabulary)):
                classes, task = detector.encode_text(prompts)
                latency[f"forward_{label}_ms"] = median_ms(
                    lambda: detector.forward(pixels, classes, task), device, runs=runs, warmup=2)
                latency[f"predict_{label}_ms"] = median_ms(
                    lambda: detector(first, prompts), device, runs=runs, warmup=2)
            raw[name] = {}
            for img_id in ids:
                image = load_image(path(img_id))
                boxes, logits = detector.raw(image, vocabulary)
                raw[name][img_id] = (boxes.cpu(), logits.cpu(), image.shape[:2])
        report["runtimes"][name] = latency
        print(f"  {name}: {json.dumps({k: round(v, 1) for k, v in latency.items()})}", flush=True)

    reference = names[0]
    for name in names:
        coco_rows, changes, count = [], 0, 0
        for img_id, (boxes, logits, size) in raw[name].items():
            b, s, labels = postprocess(boxes, logits, size, 0.001)
            for k in s.argsort(descending=True)[:100].tolist():
                x1, y1, x2, y2 = b[k].tolist()
                coco_rows.append({"image_id": img_id, "category_id": cat_ids[int(labels[k])],
                                  "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(s[k])})
            mine = postprocess(boxes, logits, size, THRESHOLD)
            theirs = postprocess(*raw[reference][img_id], THRESHOLD)
            count += len(theirs[1])
            if name != reference:
                changes += _unmatched(theirs, mine) + _unmatched(mine, theirs)
        with contextlib.redirect_stdout(io.StringIO()):
            evaluation = COCOeval(gt, gt.loadRes(coco_rows), "bbox")
            evaluation.params.imgIds = ids
            evaluation.evaluate(), evaluation.accumulate(), evaluation.summarize()
        report["runtimes"][name].update(AP=round(100 * evaluation.stats[0], 2),
                                        AP50=round(100 * evaluation.stats[1], 2),
                                        changes=changes, of=count)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--runtimes", nargs="+",
                        default=["torch-fp32", "floor", "autocast", "tensorrt-fp32",
                                 "tensorrt-fp16"])
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, help="also write the report here")
    args = parser.parse_args()
    report = measure(args.coco, args.device, args.runtimes, args.n, args.seed)
    print(json.dumps(report, indent=2))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

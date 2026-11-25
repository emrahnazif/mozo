#!/usr/bin/env python3
"""Measure Florence-2 detection: COCO accuracy per runtime, fp16 agreement, and latency.

Bootstrap tooling; never ships. ``auto``'s choice on CUDA, the choice of ``large``'s revision, and
the latency rows of the docs page rest on this.

    python tools/bench/florence2_detect.py accuracy --coco /data/coco --variants base large
    python tools/bench/florence2_detect.py accuracy --coco /data/coco --weights large=old.pth
    python tools/bench/florence2_detect.py latency --device cuda --variants base large

``--coco`` holds ``val2017/`` and ``annotations/instances_val2017.json`` (or the json beside
``val2017``). ``--weights VARIANT=PATH`` runs a checkpoint other than the published one with that
variant's geometry (used to compare ``large``'s two Microsoft revisions).

**How a free-vocabulary detector is scored on COCO.** ``<OD>`` names objects in its own words, so a
box counts for a COCO category only when its label *is* that category's name (lower case, exact):
"person" scores, "man" does not. That is harsh on the zero-shot ``base``/``large``, which say "man"
and "human face", and fair to the ``-ft`` checkpoints, which were tuned to COCO's names -- so the
absolute AP is a lower bound for the zero-shot models and the comparison that matters is *between
runtimes of one checkpoint* (and between ``large``'s revisions, scored identically). Scores are
upstream's ``<OD>`` scores; boxes are as returned.

**Agreement** is fp16 against fp32 on the same images: the share of fp32 boxes that fp16 returns
with the same label and IoU >= 0.9, and the share of images whose whole answer is identical.

Everything on CUDA runs with TF32 off: the reference is IEEE fp32.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from common import FIXTURES, synchronise  # noqa: E402
from fetch.florence2_detect import published  # noqa: E402

from mozo.image import load_image  # noqa: E402
from mozo.runtimes import free_device_memory  # noqa: E402
from mozo.vendors.florence2_deploy import VARIANTS, Florence2Predictor  # noqa: E402

DTYPES = {"torch-fp32": torch.float32, "torch-fp16": torch.float16}


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _coco(root: Path):
    for candidate in (root / "annotations" / "instances_val2017.json",
                      root / "instances_val2017.json"):
        if candidate.is_file():
            return candidate
    raise SystemExit(f"no instances_val2017.json under {root}")


def accuracy(args) -> None:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(_coco(args.coco)))
    by_name = {c["name"]: c["id"] for c in coco.loadCats(coco.getCatIds())}
    ids = sorted(coco.getImgIds())
    random.Random(0).shuffle(ids)
    ids = ids[:args.images]
    overrides = dict(item.split("=", 1) for item in args.weights)
    results = {}
    for variant in args.variants:
        weights = Path(overrides.get(variant, published(variant)))
        answers = {}
        for runtime in args.runtimes:
            model = Florence2Predictor(weights, variant, args.device, DTYPES[runtime])
            detections, answer, labels, started = [], {}, {}, time.perf_counter()
            for n, image_id in enumerate(ids):
                info = coco.loadImgs(image_id)[0]
                boxes = model.detect(load_image(str(args.coco / "val2017" / info["file_name"])))
                answer[image_id] = boxes
                for box in boxes:
                    labels[box.label] = labels.get(box.label, 0) + 1
                    if box.label.lower() in by_name:
                        x1, y1, x2, y2 = box.box
                        detections.append({"image_id": image_id,
                                           "category_id": by_name[box.label.lower()],
                                           "bbox": [x1, y1, x2 - x1, y2 - y1],
                                           "score": box.score})
                if (n + 1) % 100 == 0:
                    print(f"  {variant} {runtime} {n + 1}/{len(ids)}", flush=True)
            seconds = time.perf_counter() - started
            ap = ap50 = float("nan")
            if detections:
                with contextlib.redirect_stdout(io.StringIO()):
                    evaluation = COCOeval(coco, coco.loadRes(detections), "bbox")
                    evaluation.params.imgIds = ids
                    evaluation.evaluate()
                    evaluation.accumulate()
                    evaluation.summarize()
                ap, ap50 = evaluation.stats[0], evaluation.stats[1]
            boxes_total = sum(len(v) for v in answer.values())
            matched = len(detections)
            answers[runtime] = answer
            row = {"AP": round(ap * 100, 2), "AP50": round(ap50 * 100, 2), "boxes": boxes_total,
                   "boxes_with_coco_name": matched, "seconds_per_image": round(seconds / len(ids), 3),
                   "top_labels": sorted(labels.items(), key=lambda kv: -kv[1])[:12]}
            results[f"{variant} {runtime}"] = row
            print(f"{variant:9s} {runtime:10s} AP {row['AP']:5.2f}  AP50 {row['AP50']:5.2f}  "
                  f"{matched}/{boxes_total} boxes carry a COCO name, "
                  f"{row['seconds_per_image']} s/image", flush=True)
            del model
            free_device_memory()
        if "torch-fp32" in answers and "torch-fp16" in answers:
            full, half = answers["torch-fp32"], answers["torch-fp16"]
            kept = total = same = 0
            for image_id in ids:
                same += [(b.label, b.box) for b in full[image_id]] == [
                    (b.label, b.box) for b in half[image_id]]
                for box in full[image_id]:
                    total += 1
                    kept += any(other.label == box.label and _iou(box.box, other.box) >= 0.9
                                for other in half[image_id])
            agreement = {"boxes_kept": round(kept / max(total, 1) * 100, 2),
                         "images_identical": round(same / len(ids) * 100, 2)}
            results[f"{variant} agreement"] = agreement
            print(f"{variant:9s} fp16 vs fp32: {agreement['boxes_kept']}% of fp32 boxes kept "
                  f"(same label, IoU>=0.9); {agreement['images_identical']}% of images identical",
                  flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


def _reference(model_dir: str, device: str, dtype):
    """Microsoft's code, as the gate loads it, returning a callable that detects on an array."""
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, torch_dtype=dtype,
        attn_implementation="eager").eval().to(device)
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

    @torch.inference_mode()
    def detect(image):
        inputs = processor(text="<OD>", images=Image.fromarray(image), return_tensors="pt")
        return model.generate(input_ids=inputs["input_ids"].to(device),
                              pixel_values=inputs["pixel_values"].to(device, dtype),
                              max_new_tokens=1024)
    return detect


def latency(args) -> None:
    """Median of ``--runs`` warm, synchronised calls on the example photograph.

    With ``--reference VARIANT=DIR`` the reference runs too, interleaved call by call with mozo's
    vendored path so drift moves both alike.
    """
    image = load_image(str(FIXTURES / "example.jpg"))
    references = dict(item.split("=", 1) for item in args.reference)
    for variant in args.variants:
        for runtime in args.runtimes:
            if runtime == "torch-fp16" and args.device != "cuda":
                continue
            sides = {"mozo": Florence2Predictor(published(variant), variant, args.device,
                                                DTYPES[runtime]).detect}
            if variant in references:
                sides["reference"] = _reference(references[variant], args.device, DTYPES[runtime])
            for work in sides.values():
                for _ in range(2):
                    work(image)
            times = {side: [] for side in sides}
            for _ in range(args.runs):
                for side, work in sides.items():
                    synchronise(args.device)
                    started = time.perf_counter()
                    work(image)
                    synchronise(args.device)
                    times[side].append(time.perf_counter() - started)
            print(f"{variant:9s} {runtime:10s} {args.device}: " + ", ".join(
                f"{side} {statistics.median(t) * 1e3:.1f} ms" for side, t in times.items())
                  + f" (median of {args.runs})", flush=True)
            del sides
            free_device_memory()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("accuracy", "latency"))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--runtimes", nargs="+", default=list(DTYPES), choices=list(DTYPES))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--coco", type=Path)
    parser.add_argument("--images", type=int, default=500)
    parser.add_argument("--weights", nargs="*", default=[], metavar="VARIANT=PATH")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--reference", nargs="*", default=[], metavar="VARIANT=DIR",
                        help="latency: also time Microsoft's code from this snapshot")
    parser.add_argument("--out", help="write the accuracy rows as JSON here")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.mode == "accuracy":
        if args.coco is None:
            raise SystemExit("accuracy needs --coco")
        accuracy(args)
    else:
        latency(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

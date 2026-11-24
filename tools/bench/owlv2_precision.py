#!/usr/bin/env python3
"""Measure OWLv2's reduced-precision runtimes against fp32: accuracy, agreement and latency.

Bootstrap tooling; never ships. This is what ``auto`` choosing ``torch-fp16`` on CUDA rests on,
and what ``tensorrt-fp16``'s documented drift was measured with.

    python tools/bench/owlv2_precision.py --coco /data/coco --variants base-ensemble large
    python tools/bench/owlv2_precision.py --coco /data/coco --runtimes torch-fp16 tensorrt-fp16

``--coco`` is a directory holding ``val2017/`` and ``instances_val2017.json``.

**Three questions, kept apart.**

``accuracy``   COCO box AP per runtime, the 80 class names verbatim as the vocabulary, the top 100
               patches per image. If a runtime's drift is real but AP does not move, the drift
               is not costing accuracy.
``agreement``  The same patch on both sides -- OWLv2 predicts one box per patch, so no matching
               heuristic is needed. Score and box drift on the patches that matter, label flips,
               and threshold crossings, split into those within 0.02 of the threshold (noise)
               and those clear of it (behaviour a user would see change).
``latency``    The image half alone, median of 100 warm, synchronised runs on one image. The text
               half is cached per vocabulary and the heads are shared, so this is the number a
               runtime changes.

**Why there is a noise floor row.** OWLv2 is unstable on a few patches even in fp32: rounding only
the *input* to fp16, an invisible change, moves one base-ensemble box 38.7 px. A box head choosing
between two plausible boxes -- one object or the group it stands in -- tips over under any tiny
perturbation. So the largest box shift says nothing about a runtime; how often it changes a
detection, compared with ``floor``, is the measure. ``floor`` is fp32 with its input rounded.

Everything runs with TF32 off: the reference is IEEE fp32, so a runtime's drift is its own.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from common import median_ms  # noqa: E402
from mozo.image import load_image  # noqa: E402
from mozo.vendors.owlv2_deploy.image import preprocess, to_original  # noqa: E402
from mozo.vendors.owlv2_deploy.predictor import Detector  # noqa: E402
from mozo.weights import resolve  # noqa: E402

THRESHOLDS = (0.1, 0.2, 0.3)
MARGIN = 0.02
TOPK = 100
RUNTIMES = ("floor", "torch-fp16", "tensorrt-fp16")


def _encoders(variant: str, runtimes: list[str]):
    """The fp32 reference detector, and runtime name -> fn(pixels) -> patches, fp32 first."""
    checkpoint = resolve("owlv2", variant, "torch-fp32")
    reference = Detector(checkpoint, variant, device="cuda")
    out = {"torch-fp32": reference.encode_pixels}
    if "floor" in runtimes:
        out["floor"] = lambda px: reference.encode_pixels(px.half().float())
    if "torch-fp16" in runtimes:
        out["torch-fp16"] = Detector(checkpoint, variant, device="cuda",
                                     half_trunk=True).encode_pixels
    if "tensorrt-fp16" in runtimes:
        from mozo.adapters.owlv2 import OwlV2Predictor

        out["tensorrt-fp16"] = OwlV2Predictor(
            variant, device="cuda", runtime="tensorrt-fp16")._detector.encode_pixels
    return reference, out


def measure(variant: str, coco: Path, n: int, seed: int, runtimes: list[str]) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(coco / "instances_val2017.json"))
    cats = gt.loadCats(gt.getCatIds())
    names, cat_ids = [c["name"] for c in cats], [c["id"] for c in cats]
    img_ids = sorted(gt.getImgIds())
    random.Random(seed).shuffle(img_ids)
    img_ids = img_ids[:n]

    reference, encoders = _encoders(variant, runtimes)
    model, size = reference.model, reference.image_size
    queries, query_mask = reference.encode_text(names)
    torch.cuda.set_stream(torch.cuda.Stream())

    def load(img_id):
        return load_image(str(coco / "val2017" / gt.loadImgs(img_id)[0]["file_name"]))

    def candidates(patches, shape):
        logits, boxes, _ = model.detect(patches, queries, query_mask)
        best = logits[0].max(dim=-1)
        return (to_original(boxes[0], shape).cpu(), torch.sigmoid(best.values).cpu(),
                best.indices.cpu())

    stats = {name: {"coco": [], "sd": [], "bd": [], "big": 0, "flips": 0,
                    "cross": {t: [0, 0, 0] for t in THRESHOLDS}} for name in encoders}
    with torch.no_grad():
        px0 = preprocess(load(img_ids[0]), size).cuda()
        latency = {name: median_ms(lambda fn=fn: fn(px0), "cuda", runs=100, warmup=10)
                   for name, fn in encoders.items()}

        for k, img_id in enumerate(img_ids):
            image = load(img_id)
            shape = image.shape[:2]
            px = preprocess(image, size).cuda()
            got = {name: candidates(fn(px), shape) for name, fn in encoders.items()}
            rb, rs, rl = got["torch-fp32"]
            for name, (gb, gs, gl) in got.items():
                s = stats[name]
                for i in gs.topk(TOPK).indices.tolist():
                    x1, y1, x2, y2 = gb[i].tolist()
                    s["coco"].append({"image_id": img_id, "category_id": cat_ids[int(gl[i])],
                                      "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(gs[i])})
                if name == "torch-fp32":
                    continue
                s["sd"].append((gs - rs)[(rs > 0.1) | (gs > 0.1)].abs())
                kept = rs > 0.1
                shift = (gb[kept] - rb[kept]).abs().max(dim=-1).values
                s["bd"].append(shift)
                s["big"] += int((shift > 10).sum())
                s["flips"] += int((gl[kept] != rl[kept]).sum())
                for t in THRESHOLDS:
                    crossed, near = (rs > t) != (gs > t), (rs - t).abs() < MARGIN
                    s["cross"][t][0] += int((crossed & ~near).sum())
                    s["cross"][t][1] += int((crossed & near).sum())
                    s["cross"][t][2] += int((rs > t).sum())
            if (k + 1) % 100 == 0:
                print(f"  {variant}: {k + 1}/{len(img_ids)}", flush=True)

    report = {"variant": variant, "images": len(img_ids), "seed": seed, "image_size": size,
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "runtimes": {}}
    for name, s in stats.items():
        with contextlib.redirect_stdout(io.StringIO()):
            ev = COCOeval(gt, gt.loadRes(s["coco"]), "bbox")
            ev.params.imgIds = img_ids
            ev.evaluate(), ev.accumulate(), ev.summarize()
        r = {"image_ms": latency[name], "speedup": latency["torch-fp32"] / latency[name],
             **dict(zip(("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"),
                        map(float, ev.stats[:6])))}
        if s["sd"]:
            sd, bd = torch.cat(s["sd"]).numpy(), torch.cat(s["bd"]).numpy()
            r.update(score_p99=float(np.percentile(sd, 99)),
                     box_p99_px=float(np.percentile(bd, 99)), box_max_px=float(bd.max()),
                     boxes_over_10px=s["big"], label_flips=s["flips"],
                     crossings={str(t): dict(zip(("clear", "near", "detections"), v))
                                for t, v in s["cross"].items()})
        report["runtimes"][name] = r
    return report


def _print(report: dict) -> None:
    print(f"\n{report['variant']} ({report['images']} images, {report['image_size']}px, "
          f"{report['gpu']})")
    print(f"{'runtime':14s} {'ms':>7s} {'x':>5s} {'AP':>6s} {'AP50':>6s} {'sc p99':>7s} "
          f"{'>10px':>6s} {'clear@.3':>9s} {'near@.3':>8s} {'clear@.1':>9s}")
    for name, r in report["runtimes"].items():
        line = (f"{name:14s} {r['image_ms']:7.1f} {r['speedup']:5.2f} {100 * r['AP']:6.2f} "
                f"{100 * r['AP50']:6.2f}")
        if "crossings" in r:
            c1, c3 = r["crossings"]["0.1"], r["crossings"]["0.3"]
            line += (f" {r['score_p99']:7.4f} {r['boxes_over_10px']:6d} {c3['clear']:9d} "
                     f"{c3['near']:8d} {c1['clear']:9d}")
        print(line)
    c = next(r["crossings"] for r in report["runtimes"].values() if "crossings" in r)
    print(f"detections in fp32: {c['0.1']['detections']} above 0.1, "
          f"{c['0.3']['detections']} above 0.3")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--coco", type=Path, required=True)
    ap.add_argument("--variants", nargs="+", default=["base-ensemble"])
    ap.add_argument("--runtimes", nargs="+", default=["floor", "torch-fp16"], choices=RUNTIMES)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, help="write every report here as JSON")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("these runtimes are CUDA-only")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    reports = []
    for variant in args.variants:
        reports.append(measure(variant, args.coco, args.n, args.seed, args.runtimes))
        _print(reports[-1])
        torch.cuda.empty_cache()
    if args.out:
        args.out.write_text(json.dumps(reports, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""PP-OCRv6 across runtimes: speed, agreement with torch fp32, and accuracy on ICDAR 2015.

    python tools/bench/ppocrv6.py medium --runtime torch-fp32 --device cpu --images DIR --gt DIR
    python tools/bench/ppocrv6.py medium --runtime tensorrt-fp16 --device cuda --images DIR --gt DIR \\
        --graphs weights/ppocrv6/medium/2026-11-25 --against bench/ppocrv6-medium-torch-fp32-cpu.json

Every runtime runs mozo's own pipeline (``Reader``) with only the two network calls swapped, so
what differs between rows is the graph and nothing around it. Each run writes its predictions to
``bench/``; ``--against`` compares them line for line with another run's.

**Accuracy** is measured on the ICDAR 2015 incidental-scene-text test set (500 images; the
images and ``gt_img_*.txt`` from the Robust Reading Competition, which are not redistributed
here). Two numbers: detection precision/recall/hmean at IoU 0.5 with don't-care regions ignored,
and *word recall* -- the share of ground-truth words that some predicted line covers by at least
half and contains as a token (lowercase, edge punctuation stripped). PP-OCR reads lines and the
ground truth is words, so the detection hmean is low by construction; both numbers are for
comparing runtimes with each other, not with published leaderboards.

**Timing** is per page, median: ``graphs`` is the two networks alone (device synchronised),
``page`` the whole call including OpenCV.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from mozo.image import load_image  # noqa: E402
from mozo.vendors.ppocrv6_deploy import SPECS, Reader  # noqa: E402
from mozo.weights import resolve  # noqa: E402

#: TensorRT optimisation profiles: (min, preferred, max). The detector's maximum is a
#: 2048-pixel page, the recogniser's a batch of six 3200-pixel lines.
TRT_SHAPES = {
    "det": {"pixels": ((1, 3, 32, 32), (1, 3, 736, 1280), (1, 3, 2048, 2048))},
    "rec": {"pixels": ((1, 3, 48, 320), (6, 3, 48, 640), (6, 3, 48, 3200))},
}


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def build_reader(variant: str, runtime: str, device: str, graphs: Path | None) -> Reader:
    spec = SPECS[variant]
    checkpoint = resolve("ppocrv6", variant, "torch-fp32")
    if runtime == "torch-fp32":
        return Reader(checkpoint, spec, device=device)
    precision = runtime.split("-")[1]
    paths = {kind: graphs / f"onnx-{precision}-{kind}.onnx" for kind in ("det", "rec")}
    if runtime.startswith("onnx"):
        import onnxruntime

        from mozo.runtimes import OnnxRunner

        if device.startswith("cuda") and hasattr(onnxruntime, "preload_dlls"):
            # onnxruntime-gpu 1.23 is built against CUDA 12; with a CUDA 13 torch the CUDA 12
            # libraries come from the nvidia-*-cu12 wheels, which this loads.
            onnxruntime.preload_dlls()
        det, rec = (OnnxRunner(paths[k], device=device) for k in ("det", "rec"))
        return Reader(None, spec, device=device,
                      detect=lambda x: det(x)[0].astype(np.float32),
                      recognise=lambda x: rec(x)[0].astype(np.float32))
    from mozo.runtimes import TensorRTRunner

    engines = graphs / "engines"
    det = TensorRTRunner(paths["det"], device, engines=engines, shapes=TRT_SHAPES["det"])
    rec = TensorRTRunner(paths["rec"], device, engines=engines, shapes=TRT_SHAPES["rec"])

    def run(engine):
        return lambda x: engine(torch.from_numpy(x))[0].float().cpu().numpy()

    return Reader(None, spec, device=device, detect=run(det), recognise=run(rec))


def run(reader: Reader, images: list[Path], device: str) -> tuple[dict, dict]:
    graph_ms: list[float] = []
    page_ms: list[float] = []
    spent = [0.0]

    def timed(fn):
        def wrapper(x):
            _sync(device)
            start = time.perf_counter()
            out = fn(x)
            _sync(device)
            spent[0] += time.perf_counter() - start
            return out
        return wrapper

    reader.detect, reader.recognise = timed(reader.detect), timed(reader.recognise)
    predictions = {}
    for i, path in enumerate(images):
        rgb = load_image(path)
        spent[0] = 0.0
        start = time.perf_counter()
        regions = reader(rgb)
        elapsed = time.perf_counter() - start
        if i >= 3:  # warm-up pages are not timed
            graph_ms.append(spent[0] * 1000)
            page_ms.append(elapsed * 1000)
        predictions[path.name] = [[r.quad.tolist(), r.text, r.confidence] for r in regions]
    timing = {"graphs_ms": statistics.median(graph_ms) if graph_ms else None,
              "page_ms": statistics.median(page_ms) if page_ms else None, "pages": len(graph_ms)}
    return predictions, timing


# -- ICDAR 2015 -----------------------------------------------------------------------------------

def load_gt(folder: Path) -> dict:
    out = {}
    for f in sorted(folder.glob("gt_img_*.txt")):
        rows = []
        for line in f.read_text(encoding="utf-8-sig").splitlines():
            parts = line.strip().split(",")
            if len(parts) >= 9:
                rows.append((np.array(list(map(int, parts[:8]))).reshape(4, 2), ",".join(parts[8:])))
        out[f.name[3:-4] + ".jpg"] = rows
    return out


def _areas(a, b) -> tuple[float, float, float]:
    """Intersection, area of a, area of b, by rasterising both quads on one grid."""
    qs = np.concatenate([np.asarray(a, float), np.asarray(b, float)])
    x0, y0 = np.floor(qs.min(0)).astype(int)
    x1, y1 = np.ceil(qs.max(0)).astype(int)
    masks = []
    for q in (a, b):
        m = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
        cv2.fillPoly(m, [np.round(np.asarray(q, float) - [x0, y0]).astype(np.int32)], 1)
        masks.append(m)
    return float((masks[0] & masks[1]).sum()), float(masks[0].sum()), float(masks[1].sum())


def _norm(text: str) -> str:
    return re.sub(r"^[^\w]+|[^\w]+$", "", text.lower())


def icdar(predictions: dict, gt: dict) -> dict:
    tp = n_pred = n_gt = words = hits = 0
    for name, rows in gt.items():
        found = [(np.asarray(q, float), t) for q, t, *_ in predictions.get(name, [])]
        care = [(q, t) for q, t in rows if t != "###"]
        dont = [q for q, t in rows if t == "###"]
        kept = [(q, t) for q, t in found
                if not any((lambda i, ap: ap > 0 and i / ap > 0.5)(*_areas(q, d)[:2]) for d in dont)]
        n_pred += len(kept)
        n_gt += len(care)
        pairs = []
        for gi, (g, _) in enumerate(care):
            for pi, (q, _) in enumerate(kept):
                inter, ag, ap = _areas(g, q)
                if ag + ap - inter > 0 and inter / (ag + ap - inter) >= 0.5:
                    pairs.append((inter / (ag + ap - inter), gi, pi))
        used_g, used_p = set(), set()
        for _, gi, pi in sorted(pairs, reverse=True):
            if gi not in used_g and pi not in used_p:
                used_g.add(gi)
                used_p.add(pi)
                tp += 1
        for g, t in care:
            words += 1
            if any((lambda i, ag: ag > 0 and i / ag >= 0.5)(*_areas(g, q)[:2])
                   and _norm(t) in [_norm(x) for x in pt.split()] for q, pt in found):
                hits += 1
    p, r = tp / max(n_pred, 1), tp / max(n_gt, 1)
    return {"precision": p, "recall": r, "hmean": 2 * p * r / max(p + r, 1e-9),
            "word_recall": hits / max(words, 1)}


def agreement(mine: dict, other: dict) -> dict:
    """Line by line against another run, as a share of *other*'s lines.

    ``identical``: same quad, same string. ``moved``: same string, quad moved (paired by overlap,
    IoU >= 0.5). ``reread``: paired by overlap, string changed. ``unpaired``: lines on either side
    with no partner -- found by one run and not the other. And the largest confidence move among
    identical lines.
    """
    identical = moved = reread = unpaired = total = 0
    worst = 0.0
    for name, rows in other.items():
        theirs = list(rows)
        ours = list(mine.get(name, []))
        total += len(theirs)
        left = []
        for q, t, c in theirs:
            hit = next((k for k, (q2, t2, _) in enumerate(ours) if q2 == q and t2 == t), None)
            if hit is None:
                left.append((q, t))
            else:
                identical += 1
                worst = max(worst, abs(ours.pop(hit)[2] - c))
        for q, t in left:
            best, best_iou = None, 0.5
            for k, (q2, _, _) in enumerate(ours):
                inter, a, b = _areas(q, q2)
                iou = inter / max(a + b - inter, 1.0)
                if iou >= best_iou:
                    best, best_iou = k, iou
            if best is None:
                unpaired += 1
            else:
                moved, reread = (moved + 1, reread) if ours.pop(best)[1] == t else (moved, reread + 1)
        unpaired += len(ours)
    share = lambda n: round(n / max(total, 1), 4)  # noqa: E731
    return {"lines": total, "identical": share(identical), "moved": share(moved),
            "reread": share(reread), "unpaired_lines": unpaired,
            "max_confidence_delta": worst}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variant", choices=list(SPECS))
    parser.add_argument("--runtime", default="torch-fp32",
                        choices=["torch-fp32", "onnx-fp32", "onnx-fp16", "tensorrt-fp32",
                                 "tensorrt-fp16"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--gt", type=Path, help="folder of gt_img_*.txt for accuracy")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--graphs", type=Path, help="folder holding onnx-<precision>-{det,rec}.onnx")
    parser.add_argument("--against", type=Path, help="another run's JSON to compare lines with")
    parser.add_argument("--threads", type=int, help="torch CPU threads")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    images = sorted(args.images.glob("*.jpg"))[:args.limit]
    reader = build_reader(args.variant, args.runtime, args.device, args.graphs)
    predictions, timing = run(reader, images, args.device)
    report = {"variant": args.variant, "runtime": args.runtime, "device": args.device,
              "images": len(images), **timing}
    if args.gt:
        report.update(icdar(predictions, load_gt(args.gt)))
    if args.against:
        report.update(agreement(predictions, json.loads(args.against.read_text())["predictions"]))
    out = ROOT / "bench" / f"ppocrv6-{args.variant}-{args.runtime}-{args.device.split(':')[0]}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"report": report, "predictions": predictions}))
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

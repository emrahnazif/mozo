#!/usr/bin/env python3
"""Measure EoMT's runtimes: speed, and accuracy as the task measures it (PQ, mIoU).

Runs on a machine you control, never ships.

    python tools/bench/eomt.py coco-panoptic-small --coco-panoptic /data/coco --count 500
    python tools/bench/eomt.py ade20k-semantic-large --ade /data/ADEChallengeData2016

Each runtime is built from the files in the local ``weights/`` tree -- the torch checkpoint and the
ONNX graphs ``tools/export/eomt.py`` wrote -- behind the vendor's own pre- and post-processing, so
only the network differs between rows:

``torch-fp32``      the published checkpoint.
``torch-fp16``      the same under CUDA fp16 autocast. For comparison only; not offered.
``onnx-fp32``       ONNX Runtime, on the CPU (and on CUDA where onnxruntime-gpu is installed).
``tensorrt-fp32``   a TensorRT engine from ``onnx-fp32``, TF32 off.
``tensorrt-fp16``   a TensorRT engine from ``onnx-fp16``, strongly typed.

**Speed:** the median of the network alone and of a whole ``segment`` call on the fixture.
**Accuracy:** panoptic quality on COCO val2017 panoptic (``--coco-panoptic``: a directory holding
``panoptic_val2017.json``, ``panoptic_val2017/`` and ``val2017/``), computed as the COCO panoptic
API does; or mIoU on ADE20K validation (``--ade``: ``ADEChallengeData2016``), labels 1-150 with 0
ignored, as upstream's ``MulticlassJaccardIndex(ignore_index=255)`` does after shifting by one. Also
the share of pixels whose class differs from torch fp32 on the same images.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault("MOZO_BASE_URL", f"file://{ROOT / 'weights'}")
sys.stdout.reconfigure(line_buffering=True)

from common import fixtures, synchronise  # noqa: E402
from mozo.image import load_image  # noqa: E402
from mozo.vendors.eomt_deploy import SPECS, Segmenter  # noqa: E402
from mozo.vendors.eomt_deploy.checkpoint import load_network  # noqa: E402
from mozo.weights import resolve  # noqa: E402

RUNTIMES = ("torch-fp32", "torch-fp16", "onnx-fp32", "tensorrt-fp32", "tensorrt-fp16")


def _one_at_a_time(run):
    def network(pixels):
        outs = [run(pixels[i:i + 1].contiguous()) for i in range(len(pixels))]
        return torch.cat([o[0] for o in outs]), torch.cat([o[1] for o in outs])
    return network


def build(variant: str, runtime: str, device: str):
    """``(segmenter, network)`` for *runtime*, or ``None`` where it cannot run here."""
    spec = SPECS[variant]
    checkpoint = resolve("eomt", variant, "torch-fp32")
    # The graphs tools/export/eomt.py wrote into the tree, which may not be published (yet).
    tree = ROOT / "weights" / "eomt" / variant / checkpoint.parent.name
    graph = lambda key: tree / f"{key}.onnx"  # noqa: E731
    if runtime == "torch-fp32":
        network = load_network(spec, checkpoint, device)
    elif runtime == "torch-fp16":
        if device != "cuda":
            return None
        torch_net = load_network(spec, checkpoint, device)

        def network(x):
            with torch.autocast("cuda", torch.float16):
                mask_logits, class_logits = torch_net(x)
            return mask_logits.float(), class_logits.float()
    elif runtime == "onnx-fp32":
        if not graph("onnx-fp32").is_file():
            return None
        from mozo.runtimes import OnnxRunner

        runner = OnnxRunner(graph("onnx-fp32"), device=device)
        network = _one_at_a_time(lambda x: tuple(torch.from_numpy(o).to(x.device)
                                                 for o in runner(x.cpu().numpy())))
    else:
        source = graph("onnx-fp16" if runtime.endswith("fp16") else "onnx-fp32")
        if device != "cuda" or not source.is_file():
            return None
        from mozo.runtimes import TensorRTRunner

        runner = TensorRTRunner(source, device)
        order = [runner.outputs.index(name) for name in ("mask_logits", "class_logits")]

        def run(x):
            outs = runner(x)
            return tuple(outs[i].float() for i in order)
        network = _one_at_a_time(run)
    return Segmenter(None, spec, device=device, network=network), network


def per_pixel(found) -> np.ndarray:
    """The class of every pixel, -1 for void."""
    out = np.full(found.masks.shape[1:], -1, np.int64)
    for i, mask in enumerate(found.masks):
        out[mask] = int(found.class_ids[i])
    return out


# --- panoptic quality, as panopticapi's pq_compute_single_core computes it ---

class PQ:
    """Accumulates per-category TP/FP/FN and IoU sums over images."""

    def __init__(self, categories: dict[int, bool]) -> None:
        self.isthing = categories
        self.stat = {c: [0.0, 0, 0, 0] for c in categories}  # iou, tp, fp, fn

    def add(self, gt_ids: np.ndarray, gt_info: dict[int, dict], pred_ids: np.ndarray,
            pred_cat: dict[int, int]) -> None:
        VOID = 0
        gt_area = {i: int((gt_ids == i).sum()) for i in gt_info}
        pred_area = {i: int((pred_ids == i).sum()) for i in pred_cat}
        pairs, counts = np.unique(gt_ids.astype(np.uint64) * (1 << 32) + pred_ids.astype(np.uint64),
                                  return_counts=True)
        inter = {(int(p >> 32), int(p & 0xFFFFFFFF)): int(c) for p, c in zip(pairs, counts)}
        gt_matched, pred_matched = set(), set()
        for (g, p), n in inter.items():
            if g not in gt_info or p not in pred_cat:
                continue
            if gt_info[g]["iscrowd"] or gt_info[g]["category"] != pred_cat[p]:
                continue
            union = pred_area[p] + gt_area[g] - n - inter.get((VOID, p), 0)
            iou = n / union
            if iou > 0.5:
                s = self.stat[gt_info[g]["category"]]
                s[0] += iou
                s[1] += 1
                gt_matched.add(g)
                pred_matched.add(p)
        crowd = {}
        for g, info in gt_info.items():
            if info["iscrowd"]:
                crowd[info["category"]] = g
            elif g not in gt_matched:
                self.stat[info["category"]][3] += 1
        for p, cat in pred_cat.items():
            if p in pred_matched:
                continue
            ignored = inter.get((VOID, p), 0)
            if cat in crowd:
                ignored += inter.get((crowd[cat], p), 0)
            if ignored / pred_area[p] > 0.5:
                continue
            self.stat[cat][2] += 1

    def result(self) -> dict[str, float]:
        def mean(which):
            values = [s[0] / (s[1] + 0.5 * s[2] + 0.5 * s[3])
                      for c, s in self.stat.items()
                      if (which is None or self.isthing[c] == which) and s[1] + s[2] + s[3] > 0]
            return 100 * sum(values) / len(values) if values else float("nan")
        return {"PQ": mean(None), "PQ_th": mean(True), "PQ_st": mean(False)}


def coco_panoptic(root: Path, count: int, names: list[str]):
    """``(rgb, gt_ids, gt_info)`` for the first *count* val images, and category -> isthing, with
    upstream's contiguous ids matched to COCO's by name."""
    meta = json.loads((root / "panoptic_val2017.json").read_text())
    by_name = {c["name"]: c for c in meta["categories"]}
    contiguous = {by_name[n]["id"]: i for i, n in enumerate(names)}
    categories = {i: bool(by_name[n]["isthing"]) for i, n in enumerate(names)}
    annotations = sorted(meta["annotations"], key=lambda a: a["file_name"])[:count]

    def items():
        for ann in annotations:
            rgb = load_image(str(root / "val2017" / ann["file_name"].replace(".png", ".jpg")))
            png = np.asarray(Image.open(root / "panoptic_val2017" / ann["file_name"]), np.uint32)
            ids = png[..., 0] + 256 * png[..., 1] + 256 * 256 * png[..., 2]
            info = {s["id"]: {"category": contiguous[s["category_id"]], "iscrowd": s["iscrowd"]}
                    for s in ann["segments_info"]}
            yield rgb, ids, info
    return items(), categories, len(annotations)


def ade_semantic(root: Path, count: int):
    paths = sorted((root / "images" / "validation").glob("*.jpg"))[:count]

    def items():
        for path in paths:
            label = np.asarray(Image.open(root / "annotations" / "validation" / (path.stem + ".png")))
            yield load_image(str(path)), label.astype(np.int64) - 1  # 0 (other) -> -1, ignored
    return items(), len(paths)


def accuracy(variant: str, segmenters: dict, args) -> None:
    spec = SPECS[variant]
    names = [entry["name"] for entry in json.loads(resolve("eomt", variant, "labels").read_text())]
    moved = {r: [] for r in segmenters}
    started = time.perf_counter()
    if spec.task == "panoptic":
        items, categories, total = coco_panoptic(args.coco_panoptic, args.count, names)
        scores = {r: PQ(categories) for r in segmenters}
    else:
        items, total = ade_semantic(args.ade, args.count)
        confusion = {r: np.zeros((spec.classes, spec.classes), np.int64) for r in segmenters}
    for n, item in enumerate(items, 1):
        rgb = item[0]
        answers = {r: s.segment(rgb) for r, s in segmenters.items()}
        reference = per_pixel(answers["torch-fp32"]) if "torch-fp32" in answers else None
        for r, found in answers.items():
            if reference is not None:
                moved[r].append(float((per_pixel(found) != reference).mean()))
            if spec.task == "panoptic":
                pred = np.zeros(rgb.shape[:2], np.int64)
                for i, mask in enumerate(found.masks):
                    pred[mask] = i + 1
                scores[r].add(item[1], item[2], pred,
                              {i + 1: int(c) for i, c in enumerate(found.class_ids)})
            else:
                label = item[1]
                pred = np.zeros(rgb.shape[:2], np.int64)
                for i, mask in enumerate(found.masks):
                    pred[mask] = found.class_ids[i]
                valid = label >= 0
                confusion[r] += np.bincount(label[valid] * spec.classes + pred[valid],
                                            minlength=spec.classes ** 2).reshape(spec.classes, -1)
        if n % 100 == 0:
            print(f"  {n}/{total} images, {time.perf_counter() - started:.0f} s")
    print(f"\n  accuracy on {total} images ({'COCO val2017 panoptic' if spec.task == 'panoptic' else 'ADE20K val'}):")
    for r in segmenters:
        if spec.task == "panoptic":
            res = scores[r].result()
            line = f"PQ {res['PQ']:.2f}  PQ_th {res['PQ_th']:.2f}  PQ_st {res['PQ_st']:.2f}"
        else:
            c = confusion[r]
            union = c.sum(0) + c.sum(1) - np.diag(c)
            iou = np.diag(c)[union > 0] / union[union > 0]
            line = f"mIoU {100 * iou.mean():.2f}"
        mv = moved[r]
        drift = (f"; pixels moved vs torch-fp32: mean {np.mean(mv):.2e}, max {np.max(mv):.2e}"
                 if mv and r != "torch-fp32" else "")
        print(f"    {r:14s} {line}{drift}")


def speed(variant: str, built: dict, device: str, runs: int) -> None:
    spec = SPECS[variant]
    rgb = load_image(str(fixtures()[0]))
    print(f"\n  speed on {fixtures()[0].name} {rgb.shape[1]}x{rgb.shape[0]}, {device}, median of {runs}:")
    for runtime, (segmenter, network) in built.items():
        with torch.inference_mode():
            pixels = segmenter.raw(rgb)[0].float() / 255.0

            def net():
                network(pixels)

            def call():
                segmenter.segment(rgb)
            net_ms = _median(net, device, runs)
            call_ms = _median(call, device, runs)
        print(f"    {runtime:14s} network {net_ms:8.1f} ms   call {call_ms:8.1f} ms")
    if device == "cuda":
        print(f"    peak CUDA memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB "
              f"(all runtimes loaded together)")
    del spec


def _median(work, device: str, runs: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        work()
    synchronise(device)
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        work()
        synchronise(device)
        timings.append((time.perf_counter() - start) * 1000)
    return statistics.median(timings)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variants", nargs="+")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--runtimes", nargs="+", default=list(RUNTIMES))
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--coco-panoptic", type=Path, default=None)
    parser.add_argument("--ade", type=Path, default=None)
    parser.add_argument("--count", type=int, default=500, help="images per accuracy measurement")
    parser.add_argument("--no-speed", action="store_true")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print(f"EoMT bench: torch {torch.__version__}, "
          f"{torch.cuda.get_device_name(0) if args.device == 'cuda' else 'CPU'}")

    for variant in args.variants:
        print(f"\n=== eomt/{variant}")
        built = {}
        for runtime in args.runtimes:
            try:
                made = build(variant, runtime, args.device)
            except Exception as error:  # a runtime that cannot build here is a result, not a crash
                print(f"  {runtime}: could not build ({type(error).__name__}: {error})")
                continue
            if made is None:
                print(f"  {runtime}: not available here")
            else:
                built[runtime] = made
        if not args.no_speed:
            speed(variant, built, args.device, args.runs)
        task = SPECS[variant].task
        if (task == "panoptic" and args.coco_panoptic) or (task == "semantic" and args.ade):
            accuracy(variant, {r: s for r, (s, _) in built.items()}, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

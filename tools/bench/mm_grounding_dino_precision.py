#!/usr/bin/env python3
"""Measure every way MM-Grounding-DINO (LLMDet) can run: accuracy, agreement and latency.

Bootstrap tooling; never ships. What ``auto`` choosing ``torch-fp16`` rests on, why no TensorRT
runtime is offered, and the latency table in the family's docs.

    python tools/bench/mm_grounding_dino_precision.py --coco /data/coco
    python tools/bench/mm_grounding_dino_precision.py --coco /data/coco --sizes 800 --n 100
    python tools/bench/mm_grounding_dino_precision.py --coco /data/coco --onnx --n 20

``--coco`` holds ``val2017/`` and ``annotations/instances_val2017.json``. TensorRT needs 11 or later.

COCO train2017 is part of LLMDet's training data (GroundingCap-1M), so the AP here is not a
zero-shot number; it is the yardstick for what a runtime changes, which is all it is used for.

**What is compared.** Two preprocessings -- upstream's resize (short side 800, long side up to
1333, nothing padded) and a fixed square (:func:`~mozo.vendors.mm_grounding_dino_deploy.image.fit_square`,
padded, with a mask) at each ``--sizes`` -- and on each, torch fp32, torch fp16 (autocast),
TensorRT fp16 and fp32 (built here; mozo does not offer them for this family). ``--onnx`` instead measures the fp32 graph under ONNX Runtime
(CPU, and CUDA where the provider is installed) against torch fp32 on the same square.

``accuracy``   COCO box AP, the 80 class names as one caption (195 tokens), top 100 queries.
``agreement``  Against torch fp32 *on the same preprocessing*, so it measures the runtime, not the
               square. A detection changes when it scores above 0.42 on one side with no IoU>0.5
               same-label partner above 0.40 on the other. ``floor`` is fp32 with only its input
               rounded to fp16: how often the model's detections change on their own.
``latency``    Three numbers, all medians of synchronised runs on one image:
               ``forward``  the network alone, inputs already on the GPU;
               ``predict``  what a caller waits for -- ``predict(image, prompts)`` from an array,
                            preprocessing, tokenising and decoding included -- with 3 prompts
                            (8 tokens) and with the 80 COCO names (195 tokens);
               ``load``     building the adapter, first (an engine built) and again (cached).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from common import median_ms  # noqa: E402
from mozo.adapters.mm_grounding_dino import MMGroundingDinoPredictor  # noqa: E402
from mozo.image import load_image  # noqa: E402
from mozo.runtimes import TensorRTRunner  # noqa: E402
from mozo.vendors.mm_grounding_dino_deploy import SPECS  # noqa: E402
from mm_grounding_dino_export import example, export, profile  # noqa: E402
from mozo.vendors.mm_grounding_dino_deploy.image import fit_square, preprocess  # noqa: E402
from mozo.vendors.mm_grounding_dino_deploy.network import phrase_masks  # noqa: E402
from mozo.vendors.mm_grounding_dino_deploy.predictor import caption_for  # noqa: E402
from mozo.weights import resolve  # noqa: E402

THRESHOLD = 0.4
MARGIN = 0.02
FEW = ["person", "car", "dog"]


def _candidates(logits, boxes, phrase_map, height, width):
    """Per query: xyxy in source pixels, score, prompt index (-1 where the peak names none)."""
    scored = logits[0].float().sigmoid().cpu()
    best, peak = scored.max(dim=1)
    padded = torch.zeros((phrase_map.shape[0], scored.shape[1]), dtype=torch.bool)
    padded[:, :phrase_map.shape[1]] = phrase_map
    belongs = padded[:, peak].t()
    label = torch.where(belongs.any(1), belongs.int().argmax(1), torch.tensor(-1))
    b = boxes[0].float().cpu()
    xyxy = torch.stack([(b[:, 0] - b[:, 2] / 2) * width, (b[:, 1] - b[:, 3] / 2) * height,
                        (b[:, 0] + b[:, 2] / 2) * width, (b[:, 1] + b[:, 3] / 2) * height], 1)
    return xyxy, best, label


def _unmatched(src, dst) -> int:
    from torchvision.ops import box_iou

    (sb, ss, sl), (db, ds, dl) = src, dst
    mine = (ss > THRESHOLD + MARGIN) & (sl >= 0)
    theirs = (ds > THRESHOLD) & (dl >= 0)
    if not mine.any() or not theirs.any():
        return int(mine.sum())
    overlap = box_iou(sb[mine], db[theirs]) * (sl[mine][:, None] == dl[theirs][None])
    return int((overlap.max(1).values <= 0.5).sum())


def _timed_load(**options):
    torch.cuda.synchronize()
    start = time.perf_counter()
    adapter = MMGroundingDinoPredictor(device="cuda", **options)
    torch.cuda.synchronize()
    return adapter, time.perf_counter() - start


def measure(variant: str, coco: Path, n: int, seed: int, sizes: list[int], work: Path) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(coco / "annotations" / "instances_val2017.json"))
    cats = gt.loadCats(gt.getCatIds())
    names, cat_ids = [c["name"] for c in cats], [c["id"] for c in cats]
    img_ids = sorted(gt.getImgIds())
    random.Random(seed).shuffle(img_ids)
    img_ids = img_ids[:n]
    spec = SPECS[variant]

    def load(img_id):
        info = gt.loadImgs(img_id)[0]
        return load_image(str(coco / "val2017" / info["file_name"])), info

    # Adapters: what ships. Every runtime here is reached through the public constructor.
    loads, adapters = {}, {}
    adapters["resize/torch-fp32"], loads["torch"] = _timed_load(variant=variant, runtime="torch-fp32")
    adapters["resize/torch-fp16"], _ = _timed_load(variant=variant, runtime="torch-fp16")
    for size in sizes:
        adapters[f"{size}/torch-fp32"], _ = _timed_load(variant=variant, runtime="torch-fp32",
                                                        input_size=size)
        adapters[f"{size}/torch-fp16"], _ = _timed_load(variant=variant, runtime="torch-fp16",
                                                        input_size=size)

    model = adapters["resize/torch-fp32"]._predictor.model
    tokenizer = adapters["resize/torch-fp32"]._predictor.tokenizer
    separators = adapters["resize/torch-fp32"]._predictor._separators

    def text_for(prompts):
        ids, types, mask = tokenizer.encode(caption_for(prompts))
        input_ids = torch.tensor([ids])
        attention, position_ids, phrase_map = phrase_masks(input_ids, separators)
        tensors = [t.cuda() for t in (input_ids, torch.tensor([types]),
                                      torch.tensor([mask], dtype=torch.bool), attention,
                                      position_ids)]
        return tensors, phrase_map[0], len(ids)

    text, phrase_map, tokens = text_for(names)

    # TensorRT, for the record: not a runtime mozo offers for this family, so built here, outside
    # the cache -- fp16 from a graph exported under autocast (Grounding DINO's recipe), and fp32.
    engines = {}
    for size in sizes:
        for precision in ("fp16", "fp32"):
            folder = work / f"{variant}-{precision}"
            start = time.perf_counter()
            engines[size, precision] = TensorRTRunner(
                folder / f"square{size}-{precision}.onnx", "cuda", engines=folder,
                export=lambda path, size=size, precision=precision: export(
                    model, path, size, fp16=precision == "fp16"),
                shapes=profile(64, model.spec.max_text_len), probe=example(size, "cuda"))
            loads[f"{size}/tensorrt-{precision}"] = time.perf_counter() - start

    def autocast(*inputs):
        with torch.autocast("cuda", torch.float16):
            return model(*inputs)

    # name -> (preprocess(image) -> image inputs, forward(*image inputs, text) -> outputs)
    resize = lambda image: (preprocess(image, spec.short_side, spec.max_side)[None].cuda(),)  # noqa: E731

    def square(size):
        def prep(image):
            canvas, padding = fit_square(image, size)
            return canvas[None].cuda(), padding[None].cuda()
        return prep

    runtimes = {
        "resize/torch-fp32": (resize, lambda x, t: model(x, *t)),
        "resize/floor": (resize, lambda x, t: model(x.half().float(), *t)),
        "resize/torch-fp16": (resize, lambda x, t: autocast(x, *t)),
    }
    for size in sizes:
        prep = square(size)
        runtimes[f"{size}/torch-fp32"] = (prep, lambda x, m, t: model(x, *t, m))
        runtimes[f"{size}/torch-fp16"] = (prep, lambda x, m, t: autocast(x, *t, m))
        for precision in ("fp16", "fp32"):
            runtimes[f"{size}/tensorrt-{precision}"] = (
                prep, lambda x, m, t, e=engines[size, precision]: e(x, *t, m))

    def reference_of(name):
        return name.split("/")[0] + "/torch-fp32"

    stats = {name: {"coco": [], "changes": 0} for name in runtimes}
    detections = {name: 0 for name in runtimes}
    first_image, _ = load(img_ids[0])
    few_text, _, few_tokens = text_for(FEW)
    latency = {}
    with torch.inference_mode():
        for name, (prep, forward) in runtimes.items():
            inputs = prep(first_image)
            latency[name] = {
                "forward_ms": median_ms(lambda: forward(*inputs, text), "cuda", runs=30, warmup=5),
                "forward_few_ms": median_ms(lambda: forward(*inputs, few_text), "cuda", runs=30,
                                            warmup=5)}
        for name, adapter in adapters.items():
            latency[name]["predict_few_ms"] = median_ms(
                lambda: adapter.predict(first_image, FEW), "cuda", runs=20, warmup=3)
            latency[name]["predict_coco_ms"] = median_ms(
                lambda: adapter.predict(first_image, names), "cuda", runs=20, warmup=3)
        print("  latency done", flush=True)

        for k, img_id in enumerate(img_ids):
            image, info = load(img_id)
            prepared = {}
            got = {}
            for name, (prep, forward) in runtimes.items():
                key = name.split("/")[0]
                if key not in prepared:
                    prepared[key] = prep(image)
                got[name] = _candidates(*forward(*prepared[key], text), phrase_map,
                                        info["height"], info["width"])
            for name, (b, s, l) in got.items():
                ranked = torch.where(l >= 0, s, torch.tensor(-1.0)).argsort(descending=True)[:100]
                for i in ranked.tolist():
                    if l[i] >= 0:
                        x1, y1, x2, y2 = b[i].tolist()
                        stats[name]["coco"].append({
                            "image_id": img_id, "category_id": cat_ids[int(l[i])],
                            "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(s[i])})
                reference = got[reference_of(name)]
                detections[name] += int(((reference[1] > THRESHOLD) & (reference[2] >= 0)).sum())
                if name != reference_of(name):
                    stats[name]["changes"] += (_unmatched(reference, got[name])
                                               + _unmatched(got[name], reference))
            if (k + 1) % 50 == 0:
                print(f"  {variant}: {k + 1}/{len(img_ids)}", flush=True)

    report = {"variant": variant, "images": len(img_ids), "seed": seed, "coco_tokens": tokens,
              "few_tokens": few_tokens, "gpu": torch.cuda.get_device_name(),
              "torch": torch.__version__, "loads_s": loads, "runtimes": {}}
    import tensorrt
    report["tensorrt"] = tensorrt.__version__
    for name, s in stats.items():
        with contextlib.redirect_stdout(io.StringIO()):
            ev = COCOeval(gt, gt.loadRes(s["coco"]), "bbox")
            ev.params.imgIds = img_ids
            ev.evaluate(), ev.accumulate(), ev.summarize()
        report["runtimes"][name] = {
            **latency[name], "AP": float(ev.stats[0]), "AP50": float(ev.stats[1]),
            "AP_small": float(ev.stats[3]),
            **({"changes": s["changes"], "detections": detections[name]}
               if name != reference_of(name) else {})}
    return report


def measure_onnx(variant: str, coco: Path, n: int, seed: int, size: int, work: Path) -> dict:
    """The fp32 graph under ONNX Runtime against torch fp32 on the same square.

    Parity is the largest difference in boxes and in finite logits over *n* COCO images, and the
    detection changes between the two at the bench threshold; latency is the forward alone, the
    inputs already where each runtime wants them. Torch on the CPU is compared with ORT on the
    CPU and torch on CUDA with ORT's CUDA provider, where it is installed.
    """
    import onnxruntime as ort
    from pycocotools.coco import COCO

    from mozo.vendors.mm_grounding_dino_deploy.checkpoint import build

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(coco / "annotations" / "instances_val2017.json"))
    names = [c["name"] for c in gt.loadCats(gt.getCatIds())]
    img_ids = sorted(gt.getImgIds())
    random.Random(seed).shuffle(img_ids)
    img_ids = img_ids[:n]

    spec = SPECS[variant]
    weights = resolve("mm_grounding_dino", variant, "torch-fp32")
    graph = work / f"{variant}-square{size}-fp32.onnx"
    export(build(spec, weights, device="cuda"), graph, size, fp16=False)
    torch.cuda.empty_cache()

    predictor = MMGroundingDinoPredictor(variant, device="cpu", runtime="torch-fp32")._predictor
    ids, types, mask = predictor.tokenizer.encode(caption_for(names))
    input_ids = torch.tensor([ids])
    attention, position_ids, phrase_map = phrase_masks(input_ids, predictor._separators)
    text = (input_ids, torch.tensor([types]), torch.tensor([mask], dtype=torch.bool), attention,
            position_ids)

    devices = {"cpu": ("CPUExecutionProvider", predictor.model)}
    if "CUDAExecutionProvider" in ort.get_available_providers():
        devices["cuda"] = ("CUDAExecutionProvider", build(spec, weights, device="cuda"))
    report = {"variant": variant, "size": size, "images": len(img_ids), "onnxruntime":
              ort.__version__, "torch": torch.__version__, "devices": {}}
    for device, (provider, model) in devices.items():
        session = ort.InferenceSession(str(graph), providers=[provider])
        feed_names = [i.name for i in session.get_inputs()]
        worst_boxes = worst_logits = 0.0
        changes = detections = 0
        timing = None
        with torch.inference_mode():
            for k, img_id in enumerate(img_ids):
                info = gt.loadImgs(img_id)[0]
                image = load_image(str(coco / "val2017" / info["file_name"]))
                canvas, padding = fit_square(image, size)
                inputs = (canvas[None], *text, padding[None])
                torch_out = model(*(t.to(device) for t in inputs))
                feed = dict(zip(feed_names, (t.numpy() for t in inputs)))
                ort_out = [torch.from_numpy(a) for a in session.run(None, feed)]
                want = [t.float().cpu() for t in torch_out]
                finite = want[0].isfinite()
                worst_logits = max(worst_logits,
                                   (ort_out[0][finite] - want[0][finite]).abs().max().item())
                worst_boxes = max(worst_boxes, (ort_out[1] - want[1]).abs().max().item())
                a = _candidates(*want, phrase_map[0], info["height"], info["width"])
                b = _candidates(*ort_out, phrase_map[0], info["height"], info["width"])
                detections += int(((a[1] > THRESHOLD) & (a[2] >= 0)).sum())
                changes += _unmatched(a, b) + _unmatched(b, a)
                if timing is None:
                    on_device = tuple(t.to(device) for t in inputs)
                    timing = {
                        "torch_ms": median_ms(lambda: model(*on_device), device, runs=10, warmup=2),
                        "onnxruntime_ms": median_ms(lambda: session.run(None, feed), "cpu",
                                                    runs=10, warmup=2)}
        report["devices"][device] = {**timing, "max_box_diff": worst_boxes,
                                     "max_logit_diff": worst_logits, "changes": changes,
                                     "detections": detections}
        print(f"  {variant} {device}: {json.dumps(report['devices'][device])}", flush=True)
    return report


def _print(report: dict) -> None:
    r0 = report["runtimes"]["resize/torch-fp32"]
    print(f"\n{report['variant']}: {report['images']} images, {report['gpu']}, torch "
          f"{report['torch']}, TensorRT {report['tensorrt']}; captions of {report['few_tokens']} "
          f"and {report['coco_tokens']} tokens")
    print(f"  {'runtime':22s} {'forward':>8s} {'x':>5s} {'predict3':>9s} {'predict80':>10s} {'x':>5s} "
          f"{'AP':>6s} {'APs':>6s} {'changes':>12s}")
    for name, r in report["runtimes"].items():
        changes = f"{r['changes']}/{r['detections']}" if "changes" in r else ""
        p3 = f"{r['predict_few_ms']:9.1f}" if "predict_few_ms" in r else f"{'':9s}"
        p80 = f"{r['predict_coco_ms']:10.1f}" if "predict_coco_ms" in r else f"{'':10s}"
        x2 = (f"{r0['predict_coco_ms'] / r['predict_coco_ms']:5.2f}" if "predict_coco_ms" in r
              else f"{'':5s}")
        print(f"  {name:22s} {r['forward_ms']:8.1f} {r0['forward_ms'] / r['forward_ms']:5.2f} "
              f"{p3} {p80} {x2} {100 * r['AP']:6.2f} {100 * r['AP_small']:6.2f} {changes:>12s}")
    print(f"  loads (s): {json.dumps(report['loads_s'])}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--coco", type=Path, required=True)
    ap.add_argument("--variants", nargs="+", default=list(SPECS), choices=list(SPECS))
    ap.add_argument("--sizes", nargs="+", type=int, default=[800])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--onnx", action="store_true",
                    help="measure the fp32 graph under ONNX Runtime instead (first --sizes only)")
    ap.add_argument("--out", type=Path, help="write every report here as JSON")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("these runtimes are CUDA-only")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    work = Path(tempfile.mkdtemp(prefix="mmgdino-bench-"))
    reports = []
    try:
        for variant in args.variants:
            if args.onnx:
                reports.append(measure_onnx(variant, args.coco, args.n, args.seed,
                                            args.sizes[0], work))
            else:
                reports.append(measure(variant, args.coco, args.n, args.seed, args.sizes, work))
                _print(reports[-1])
            torch.cuda.empty_cache()
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if args.out:
        args.out.write_text(json.dumps(reports, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

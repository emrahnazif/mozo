#!/usr/bin/env python3
"""Measure BiRefNet's runtimes: speed, agreement with torch fp32, and accuracy on a benchmark.

Bootstrap tooling; never ships. ``tools/verify/birefnet.py`` answers whether mozo returns
upstream's numbers; this answers what each runtime costs and what it changes.

    python tools/bench/birefnet.py lite --data DIS_VD.parquet --limit 100
    python tools/bench/birefnet.py matting --data P3M_500_NP.parquet --runtimes torch-cuda trt-fp32
    python tools/bench/birefnet.py general --speed-only --runtimes torch-cpu onnx-cpu

**Data.** A parquet of image/mask pairs as Hugging Face serves them: ``nobg/DIS5K``'s
``DIS_VD`` split (470 images, the authors' validation set for the segmentation checkpoints) or
``nobg/P3M-10K``'s ``P3M_500_NP`` (500 portraits, the matting checkpoint's test set). Evaluation
data only; nothing from either is published.

**Metrics.** Against the ground-truth mask, at the image's own size, from the uint8 alpha mozo
returns: MAE (mean absolute error, in 0..1) and the S-measure (Fan et al., ICCV 2017, alpha 0.5,
as the authors' evaluation computes it). Against torch fp32 (on CUDA with TF32 off where there is a GPU): the mean and the largest per-pixel
difference in the alpha, in grey levels. The two together say whether a runtime is *different*
and whether the difference *matters*.

**Runtimes**: ``torch-cpu``, ``torch-cuda`` (fp32, TF32 off, as TensorRT's fp32 is built),
``torch-cuda-fp16`` (autocast), ``onnx-cpu``, ``onnx-cuda``, ``trt-fp32``, ``trt-fp16``. Graphs
come from ``tools/export/birefnet.py``. Every timing synchronises the device; latency is the
median network-only forward on a preprocessed tensor, and ``predict`` adds the vendor's pre- and
post-processing on the fixture photograph.
"""

from __future__ import annotations

import argparse
import io
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from common import FIXTURES, synchronise  # noqa: E402
from mozo.vendors.birefnet_deploy import Predictor, postprocess, preprocess  # noqa: E402

REVISION = "2026-11-25"


# --------------------------------------------------------------------------------------------
# S-measure (Fan et al. 2017), as PySODMetrics computes it

def _ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    h, w = pred.shape
    n = h * w
    x, y = pred.mean(), gt.mean()
    sx = ((pred - x) ** 2).sum() / (n - 1 + 1e-8)
    sy = ((gt - y) ** 2).sum() / (n - 1 + 1e-8)
    sxy = ((pred - x) * (gt - y)).sum() / (n - 1 + 1e-8)
    alpha = 4 * x * y * sxy
    beta = (x ** 2 + y ** 2) * (sx + sy)
    if alpha != 0:
        return alpha / (beta + 1e-8)
    return 1.0 if alpha == 0 and beta == 0 else 0.0


def _region(pred: np.ndarray, gt: np.ndarray) -> float:
    h, w = gt.shape
    area = h * w
    if gt.sum() == 0:
        cx, cy = round(w / 2), round(h / 2)
    else:
        ys, xs = np.nonzero(gt)
        cx, cy = int(round(xs.mean())) + 1, int(round(ys.mean())) + 1
    score = 0.0
    for (y0, y1, x0, x1) in ((0, cy, 0, cx), (0, cy, cx, w), (cy, h, 0, cx), (cy, h, cx, w)):
        part_gt, part_pred = gt[y0:y1, x0:x1], pred[y0:y1, x0:x1]
        weight = part_gt.size / area
        if part_gt.size:
            score += weight * _ssim(part_pred, part_gt)
    return score


def _object(pred: np.ndarray, gt: np.ndarray) -> float:
    def one(p: np.ndarray, mask: np.ndarray) -> float:
        x = p[mask].mean() if mask.any() else 0.0
        sigma = p[mask].std(ddof=1) if mask.sum() > 1 else 0.0
        return 2 * x / (x ** 2 + 1 + sigma + 1e-8)

    fg = gt.astype(bool)
    u = gt.mean()
    return u * one(pred, fg) + (1 - u) * one(1 - pred, ~fg)


def s_measure(pred: np.ndarray, gt: np.ndarray) -> float:
    """*pred* in 0..1 float, *gt* binary {0, 1}."""
    y = gt.mean()
    if y == 0:
        return 1 - pred.mean()
    if y == 1:
        return pred.mean()
    return max(0.0, 0.5 * _object(pred, gt) + 0.5 * _region(pred, gt))


# --------------------------------------------------------------------------------------------
# runtimes: each is a callable pixels (1, 3, S, S) float32 CPU tensor -> logits CPU tensor

def build(name: str, variant: str, folder: Path, engines: Path):
    if name.startswith("torch"):
        device = "cpu" if name == "torch-cpu" else "cuda"
        predictor = Predictor.from_pretrained(variant, folder / "torch-fp32.pth", device=device)
        fp16 = name.endswith("fp16")

        @torch.inference_mode()
        def run(pixels: torch.Tensor) -> torch.Tensor:
            with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
                return predictor.model(pixels.to(device)).float()
        return run, device
    if name.startswith("onnx"):
        from mozo.runtimes import OnnxRunner

        device = "cuda" if name == "onnx-cuda" else "cpu"
        if device == "cuda":
            import onnxruntime

            # The CUDA 12 build of onnxruntime-gpu beside a CUDA 13 torch: the cu12 libraries come
            # from nvidia-*-cu12 wheels and have to be loaded before the session is made.
            getattr(onnxruntime, "preload_dlls", lambda **_: None)(cuda=True, cudnn=True, msvc=False)
        runner = OnnxRunner(folder / "onnx-fp32.onnx", device=device)
        print(f"  {name}: provider {runner.provider}")
        return (lambda pixels: torch.from_numpy(runner(pixels.numpy())[0]).float()), "cpu"
    if name.startswith("trt"):
        from mozo.runtimes import TensorRTRunner

        graph = folder / ("onnx-fp16.onnx" if name == "trt-fp16" else "onnx-fp32.onnx")
        start = time.perf_counter()
        # Per variant and graph file: an engine is named for its graph's file name, which every
        # variant shares, so a shared directory would hand one variant another's engine.
        runner = TensorRTRunner(graph, "cuda", engines=engines / variant / str(graph.stat().st_mtime_ns))
        print(f"  {name}: engine {'built' if runner.built else 'loaded'} in "
              f"{time.perf_counter() - start:.0f} s")
        run = lambda pixels: runner(pixels.cuda())[0].float()  # noqa: E731
        # What the execution context allocates outside torch, which torch's counters never see.
        run.scratch = runner._engine.device_memory_size_v2 / 2 ** 20
        return run, "cuda"
    raise SystemExit(f"unknown runtime {name}")


def median_ms(work, device: str, runs: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        work()
    synchronise(device)
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        work()
        synchronise(device)
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def samples(path: Path, limit: int | None):
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    label = "label" if "label" in table.column_names else "mask"
    for i, row in enumerate(table.to_pylist()):
        if limit is not None and i >= limit:
            return
        rgb = np.asarray(Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB"))
        gt = np.asarray(Image.open(io.BytesIO(row[label]["bytes"])).convert("L"))
        yield rgb, gt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variant")
    parser.add_argument("--runtimes", nargs="+",
                        default=["torch-cuda-fp16", "onnx-cuda", "trt-fp32",
                                 "trt-fp16"])
    parser.add_argument("--data", type=Path, help="parquet of image/mask pairs")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--speed-only", action="store_true")
    parser.add_argument("--engines", type=Path, default=Path.home() / ".cache" / "mozo-bench-engines",
                        help="where built TensorRT engines are kept between runs")
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "weights")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    folder = args.weights_dir / "birefnet" / args.variant / REVISION
    # The accuracy reference: torch fp32 on CUDA (TF32 off) where there is one -- 470 Swin-L
    # forwards on a CPU take hours -- else on the CPU.
    ref_device = "cuda" if torch.cuda.is_available() else "cpu"
    reference = Predictor.from_pretrained(args.variant, folder / "torch-fp32.pth", device=ref_device)
    size = reference.resolution
    print(f"{args.variant}: {size}x{size}, torch {torch.__version__}")

    runtimes = {}
    for name in args.runtimes:
        runtimes[name] = build(name, args.variant, folder, args.engines)

    photo = np.asarray(Image.open(sorted(FIXTURES.glob("*.jpg"))[0]).convert("RGB"))
    pixels = preprocess(photo, size)
    # GPU memory: for torch, the peak torch allocates above what is resident before the call (the
    # reference model's weights stay resident throughout); for TensorRT, the engine's scratch.
    print(f"\n{'runtime':16s} {'forward ms':>10s} {'predict ms':>10s} {'GPU MiB':>9s}")
    for name, (run, device) in runtimes.items():
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            resident = torch.cuda.memory_allocated()
        runs = args.runs if device == "cuda" else max(3, args.runs // 4)
        forward = median_ms(lambda: run(pixels), device, runs)
        whole = median_ms(lambda: postprocess(run(preprocess(photo, size)), photo.shape[:2]),
                          device, runs)
        peak = (getattr(run, "scratch", None)
                or ((torch.cuda.max_memory_allocated() - resident) / 2 ** 20 if device == "cuda"
                    else float("nan")))
        print(f"{name:16s} {forward:10.1f} {whole:10.1f} {peak:9.0f}")

    if args.speed_only or args.data is None:
        return 0

    scores = {name: {"mae": [], "s": [], "dmean": [], "dmax": []} for name in runtimes}
    ref_scores = {"mae": [], "s": []}
    count = 0
    for rgb, gt in samples(args.data, args.limit):
        count += 1
        x = preprocess(rgb, size)
        with torch.inference_mode():
            want = postprocess(reference.model(x.to(ref_device)), rgb.shape[:2])
        target = (gt > 127).astype(np.float64)
        truth = gt / 255.0
        for name, (run, _) in runtimes.items():
            got = postprocess(run(x), rgb.shape[:2])
            pred = got / 255.0
            scores[name]["mae"].append(np.abs(pred - truth).mean())
            scores[name]["s"].append(s_measure(pred, target))
            diff = np.abs(got.astype(np.int16) - want.astype(np.int16))
            scores[name]["dmean"].append(diff.mean())
            scores[name]["dmax"].append(diff.max())
        pred = want / 255.0
        ref_scores["mae"].append(np.abs(pred - truth).mean())
        ref_scores["s"].append(s_measure(pred, target))

    print(f"\naccuracy on {count} images of {args.data.name}")
    print(f"{'runtime':16s} {'MAE':>8s} {'S-measure':>9s} {'vs fp32: mean':>14s} {'max':>5s}")
    print(f"{'torch fp32 ' + ref_device:16s} {np.mean(ref_scores['mae']):8.4f} {np.mean(ref_scores['s']):9.4f}"
          f" {'(reference)':>14s}")
    for name, s in scores.items():
        print(f"{name:16s} {np.mean(s['mae']):8.4f} {np.mean(s['s']):9.4f} "
              f"{np.mean(s['dmean']):14.4f} {int(np.max(s['dmax'])):5d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Measure Depth Anything 3's runtimes: speed, agreement with torch fp32, and accuracy on NYUv2.

Bootstrap tooling; never ships.

    python tools/bench/depth_anything_v3.py --nyu /path/to/nyu/val/official \\
        --variants small --runtimes torch-fp32@cuda tensorrt-fp16@cuda --images 654

``--nyu`` is a directory of the official NYU Depth V2 test split as ``.h5`` files holding
``rgb`` (3x480x640 uint8) and ``depth`` (480x640 metres), the layout of the
``sayakpaul/nyu_depth_v2`` dataset's ``val/official`` files (654 images).

Every row runs mozo end to end -- ``DepthAnythingV3Predictor(variant, device, runtime=...)``
and ``predict`` -- so a number here is what a user gets. Per row:

* **ms**: median wall time of one ``predict`` (decode excluded, pre-processing, network,
  sky fill, resize and inversion included), after three warm-up calls.
* **vs fp32**: the largest and the mean relative difference of the inverse depth against the
  ``torch-fp32`` row of the same device (``torch-fp32@cpu`` for CPU rows), over all pixels.
* **AbsRel, delta1**: accuracy against the NYU ground truth, Eigen crop, depth 1e-3..10 m. The
  prediction is relative inverse depth, so it is aligned first by least squares in scale and
  shift to the ground truth's inverse depth (the MiDaS / Depth Anything protocol), then turned
  into depth. These numbers compare runtimes; they are not the authors' published benchmark.

``--upstream`` additionally times the authors' network (same checkpoint, fp32, CUDA) on the
same inputs, with its ray branch, for the latency of the extraction against the reference.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from common import synchronise  # noqa: E402
from mozo.adapters.depth_anything_v3 import DepthAnythingV3Predictor  # noqa: E402

#: The Eigen crop of NYU's 480x640 frames, and the depth range evaluated.
CROP = (slice(45, 471), slice(41, 601))
MIN_DEPTH, MAX_DEPTH = 1e-3, 10.0


def load_nyu(directory: Path, count: int) -> list[tuple[np.ndarray, np.ndarray]]:
    import h5py

    out = []
    for path in sorted(directory.glob("*.h5"))[:count]:
        with h5py.File(path) as f:
            out.append((np.ascontiguousarray(np.array(f["rgb"]).transpose(1, 2, 0)),
                        np.array(f["depth"], dtype=np.float32)))
    return out


def accuracy(inverse: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """AbsRel and delta1 for one image, after scale-and-shift alignment in inverse depth."""
    pred, gt = inverse[CROP].astype(np.float64), gt[CROP].astype(np.float64)
    valid = (gt > MIN_DEPTH) & (gt < MAX_DEPTH)
    target = 1.0 / gt[valid]
    x = pred[valid]
    a = np.stack([x, np.ones_like(x)], 1)
    scale, shift = np.linalg.lstsq(a, target, rcond=None)[0]
    aligned = np.clip(scale * x + shift, 1.0 / MAX_DEPTH, None)
    depth, truth = 1.0 / aligned, gt[valid]
    abs_rel = float(np.mean(np.abs(depth - truth) / truth))
    delta1 = float(np.mean(np.maximum(depth / truth, truth / depth) < 1.25))
    return abs_rel, delta1


def run(variant: str, runtime: str, device: str, data, reference: list | None):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device == "cuda":
        torch.cuda.empty_cache()
        free_before = torch.cuda.mem_get_info()[0]
    model = DepthAnythingV3Predictor(variant, device=device, runtime=runtime)
    for rgb, _ in data[:3]:
        model.predict(rgb)
    timings, preds, rel_max, rel_mean, scores = [], [], [], [], []
    for i, (rgb, gt) in enumerate(data):
        synchronise(device)
        start = time.perf_counter()
        inverse = model.predict(rgb)
        synchronise(device)
        timings.append((time.perf_counter() - start) * 1000)
        preds.append(inverse)
        scores.append(accuracy(inverse, gt))
        if reference is not None:
            rel = np.abs(inverse - reference[i]) / np.abs(reference[i])
            rel_max.append(float(rel.max()))
            rel_mean.append(float(rel.mean()))
    memory = None
    if device == "cuda":
        memory = (free_before - torch.cuda.mem_get_info()[0]) / 2**20
    del model
    row = {
        "variant": variant, "runtime": runtime, "device": device, "images": len(data),
        "ms": statistics.median(timings),
        "abs_rel": float(np.mean([s[0] for s in scores])),
        "delta1": float(np.mean([s[1] for s in scores])),
        "vs_fp32_max": max(rel_max) if rel_max else None,
        "vs_fp32_mean": float(np.mean(rel_mean)) if rel_mean else None,
        "device_mib": memory,
    }
    return row, preds


def time_upstream(variant: str, checkout: Path, checkpoints: Path, data) -> float:
    """Median ms of upstream's network forward, fp32 on CUDA, on the same preprocessed inputs."""
    import types

    sys.path.insert(0, str(checkout / "src"))
    for name, attr in (("depth_anything_3.utils.export", "export"),
                       ("depth_anything_3.utils.pose_align", "align_poses_umeyama")):
        stub = types.ModuleType(name)
        setattr(stub, attr, None)
        sys.modules[name] = stub
    from depth_anything_3.api import DepthAnything3

    repo = {"small": "DA3-SMALL", "base": "DA3-BASE", "mono-large": "DA3MONO-LARGE"}[variant]
    up = DepthAnything3.from_pretrained(str(checkpoints / repo)).eval().to("cuda")
    timings = []
    with torch.inference_mode():
        for i, (rgb, _) in enumerate(data):
            imgs, _, _ = up._preprocess_inputs([rgb], None, None, 504, "upper_bound_resize")
            x, _, _ = up._prepare_model_inputs(imgs, None, None)
            torch.cuda.synchronize()
            start = time.perf_counter()
            up._convert_to_prediction(up.model(x, None, None, [], False, False))
            torch.cuda.synchronize()
            if i >= 3:
                timings.append((time.perf_counter() - start) * 1000)
    return statistics.median(timings)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nyu", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=["small", "base", "mono-large"])
    parser.add_argument("--runtimes", nargs="+", default=["torch-fp32@cuda"],
                        help="runtime@device pairs; torch-fp32 of a device is its reference")
    parser.add_argument("--images", type=int, default=654)
    parser.add_argument("--cpu-images", type=int, default=50,
                        help="how many of the images CPU rows run on")
    parser.add_argument("--upstream", type=Path, help="Depth-Anything-3 checkout")
    parser.add_argument("--checkpoints", type=Path, help="HF snapshot directories, for --upstream")
    parser.add_argument("--json", type=Path, help="append rows here")
    args = parser.parse_args()

    data = load_nyu(args.nyu, args.images)
    print(f"{len(data)} NYUv2 test images; torch {torch.__version__}"
          + (f", {torch.cuda.get_device_name()}" if torch.cuda.is_available() else ""))
    for variant in args.variants:
        references: dict[str, list] = {}
        pairs = [tuple(p.split("@")) for p in args.runtimes]
        # Each device's torch-fp32 first, so every other row has its reference.
        pairs.sort(key=lambda p: p[0] != "torch-fp32")
        for runtime, device in pairs:
            subset = data[:args.cpu_images] if device == "cpu" else data
            reference = references.get(device)
            row, preds = run(variant, runtime, device, subset,
                             None if reference is None else reference[:len(subset)])
            if runtime == "torch-fp32":
                references[device] = preds
            fmt = (lambda v, f: "-" if v is None else format(v, f))
            print(f"  {variant:<10} {runtime:<14} {device:<5} {row['images']:>4} img  "
                  f"{row['ms']:8.1f} ms  AbsRel {row['abs_rel']:.4f}  d1 {row['delta1']:.4f}  "
                  f"vs fp32 max {fmt(row['vs_fp32_max'], '.2e')} mean "
                  f"{fmt(row['vs_fp32_mean'], '.2e')}  mem {fmt(row['device_mib'], '.0f')} MiB",
                  flush=True)
            if args.json:
                with args.json.open("a") as f:
                    f.write(json.dumps(row) + "\n")
        if args.upstream:
            ms = time_upstream(variant, args.upstream, args.checkpoints, data[:30])
            print(f"  {variant:<10} upstream net fp32 cuda (network only, with ray branch) "
                  f"{ms:.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

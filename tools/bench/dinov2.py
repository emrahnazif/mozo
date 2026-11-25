#!/usr/bin/env python3
"""Measure DINOv2's runtimes: speed, parity with torch fp32, and k-NN accuracy.

    python tools/bench/dinov2.py base --device cuda \\
        --runtimes torch-fp32 torch-fp16 onnx-fp32 tensorrt-fp32 tensorrt-fp16 \\
        --coco /path/to/val2017 --count 500 --imagenette /path/to/imagenette2-320

For every runtime, against torch fp32 (TF32 off, the path the exact gate pins on the CPU):

- **latency**: median of ``encode_image`` on one decoded photograph, end to end (preprocess,
  backbone, normalise), and of the backbone alone on one preprocessed image, the device
  synchronised every call. On a 1920x1281 photograph the CPU resize is most of the end-to-end
  time, so the backbone column is the one that tells runtimes apart;
- **throughput cost of one image per forward**: torch only, the backbone on 32 images one at a
  time against one batch of 32 (the adapter never batches: a vector must not depend on its
  batch-mates);
- **parity** on ``--count`` COCO images: worst ``max|delta|``, worst cosine, and how many of each
  image's 10 nearest neighbours (among the same images) survive;
- **accuracy** (``--imagenette``): upstream's k-NN protocol -- cosine similarity, k = 20, votes
  weighted by ``exp(sim / 0.07)`` -- with train images as the index (embedded once, in torch fp32)
  and every val image as a query embedded by the runtime under test. Top-1 accuracy.

``torch-fp16`` is the torch network under CUDA fp16 autocast, which is what the fp16 graph is
exported from; it is measured here whether or not mozo offers it.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault("MOZO_BASE_URL", f"file://{ROOT / 'weights'}")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from common import fixtures, median_ms, synchronise  # noqa: E402
from mozo.adapters.dinov2 import Dinov2Predictor  # noqa: E402
from mozo.image import load_image  # noqa: E402
from mozo.vendors.dinov2_deploy import SPECS, Encoder  # noqa: E402
from mozo.vendors.dinov2_deploy.image import preprocess  # noqa: E402
from mozo.weights import resolve  # noqa: E402


def build(variant: str, runtime: str, device: str):
    """Something with ``encode_image`` for *runtime*; ``torch-fp16`` is built here."""
    if runtime != "torch-fp16":
        return Dinov2Predictor(variant, device=device, runtime=runtime)
    encoder = Encoder(resolve("dinov2", variant, "torch-fp32"), SPECS[variant], device=device)

    def half(pixels):
        with torch.autocast("cuda", torch.float16):
            return encoder.network()(pixels)

    fp16 = Encoder(None, SPECS[variant], device=device, backbone=half)
    fp16.encode_image_np = lambda images: fp16.encode_image(
        [load_image(i) for i in (images if isinstance(images, list) else [images])]).numpy()
    return fp16


def encode(model, images) -> np.ndarray:
    if hasattr(model, "encode_image_np"):
        return model.encode_image_np(images)
    return model.encode_image(images)


def embed(model, paths: list[str], chunk: int = 64) -> np.ndarray:
    return np.concatenate([encode(model, paths[i:i + chunk]) for i in range(0, len(paths), chunk)])


def knn_top1(index: np.ndarray, labels: np.ndarray, queries: np.ndarray, truth: np.ndarray,
             k: int = 20, temperature: float = 0.07) -> float:
    """Upstream's k-NN evaluation (``dinov2/eval/knn.py``): weighted vote of the k most similar."""
    classes = int(labels.max()) + 1
    correct = 0
    index_t, labels_t = torch.from_numpy(index), torch.from_numpy(labels)
    for start in range(0, len(queries), 256):
        sims = torch.from_numpy(queries[start:start + 256]) @ index_t.T
        top, where = sims.topk(k, dim=1)
        votes = torch.zeros(len(top), classes).scatter_add_(
            1, labels_t[where], (top / temperature).exp())
        correct += int((votes.argmax(1).numpy() == truth[start:start + 256]).sum())
    return correct / len(queries)


def imagenette(root: Path, per_class: int, seed: int = 0):
    """``(train paths, train labels, val paths, val labels)``; *per_class* train images a class."""
    synsets = sorted(p.name for p in (root / "train").iterdir())
    rng = random.Random(seed)
    train, train_y, val, val_y = [], [], [], []
    for label, synset in enumerate(synsets):
        names = sorted(str(p) for p in (root / "train" / synset).glob("*.JPEG"))
        for path in rng.sample(names, min(per_class, len(names))):
            train.append(path), train_y.append(label)
        for path in sorted((root / "val" / synset).glob("*.JPEG")):
            val.append(str(path)), val_y.append(label)
    return train, np.array(train_y), val, np.array(val_y)


def neighbour_overlap(a: np.ndarray, b: np.ndarray, k: int = 10) -> float:
    """Mean fraction of each row's k nearest neighbours (by cosine, excluding itself) kept."""
    def nearest(x):
        sims = x @ x.T
        np.fill_diagonal(sims, -np.inf)
        return np.argsort(-sims, axis=1)[:, :k]
    na, nb = nearest(a), nearest(b)
    return float(np.mean([len(set(r) & set(s)) / k for r, s in zip(na, nb)]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variants", nargs="+")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--runtimes", nargs="+", default=["torch-fp32"])
    parser.add_argument("--coco", type=Path, default=None)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--imagenette", type=Path, default=None)
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--machine", default="", help="printed with every timing")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    photo = load_image(str(fixtures()[0]))
    coco = []
    if args.coco:
        names = sorted(str(p) for p in args.coco.glob("*.jpg"))
        coco = random.Random(0).sample(names, min(args.count, len(names)))
    data = imagenette(args.imagenette, args.per_class) if args.imagenette else None

    for variant in args.variants:
        print(f"\n=== dinov2/{variant} on {args.device} {args.machine}")
        reference = build(variant, "torch-fp32", args.device)
        want = embed(reference, coco) if coco else None
        index = embed(reference, data[0]) if data else None

        # One image per forward against one batch, the backbone alone.
        network = reference._encoder.network()
        pixels = torch.stack([preprocess(photo)] * 32).to(args.device)
        with torch.inference_mode():
            alone = median_ms(lambda: [network(pixels[i:i + 1]) for i in range(32)],
                              args.device, runs=5, warmup=2)
            batch = median_ms(lambda: network(pixels), args.device, runs=5, warmup=2)
        print(f"  backbone, 32 images: one at a time {alone:.1f} ms, one batch {batch:.1f} ms "
              f"({alone / batch:.2f}x)")

        for runtime in args.runtimes:
            try:
                started = time.perf_counter()
                model = build(variant, runtime, args.device)
                encode(model, photo)
                synchronise(args.device)
                load = time.perf_counter() - started
            except Exception as error:  # noqa: BLE001 -- a runtime that cannot load is a finding
                print(f"  {runtime:14s} could not load: {type(error).__name__}: {error}")
                continue
            ms = median_ms(lambda: encode(model, photo), args.device, runs=args.runs)
            core = (model._encoder if hasattr(model, "_encoder") else model).network()
            one = pixels[:1]
            with torch.inference_mode():
                core_ms = median_ms(lambda: core(one), args.device, runs=args.runs)
            row = (f"  {runtime:14s} {ms:7.2f} ms/image, backbone {core_ms:6.2f} ms"
                   f"  (first load+encode {load:.1f} s)")
            if coco:
                got = embed(model, coco)
                cos = (got * want).sum(1)
                row += (f"  max|d| {np.abs(got - want).max():.2e}  min cos {cos.min():.6f}"
                        f"  10-NN kept {neighbour_overlap(got, want):.4f}")
            if data:
                row += f"  k-NN top-1 {knn_top1(index, data[1], embed(model, data[2]), data[3]):.4f}"
            print(row, flush=True)
            del model
            if args.device == "cuda":
                torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Export EoMT networks to ONNX and drop them into the local ``weights/`` tree.

Runs once, on a machine you control, and never ships.

    python tools/export/eomt.py coco-panoptic-small coco-panoptic-large
    python tools/export/eomt.py coco-panoptic-small --fp16          # on CUDA

**One fixed shape per variant, 1x3xSxS**, the square the network runs at (640, 1280 or 512).
Panoptic pre-processing always pads to it, and a semantic picture's windows are run through the
graph one at a time. The position embedding is the checkpoint's own grid, so there is nothing to
interpolate and nothing for a dynamic shape to mean.

The graph is the network alone: pixels in ``[0, 1]`` (already divided by 255) to the last layer's
``mask_logits`` and ``class_logits``. Pre-processing and post-processing stay in Python, the same
code on every runtime.

``onnx-fp32``  exported from the torch model on the CPU, checked through ONNX Runtime against torch
               on the fixtures: logits within :data:`FP32_TOLERANCE`, pixels moved within
               :data:`FP32_MOVED`.
``onnx-fp16``  ``--fp16``: exported on CUDA under fp16 autocast, so the graph carries autocast's
               types for a strongly typed TensorRT engine to run exactly those. Checked on the CPU
               against torch fp32 by how many pixels change class: at most :data:`FP16_MOVED`,
               a check that it is not broken. Whether fp16 is good enough is
               ``tools/bench/eomt.py``'s question, answered with PQ and mIoU.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
# This file is called eomt.py; its own directory must not shadow anything it imports.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
# Exports from the tree you just built, not from the published bucket.
os.environ.setdefault("MOZO_BASE_URL", f"file://{ROOT / 'weights'}")

from common import fixtures, require_one_file, variant_parser  # noqa: E402
from mozo.image import load_image  # noqa: E402
from mozo.runtimes import OnnxRunner  # noqa: E402
from mozo.vendors.eomt_deploy import SPECS, Segmenter  # noqa: E402
from mozo.vendors.eomt_deploy.checkpoint import load_network  # noqa: E402
from mozo.weights import resolve  # noqa: E402

INPUT_NAME = "pixels"
OUTPUT_NAMES = ["mask_logits", "class_logits"]
OPSET = 17
FP16_OPSET = 18

#: Largest ``max|delta|`` allowed between the fp32 graph (ONNX Runtime, CPU) and torch, on either
#: output, and the largest share of pixels it may move to another class. Mask logits reach -260 on
#: the fixture, so the first is a relative 2e-04. ONNX Runtime's arithmetic is not torch's: measured
#: 1.5e-03 / no pixel on ``coco-panoptic-small``, 1.3e-02 / one pixel in 2.5 million on ``-base``.
#: A sanity check, not the verdict: ``tools/bench/eomt.py`` measures PQ and mIoU.
FP32_TOLERANCE = 5e-2
FP32_MOVED = 1e-4

#: The share of pixels the fp16 graph may move to another class, averaged over the pictures (on a
#: flat one a single segment hovering at the 0.8 threshold can flip half the frame to void). A
#: check that the graph is not broken -- autocast itself moves 6e-05 on ``coco-panoptic-small``,
#: the graph 1e-03 -- not the verdict, which is PQ and mIoU in ``tools/bench/eomt.py``.
FP16_MOVED = 1e-2


def images() -> list[np.ndarray]:
    """The repository's photographs and a flat picture. A short check of what is written;
    ``tools/verify/eomt.py`` and ``tools/bench/eomt.py`` measure more."""
    return [load_image(str(p)) for p in fixtures()] + [np.full((480, 640, 3), 90, np.uint8)]


def _export(network, side: int, destination: Path, fp16: bool) -> None:
    """fp32 on the CPU at opset 17; fp16 under CUDA autocast at opset 18."""
    dummy = torch.rand(1, 3, side, side, device="cuda" if fp16 else "cpu")
    with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=fp16):
        torch.onnx.export(network.eval(), (dummy,), str(destination),
                          input_names=[INPUT_NAME], output_names=OUTPUT_NAMES,
                          opset_version=FP16_OPSET if fp16 else OPSET, dynamo=False)


def moved(a, b) -> float:
    """Share of pixels whose class differs between two answers (void counts as a class)."""
    def per_pixel(found):
        out = np.full(found.masks.shape[1:], -1, np.int64)
        for i, mask in enumerate(found.masks):
            out[mask] = found.class_ids[i]
        return out
    return float((per_pixel(a) != per_pixel(b)).mean())


def _one_at_a_time(run):
    def network(pixels):
        outs = [run(pixels[i:i + 1]) for i in range(len(pixels))]
        return torch.cat([o[0] for o in outs]), torch.cat([o[1] for o in outs])
    return network


def export(variant: str, revision: str | None, weights_dir: Path, fp16: bool) -> None:
    spec = SPECS[variant]
    key = "onnx-fp16" if fp16 else "onnx-fp32"
    print(f"\n=== eomt/{variant} {key}")
    checkpoint = resolve("eomt", variant, "torch-fp32", revision=revision)
    destination = weights_dir / "eomt" / variant / checkpoint.parent.name / f"{key}.onnx"
    reference = Segmenter(checkpoint, spec, device="cpu")
    pictures = images()
    want = [reference.segment(rgb) for rgb in pictures]

    if fp16:
        if not torch.cuda.is_available():
            raise SystemExit("the fp16 graph is exported on CUDA, under fp16 autocast")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        network = load_network(spec, checkpoint, "cuda")

        def autocast(x):
            with torch.autocast("cuda", torch.float16):
                mask_logits, class_logits = network(x.cuda())
            return mask_logits.float().cpu(), class_logits.float().cpu()
        half = Segmenter(None, spec, device="cpu", network=autocast)
        budget = float(np.mean([moved(half.segment(rgb), w) for rgb, w in zip(pictures, want)]))
    else:
        network = reference.network()

    with tempfile.TemporaryDirectory() as scratch:
        staged = Path(scratch) / f"{key}.onnx"
        _export(network, spec.image_size, staged, fp16)
        require_one_file(staged)
        runner = OnnxRunner(staged, device="cpu")
        run = _one_at_a_time(lambda x: tuple(torch.from_numpy(o) for o in runner(x.numpy())))
        graph = Segmenter(None, spec, device="cpu", network=run)
        got = [graph.segment(rgb) for rgb in pictures]
        drift = float(np.mean([moved(g, w) for g, w in zip(got, want)]))
        if fp16:
            print(f"  pixels moved against torch fp32: graph {drift:.2e}, torch autocast "
                  f"{budget:.2e} (limit {FP16_MOVED:.0e})")
            if drift > FP16_MOVED:
                raise SystemExit("the fp16 graph moves too many pixels; not writing it")
        else:
            with torch.inference_mode():
                deltas = []
                for rgb in pictures:
                    pixels, mask_t, class_t, _ = reference.raw(rgb)
                    mask_g, class_g = run(pixels / 255.0)
                    deltas += [float((mask_g - mask_t).abs().max()),
                               float((class_g - class_t).abs().max())]
            print(f"  max|delta| on the logits: {max(deltas):.2e} (limit {FP32_TOLERANCE:.0e}); "
                  f"pixels moved {drift:.2e}")
            if max(deltas) > FP32_TOLERANCE or drift > FP32_MOVED:
                raise SystemExit("the graph disagrees with torch; not writing it")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(staged, destination)
    print(f"  {key} {destination.stat().st_size / 1e6:.1f} MB -> {destination}")


def main() -> int:
    parser = variant_parser(__doc__, ROOT / "weights", required=True, revision=None)
    parser.add_argument("--fp16", action="store_true", help="export the onnx-fp16 graph (CUDA)")
    args = parser.parse_args()
    refused = []
    for variant in args.variants:
        try:
            export(variant, args.revision, args.weights_dir, args.fp16)
        except SystemExit as stop:  # one refusal does not stop the others
            print(f"  REFUSED: {stop}")
            refused.append(variant)
    return 1 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())

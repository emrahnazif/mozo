#!/usr/bin/env python3
"""Export PP-OCRv6's two networks to ONNX: the graphs TensorRT engines are built from.

Bootstrap tooling; it runs on a machine you control and never ships.

    python tools/export/ppocrv6.py medium small tiny            # onnx-fp32-det, onnx-fp32-rec
    python tools/export/ppocrv6.py medium --fp16                # onnx-fp16-det, onnx-fp16-rec

**Two graphs per runtime**, published as parts (``onnx-fp32-det`` and ``onnx-fp32-rec``): the
detector runs once per page, the recogniser once per batch of lines, and everything between them
is OpenCV either way. Both have dynamic shapes, because this model has no fixed geometry: the
detector takes the page at its own size rounded to 32, the recogniser a batch of 1 to 6 lines,
48 pixels high and as wide as the widest of them.

**fp16** is exported on CUDA under fp16 autocast, so a strongly typed TensorRT engine runs
autocast's types (convolutions and matrix products in fp16, softmax and norms in fp32).

Each export is checked against the torch module it came from, on random inputs at several
shapes, through ONNX Runtime on the CPU, before it is written.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from common import require_one_file, variant_parser  # noqa: E402
from mozo.runtimes import OnnxRunner  # noqa: E402
from mozo.vendors.ppocrv6_deploy.checkpoint import load  # noqa: E402
from mozo.vendors.ppocrv6_deploy.config import SPECS  # noqa: E402
from mozo.weights import resolve  # noqa: E402

OPSET = 18                     # 17's TorchScript path cannot carry autocast's casts round Resize
INPUT_NAME = "pixels"
OUTPUT_NAME = {"det": "probability", "rec": "probabilities"}
DYNAMIC = {"det": {0: "batch", 2: "height", 3: "width"}, "rec": {0: "batch", 3: "width"}}
DYNAMIC_OUT = {"det": {0: "batch", 2: "height", 3: "width"}, "rec": {0: "batch", 1: "steps"}}
EXAMPLE = {"det": (1, 3, 736, 1280), "rec": (6, 3, 48, 640)}
CHECK_SHAPES = {"det": [(1, 3, 224, 640), (1, 3, 96, 1920)], "rec": [(1, 3, 48, 320), (6, 3, 48, 1208)]}

#: How far the fp32 graph may be from torch on the CPU, in probability, on random inputs: kernel
#: order and nothing else. Measured: 1.4e-07 to 5.4e-05, except the tiny detector at 5.9e-04,
#: the network that also moves most between Paddle and torch at the same weights.
FP32_TOLERANCE = 1e-3

#: The fp16 graph's bar is only that it is the same network: probabilities within this on noise,
#: measured 3.7e-06 to 3.4e-02 (the tiny detector the farthest). Whether it is good enough to
#: publish is the bench's ICDAR 2015 run, not this.
FP16_TOLERANCE = 5e-2


def export(module: torch.nn.Module, kind: str, destination: Path, fp16: bool) -> None:
    device = "cuda" if fp16 else "cpu"
    module = module.to(device).eval()
    example = torch.randn(*EXAMPLE[kind], device=device)
    with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=fp16):
        torch.onnx.export(module, (example,), str(destination), input_names=[INPUT_NAME],
                          output_names=[OUTPUT_NAME[kind]], opset_version=OPSET, dynamo=False,
                          dynamic_axes={INPUT_NAME: DYNAMIC[kind],
                                        OUTPUT_NAME[kind]: DYNAMIC_OUT[kind]})
    require_one_file(destination)


def check(module: torch.nn.Module, kind: str, graph: Path, tolerance: float) -> float:
    runner = OnnxRunner(graph, device="cpu")
    module = module.float().cpu().eval()
    worst = 0.0
    generator = torch.Generator().manual_seed(0)
    for shape in CHECK_SHAPES[kind]:
        x = torch.randn(*shape, generator=generator)
        with torch.no_grad():
            want = module(x).numpy()
        (got,) = runner(x.numpy())
        worst = max(worst, float(np.abs(want - got.astype(np.float32)).max()))
    if worst > tolerance:
        raise SystemExit(f"{graph.name}: {worst:.2e} from torch, over {tolerance:.0e}; not written")
    return worst


def main() -> int:
    parser = variant_parser(__doc__, ROOT / "weights", required=True)
    parser.add_argument("--fp16", action="store_true", help="the autocast graphs, on CUDA")
    args = parser.parse_args()
    precision = "fp16" if args.fp16 else "fp32"
    for variant in args.variants:
        checkpoint = resolve("ppocrv6", variant, "torch-fp32")
        revision_dir = args.weights_dir / "ppocrv6" / variant / checkpoint.parent.name
        detector, recogniser = load(checkpoint, SPECS[variant])
        for kind, module in (("det", detector), ("rec", recogniser)):
            with tempfile.TemporaryDirectory() as scratch:
                staged = Path(scratch) / f"onnx-{precision}-{kind}.onnx"
                export(module, kind, staged, args.fp16)
                worst = check(module, kind, staged,
                              FP16_TOLERANCE if args.fp16 else FP32_TOLERANCE)
                revision_dir.mkdir(parents=True, exist_ok=True)
                target = revision_dir / staged.name
                target.write_bytes(staged.read_bytes())
            print(f"  {variant} {kind} {precision}: {target.stat().st_size / 1e6:.1f} MB, "
                  f"max|delta| vs torch {worst:.1e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

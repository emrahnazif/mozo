#!/usr/bin/env python3
"""Time RTMO's runtimes: the graph alone and a whole ``predict``, per variant.

Bootstrap tooling; never ships. Produces the speed table in ``rtmo_deploy/PROVENANCE.md``.
Exactness against MMPose is ``tools/verify/rtmo.py``'s job, not this one's.

    python tools/bench/rtmo.py                         # every variant, every runtime that runs here
    python tools/bench/rtmo.py small --device cuda

Runtimes are mozo's own (``RTMOPredictor``): ``torch-fp32``, ``onnx-fp32``, and on CUDA
``tensorrt-fp32``. The graph time feeds the letterboxed batch straight to the runtime (already on
the GPU for torch and TensorRT); the ``predict`` time is the whole call on the fixture photograph,
letterbox and decode included. torch on CUDA runs with TF32 off, as the engine does.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from common import fixtures, median_ms  # noqa: E402
from mozo.adapters.rtmo import RTMOPredictor  # noqa: E402
from mozo.image import load_image  # noqa: E402
from mozo.runtimes import RuntimeError_  # noqa: E402

RUNTIMES = {"cpu": ("torch-fp32", "onnx-fp32"), "cuda": ("torch-fp32", "tensorrt-fp32")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variants", nargs="*", default=list(RTMOPredictor.VARIANTS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    image = load_image(str(fixtures()[0]))
    for variant in args.variants:
        base = None
        for runtime in RUNTIMES[args.device]:
            try:
                model = RTMOPredictor(variant, device=args.device, runtime=runtime)
            except RuntimeError_ as error:
                print(f"{variant:6s} {runtime:13s} unavailable: {error}")
                continue
            vendor = model._predictor
            batch = vendor.preprocess(image)[0]
            if runtime != "onnx-fp32":
                batch = batch.to(args.device)
            graph = median_ms(lambda: vendor.dense(batch), args.device, args.runs)
            whole = median_ms(lambda: model.predict(image), args.device, args.runs)
            base = base or graph
            print(f"{variant:6s} {runtime:13s} graph {graph:7.2f} ms ({base / graph:.2f}x)   "
                  f"predict {whole:7.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

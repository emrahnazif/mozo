#!/usr/bin/env python3
"""Measure Qwen3-VL's vision tower as an ONNX graph and a TensorRT engine. Nothing is published.

An autoregressive VLM is not one graph. The vision tower runs once per image; the text decoder runs
once for the prompt and then once per token, with a key/value cache that grows every step, three
rotary position rows computed outside it, and DeepStack features added to three of its layers. Of
those, only the vision tower is a plain function of fixed-shape inputs, and it is the part a graph
runtime could speed up most simply -- so it is what this measures, at the example photograph's grid
(80 x 120 patches, 9,600 tokens). The positions, the interpolation taps and the rotary tables are
computed outside the graph (they depend on the grid through Python integers), exactly as the torch
path computes them, and passed in.

    python tools/export/qwen3_vl.py 2b --out vision.onnx          # export, then ONNX Runtime CPU
    python tools/export/qwen3_vl.py 2b --out vision.onnx --tensorrt fp32 fp16   # on a GPU

Reported: parity against the torch tower on the same inputs (max |d| on the merged tokens and the
DeepStack features), and median latency against torch on the same device.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from common import FIXTURES, median_ms  # noqa: E402

from mozo.image import load_image  # noqa: E402
from mozo.runtimes import TensorRTRunner, providers_for  # noqa: E402
from mozo.vendors.qwen3_vl_deploy import SPECS  # noqa: E402
from mozo.vendors.qwen3_vl_deploy.checkpoint import load_model  # noqa: E402
from mozo.vendors.qwen3_vl_deploy.image import preprocess  # noqa: E402
from mozo.vendors.qwen3_vl_deploy.vision import interpolation, rotary_positions  # noqa: E402
from mozo.weights import resolve  # noqa: E402

INPUTS = ["pixels", "indices", "weights", "cos", "sin"]
OUTPUTS = ["merged", "deepstack_0", "deepstack_1", "deepstack_2"]


class Tower(nn.Module):
    """The vision tower with its grid-dependent tables as inputs, one image, no split."""

    def __init__(self, visual, half: bool = False) -> None:
        super().__init__()
        self.visual = visual.half() if half else visual
        self.half = half

    def forward(self, pixels, indices, weights, cos, sin):
        v = self.visual
        if self.half:  # an fp16 graph with fp32 inputs and outputs, for a strongly typed engine
            pixels, weights = pixels.half(), weights.half()
        x = v.patch_embed(pixels)
        x = x + (v.pos_embed(indices) * weights[:, :, None]).sum(1).to(x.dtype)
        cu = torch.tensor([0, x.shape[0]], dtype=torch.int32)
        out = []
        for number, block in enumerate(v.blocks):
            x = block(x, cu, (cos, sin))
            if number in v.spec.deepstack:
                out.append(v.deepstack_merger_list[v.spec.deepstack.index(number)](x))
        outputs = (v.merger(x), *out)
        return tuple(o.float() for o in outputs) if self.half else outputs


def inputs(model, size: tuple[int, int] | None = None) -> list[torch.Tensor]:
    """The example photograph's tower inputs (resized to *size*, if given), on the CPU, computed
    as the torch path does."""
    rgb = load_image(str(FIXTURES / "example.jpg"))
    if size:
        import numpy as np
        from PIL import Image

        rgb = np.asarray(Image.fromarray(rgb).resize(size, Image.BICUBIC))
    pixels, grid = preprocess(rgb, model.spec.vision)
    s = model.spec.vision
    indices, weights = interpolation(grid, s.grid, s.merge)
    cos, sin = model.visual._rotary(pixels, rotary_positions(grid, s.merge))
    return [pixels, indices, weights, cos, sin]


def worst(got, want) -> float:
    return max(float((torch.as_tensor(g).cpu() - w).abs().max()) for g, w in zip(got, want))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variant", choices=list(SPECS))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tensorrt", nargs="*", default=[], choices=["fp32", "fp16"])
    parser.add_argument("--size", nargs=2, type=int, metavar=("W", "H"),
                        help="resize the photograph first (the graph is fixed to its grid)")
    args = parser.parse_args()

    model = load_model(SPECS[args.variant], resolve("qwen3_vl", args.variant, "torch-bf16"), "cpu")
    tower = Tower(model.visual).eval()
    feed = inputs(model, tuple(args.size) if args.size else None)
    with torch.inference_mode():
        want = [t.clone() for t in tower(*feed)]
    graphs = {"fp32": args.out, "fp16": args.out.with_name(args.out.stem + "-fp16.onnx")}
    for precision, path in graphs.items():
        if path.exists() or (precision == "fp16" and "fp16" not in args.tensorrt):
            continue
        import copy

        source = tower if precision == "fp32" else Tower(copy.deepcopy(model.visual), half=True).eval()
        started = time.perf_counter()
        torch.onnx.export(source, tuple(feed), str(path), opset_version=18, dynamo=False,
                          input_names=INPUTS, output_names=OUTPUTS)
        print(f"exported {path} in {time.perf_counter() - started:.0f}s")

    import onnxruntime as ort

    session = ort.InferenceSession(str(args.out), providers=providers_for(args.device))
    arrays = {name: t.numpy() for name, t in zip(INPUTS, feed)}
    print(f"onnxruntime {session.get_providers()[0]}: max |d| vs torch fp32 "
          f"{worst(session.run(None, arrays), want):.3e}")
    ms_ort = median_ms(lambda: session.run(None, arrays), "cpu", runs=5, warmup=1)
    device_tower = tower.to(args.device)
    device_feed = [t.to(args.device) for t in feed]
    with torch.inference_mode():
        ms_torch = median_ms(lambda: device_tower(*device_feed), args.device, runs=5, warmup=1)
    print(f"vision tower, {feed[0].shape[0]:,} patches: torch fp32 {ms_torch:.0f} ms, onnxruntime {ms_ort:.0f} ms "
          f"({ms_torch / ms_ort:.2f}x) on {args.device}")
    for precision in args.tensorrt:
        # mozo's own runner: TensorRT 11, strongly typed, TF32 off, engine kept beside the graph.
        started = time.perf_counter()
        runner = TensorRTRunner(graphs[precision], args.device, probe=tuple(device_feed))
        built = time.perf_counter() - started
        ms = median_ms(lambda: runner(*device_feed), args.device, runs=5, warmup=1)
        print(f"tensorrt {precision}: {'built' if runner.built else 'loaded'} in {built:.0f}s, "
              f"{ms:.0f} ms ({ms_torch / ms:.2f}x torch fp32), "
              f"max |d| vs torch fp32 {worst(runner(*device_feed), want):.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

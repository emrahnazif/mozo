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
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from mozo.vendors.qwen3_vl_deploy import SPECS  # noqa: E402
from mozo.vendors.qwen3_vl_deploy.checkpoint import load_model  # noqa: E402
from mozo.vendors.qwen3_vl_deploy.image import preprocess  # noqa: E402
from mozo.vendors.qwen3_vl_deploy.vision import interpolation, rotary_positions  # noqa: E402
from mozo.weights import resolve  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "images" / "example.jpg"


class Tower(nn.Module):
    """The vision tower with its grid-dependent tables as inputs, one image, no split."""

    def __init__(self, visual) -> None:
        super().__init__()
        self.visual = visual

    def forward(self, pixels, indices, weights, cos, sin):
        v = self.visual
        x = v.patch_embed(pixels)
        x = x + (v.pos_embed(indices) * weights[:, :, None]).sum(1).to(x.dtype)
        cu = torch.tensor([0, x.shape[0]], dtype=torch.int32)
        out = []
        for number, block in enumerate(v.blocks):
            x = block(x, cu, (cos, sin))
            if number in v.spec.deepstack:
                out.append(v.deepstack_merger_list[v.spec.deepstack.index(number)](x))
        return (v.merger(x), *out)


def inputs(model, device):
    from PIL import Image

    rgb = np.array(Image.open(FIXTURE).convert("RGB"))
    pixels, grid = preprocess(rgb, model.spec.vision)
    s = model.spec.vision
    indices, weights = interpolation(grid, s.grid, s.merge)
    cos, sin = model.visual._rotary(pixels, rotary_positions(grid, s.merge))
    return [t.to(device) for t in (pixels, indices, weights, cos, sin)]


def median(work, device, runs=5):
    work()
    times = []
    for _ in range(runs):
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        work()
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    return statistics.median(times) * 1000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variant", choices=list(SPECS))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tensorrt", nargs="*", default=[], choices=["fp32", "fp16"])
    args = parser.parse_args()

    model = load_model(SPECS[args.variant], resolve("qwen3_vl", args.variant, "torch-bf16"), "cpu")
    tower = Tower(model.visual).eval()
    feed = inputs(model, "cpu")
    with torch.inference_mode():
        want = [t.clone() for t in tower(*feed)]
    if not args.out.exists():
        started = time.perf_counter()
        torch.onnx.export(tower, tuple(feed), str(args.out), opset_version=18, dynamo=False,
                          input_names=["pixels", "indices", "weights", "cos", "sin"],
                          output_names=["merged", "deepstack_0", "deepstack_1", "deepstack_2"])
        print(f"exported {args.out} in {time.perf_counter() - started:.0f}s, "
              f"{sum(p.stat().st_size for p in args.out.parent.glob(args.out.name + '*')) / 1e6:.0f} MB")
    names = ["pixels", "indices", "weights", "cos", "sin"]
    arrays = {n: t.numpy() for n, t in zip(names, feed)}

    import onnxruntime as ort

    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if args.device == "cuda"
                 else ["CPUExecutionProvider"])
    session = ort.InferenceSession(str(args.out), providers=providers)
    got = session.run(None, {i.name: arrays[i.name] for i in session.get_inputs()})
    worst = max(float(np.abs(g - w.numpy()).max()) for g, w in zip(got, want))
    print(f"onnxruntime {session.get_providers()[0]}: max |d| vs torch fp32 {worst:.3e}")
    ms_ort = median(lambda: session.run(None, {i.name: arrays[i.name] for i in session.get_inputs()}),
                    "cpu")
    device_tower = tower.to(args.device)
    device_feed = [t.to(args.device) for t in feed]
    with torch.inference_mode():
        ms_torch = median(lambda: device_tower(*device_feed), args.device)
    print(f"vision tower, 9,600 patches: torch fp32 {ms_torch:.0f} ms, onnxruntime {ms_ort:.0f} ms "
          f"({ms_torch / ms_ort:.2f}x) on {args.device}")
    for precision in args.tensorrt:
        _tensorrt(args.out, precision, arrays, want, ms_torch)
    return 0


def _tensorrt(onnx_path: Path, precision: str, arrays, want, ms_torch) -> None:
    """Build a strongly typed engine (TensorRT 11) and compare it with torch fp32."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    source = onnx_path
    if precision == "fp16":
        source = onnx_path.with_name(onnx_path.stem + "-fp16.onnx")
        if not source.exists():
            import onnx
            from onnxconverter_common import float16

            graph = onnx.load(str(onnx_path))
            onnx.save(float16.convert_float_to_float16(graph, keep_io_types=True), str(source))
    if not parser.parse_from_file(str(source)):
        print(f"  tensorrt {precision}: parse failed: {parser.get_error(0)}")
        return
    config = builder.create_builder_config()
    started = time.perf_counter()
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        print(f"  tensorrt {precision}: build failed")
        return
    engine = trt.Runtime(logger).deserialize_cuda_engine(blob)
    context = engine.create_execution_context()
    buffers = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            buffers[name] = torch.from_numpy(arrays[name]).cuda()
        else:
            shape = tuple(context.get_tensor_shape(name))
            buffers[name] = torch.empty(shape, dtype=torch.float32, device="cuda")
        context.set_tensor_address(name, buffers[name].data_ptr())
    stream = torch.cuda.current_stream().cuda_stream

    def run():
        context.execute_async_v3(stream)

    ms = median(run, "cuda")
    outs = [buffers[n] for n in ("merged", "deepstack_0", "deepstack_1", "deepstack_2")]
    worst = max(float((o.cpu() - w).abs().max()) for o, w in zip(outs, want))
    print(f"  tensorrt {precision}: built in {time.perf_counter() - started:.0f}s, "
          f"{ms:.0f} ms ({ms_torch / ms:.2f}x torch fp32), max |d| {worst:.3e}, "
          f"engine {len(blob) / 1e6:.0f} MB")


if __name__ == "__main__":
    raise SystemExit(main())

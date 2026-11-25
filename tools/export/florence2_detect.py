#!/usr/bin/env python3
"""Measure Florence-2's image-and-prompt encoder as an ONNX graph and a TensorRT engine.

Nothing is published from this; it is the measurement behind that decision.

Florence-2 answers in two halves. The *encoder* half -- DaViT, the image projection, and the BART
encoder over 577 image tokens plus the ``<OD>`` prompt's 13 -- runs once per image on fixed shapes
(the image is always 768 x 768 and the prompt is always the same sentence), so it exports as one
static graph. The *decoder* half is a 3-beam search, one step per token, with a growing key/value
cache reordered between steps; that is a loop, not a graph, and it is not exported.

    python tools/export/florence2_detect.py base --out encoder.onnx                 # ONNX Runtime
    python tools/export/florence2_detect.py base --out encoder.onnx --device cuda --tensorrt fp32 fp16

Reported: parity of the encoder output against torch fp32 on the example photograph, the encoder's
latency on each runtime, and how much of a whole ``<OD>`` call the encoder is -- the most any graph
of it could save.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from common import FIXTURES, median_ms  # noqa: E402

from fetch.florence2_detect import published  # noqa: E402

from mozo.image import load_image  # noqa: E402
from mozo.runtimes import TensorRTRunner, providers_for  # noqa: E402
from mozo.vendors.florence2_deploy import SPECS, Florence2Predictor  # noqa: E402
from mozo.vendors.florence2_deploy.image import preprocess  # noqa: E402
from mozo.vendors.florence2_deploy.tokenizer import PROMPTS  # noqa: E402



class Encoder(nn.Module):
    """Pixels to the encoder's output for the ``<OD>`` prompt (a constant inside the graph)."""

    def __init__(self, network, half: bool = False) -> None:
        super().__init__()
        self.network = network.half() if half else network
        self.half = half
        self.register_buffer("prompt", torch.tensor([PROMPTS["<OD>"]]), persistent=False)

    def forward(self, pixels):
        if self.half:  # an fp16 graph with fp32 input and output, for a strongly typed engine
            pixels = pixels.half()
        out = self.network.encode(self.network.encode_image(pixels), self.prompt)
        return out.float() if self.half else out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variant", choices=list(SPECS))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tensorrt", nargs="*", default=[], choices=["fp32", "fp16"])
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    weights = published(args.variant)
    model = Florence2Predictor(weights, args.variant, "cpu")
    encoder = Encoder(model.model).eval()
    rgb = load_image(str(FIXTURES / "example.jpg"))
    pixels = preprocess(rgb)
    with torch.inference_mode():
        want = encoder(pixels).clone()
    graphs = {"fp32": args.out, "fp16": args.out.with_name(args.out.stem + "-fp16.onnx")}
    for precision, path in graphs.items():
        if path.exists() or (precision == "fp16" and "fp16" not in args.tensorrt):
            continue
        source = encoder if precision == "fp32" else Encoder(copy.deepcopy(model.model), True).eval()
        started = time.perf_counter()
        torch.onnx.export(source, (pixels,), str(path), opset_version=18, dynamo=False,
                          input_names=["pixels"], output_names=["encoded"])
        print(f"exported {path} in {time.perf_counter() - started:.0f}s", flush=True)

    import onnxruntime as ort

    session = ort.InferenceSession(str(args.out), providers=providers_for(args.device))
    feed = {"pixels": pixels.numpy()}
    got = torch.from_numpy(session.run(None, feed)[0])
    print(f"onnxruntime {session.get_providers()[0]}: max |d| vs torch fp32 "
          f"{(got - want).abs().max().item():.3e}", flush=True)
    ms_ort = median_ms(lambda: session.run(None, feed), args.device, runs=10, warmup=2)

    device_model = Florence2Predictor(weights, args.variant, args.device)
    device_encoder = Encoder(device_model.model).eval().to(args.device)
    device_pixels = pixels.to(args.device)
    with torch.inference_mode():
        ms_torch = median_ms(lambda: device_encoder(device_pixels), args.device, runs=10, warmup=2)
        ms_call = median_ms(lambda: device_model.detect(rgb), args.device, runs=5, warmup=1)
    print(f"{args.variant} on {args.device}: whole <OD> call {ms_call:.0f} ms; encoder torch fp32 "
          f"{ms_torch:.0f} ms ({ms_torch / ms_call:.0%} of the call), onnxruntime {ms_ort:.0f} ms "
          f"({ms_torch / ms_ort:.2f}x)", flush=True)
    for precision in args.tensorrt:
        started = time.perf_counter()
        runner = TensorRTRunner(graphs[precision], args.device, probe=(device_pixels,))
        built = time.perf_counter() - started
        ms = median_ms(lambda: runner(device_pixels), args.device, runs=10, warmup=2)
        out = torch.as_tensor(runner(device_pixels)[0]).float().cpu()
        print(f"tensorrt {precision}: {'built' if runner.built else 'loaded'} in {built:.0f}s, "
              f"{ms:.1f} ms ({ms_torch / ms:.2f}x torch fp32 encoder), max |d| vs torch fp32 "
              f"{(out - want).abs().max().item():.3e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

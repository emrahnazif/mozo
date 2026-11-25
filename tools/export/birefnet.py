#!/usr/bin/env python3
"""Export BiRefNet's network to ONNX, for ONNX Runtime and for TensorRT engines built from it.

    python tools/export/birefnet.py general lite matting hr          # onnx-fp32
    python tools/export/birefnet.py general --fp16                   # onnx-fp16, on CUDA

Writes ``weights/birefnet/<variant>/<revision>/onnx-fp32.onnx`` (or ``onnx-fp16.onnx``) beside the
checkpoint it reads, after checking it.

**What the graph holds.** The network only: ``pixels`` ``(1, 3, S, S)``, already resized and
normalised by the vendor's ``preprocess``, in; ``logits`` ``(1, 1, S, S)`` out. Fixed shape --
every checkpoint runs at one resolution -- so nothing about the input size is traced.

**The deformable convolution is decomposed** (:mod:`mozo.vendors.birefnet_deploy.graph`): no
exporter maps ``torchvision::deform_conv2d``, ONNX Runtime has no ``DeformConv`` kernel and
TensorRT reaches one only through a plugin. The decomposition is not bit-identical to torchvision;
on the fixture through the whole network it moves the matte by at most 6e-5 (lite, torch CPU).

**fp16** is exported on CUDA under torch's fp16 autocast, so the graph carries autocast's types;
a strongly typed TensorRT engine then runs it in exactly those. The weights autocast casts to
float16 at every use are stored in float16 (:func:`fold_fp16_weights`), which halves the file.

**The check** runs the graph through ONNX Runtime on the CPU on every shared fixture and compares
the matte (the sigmoid) with the vendor's torch fp32 forward. fp32 must agree to
:data:`FP32_LIMIT`; fp16 to :data:`FP16_LIMIT` on the mean. Accuracy on a benchmark is the real
criterion and is measured by ``tools/bench/birefnet.py``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from common import fixtures, require_one_file  # noqa: E402
from mozo.image import load_image  # noqa: E402
from mozo.vendors.birefnet_deploy import MODEL_SPECS, Predictor, preprocess  # noqa: E402
from mozo.vendors.birefnet_deploy.graph import decomposed  # noqa: E402

REVISION = "2026-11-25"
OPSET = 18

#: Largest per-pixel difference in the matte (0..1) between the fp32 graph and torch fp32. The
#: decomposition alone measured 6e-5 on lite; ONNX Runtime adds its own kernel order.
FP32_LIMIT = 1e-3

#: Mean absolute difference in the matte for the fp16 graph. Catches a broken graph; whether fp16
#: is accurate enough is decided on DIS-VD / P3M-500-NP, not here.
FP16_LIMIT = 1e-2


class _Graph(torch.nn.Module):
    def __init__(self, net: torch.nn.Module, autocast: bool) -> None:
        super().__init__()
        self.net, self.autocast = net, autocast

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        with torch.autocast(pixels.device.type, dtype=torch.float16, enabled=self.autocast):
            return self.net(pixels)


def _export(predictor: Predictor, destination: Path, fp16: bool) -> None:
    device = torch.device("cuda" if fp16 else "cpu")
    net = predictor.model.to(device)
    size = predictor.resolution
    pixels = torch.zeros(1, 3, size, size, device=device)
    with decomposed(), torch.inference_mode():
        torch.onnx.export(_Graph(net, fp16).eval(), (pixels,), str(destination), dynamo=True,
                          opset_version=OPSET, input_names=["pixels"], output_names=["logits"],
                          external_data=False)
    predictor.model.to("cpu")


def fold_fp16_weights(path: Path) -> int:
    """Store in float16 every weight the fp16 graph only ever reads through a cast to float16.

    Autocast leaves the parameters float32 and casts them at each use, so an exported fp16 graph
    is as large as the fp32 one and carries a ``Cast`` per weight. Folding the cast into the
    initializer computes the same float16 values (round to nearest, as ``Cast`` does), halves the
    file, and changes nothing a runtime computes. Weights that something reads in float32 --
    LayerNorm, which autocast keeps in float32 -- are left as they are. Returns how many folded.
    """
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(path))
    graph = model.graph
    weights = {init.name: init for init in graph.initializer}
    readers: dict[str, list] = {}
    for node in graph.node:
        for name in node.input:
            readers.setdefault(name, []).append(node)
    to_half = onnx.TensorProto.FLOAT16
    folded = 0
    for name, init in list(weights.items()):
        nodes = readers.get(name, [])
        if init.data_type != onnx.TensorProto.FLOAT or not nodes or not all(
                n.op_type == "Cast" and any(a.name == "to" and a.i == to_half for a in n.attribute)
                for n in nodes):
            continue
        half = numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), name)
        graph.initializer.remove(init)
        graph.initializer.append(half)
        for cast in nodes:
            # Everything that read the cast's output reads the half-precision weight instead.
            for node in graph.node:
                for i, inp in enumerate(node.input):
                    if inp == cast.output[0]:
                        node.input[i] = name
            for out in graph.output:
                if out.name == cast.output[0]:
                    raise SystemExit("a graph output is a cast weight; refusing to fold")
            graph.node.remove(cast)
        folded += 1
    onnx.save(model, str(path))
    return folded


def store_exact_weights_in_fp16(path: Path) -> int:
    """Store in float16, behind a ``Cast`` to float32, every float32 weight float16 holds exactly.

    ``general`` and ``hr`` are published in float16; their fp32 graph would otherwise carry every
    weight at twice the size the publisher stored it, for values that are float16 values anyway.
    Only an initializer whose every element survives the round trip bit for bit is converted, so
    the graph computes exactly what it did. Returns how many were converted.
    """
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(str(path))
    graph = model.graph
    casts, converted = [], 0
    for init in list(graph.initializer):
        if init.data_type != onnx.TensorProto.FLOAT:
            continue
        values = numpy_helper.to_array(init)
        half = values.astype(np.float16)
        if values.size < 16 or not np.array_equal(half.astype(np.float32), values):
            continue
        graph.initializer.remove(init)
        graph.initializer.append(numpy_helper.from_array(half, f"{init.name}__fp16"))
        casts.append(helper.make_node("Cast", [f"{init.name}__fp16"], [init.name],
                                      to=onnx.TensorProto.FLOAT))
        converted += 1
    for node in reversed(casts):
        graph.node.insert(0, node)
    onnx.save(model, str(path))
    return converted


def _check(predictor: Predictor, path: Path, fp16: bool) -> None:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    # Under autocast the graph still takes float32 pixels and casts inside; ask it.
    dtype = np.float16 if session.get_inputs()[0].type == "tensor(float16)" else np.float32
    for image in fixtures():
        pixels = preprocess(load_image(str(image)), predictor.resolution)
        with torch.inference_mode():
            want = predictor.model(pixels).sigmoid().numpy()
        (got,) = session.run(None, {"pixels": pixels.numpy().astype(dtype)})
        got = 1 / (1 + np.exp(-got.astype(np.float64)))
        delta = np.abs(got - want)
        worst, mean = float(delta.max()), float(delta.mean())
        ok = mean <= FP16_LIMIT if fp16 else worst <= FP32_LIMIT
        print(f"  {image.name:24s} matte max {worst:.2e}  mean {mean:.2e}  "
              f"{'ok' if ok else 'REFUSED'}")
        if not ok:
            raise SystemExit(f"{path.name} does not hold parity on {image.name}; not writing it")


def export(variant: str, weights_dir: Path, revision: str, fp16: bool) -> None:
    if fp16 and not torch.cuda.is_available():
        raise SystemExit("the fp16 graph is exported on CUDA, under fp16 autocast")
    folder = weights_dir / "birefnet" / variant / revision
    predictor = Predictor.from_pretrained(variant, folder / "torch-fp32.pth")
    predictor.published_fp16 = any(t.dtype == torch.float16 for t in torch.load(
        folder / "torch-fp32.pth", map_location="cpu", weights_only=True).values())
    key = "onnx-fp16" if fp16 else "onnx-fp32"
    print(f"{variant}: {key}, {MODEL_SPECS[variant].resolution}x{MODEL_SPECS[variant].resolution}")
    with tempfile.TemporaryDirectory(dir=folder) as scratch:
        staged = Path(scratch) / f"{key}.onnx"
        _export(predictor, staged, fp16)
        require_one_file(staged)
        if fp16:
            print(f"  folded    {fold_fp16_weights(staged)} weights to float16")
        elif predictor.published_fp16:
            print(f"  stored    {store_exact_weights_in_fp16(staged)} weights in float16, as published")
        print(f"  exported  opset {OPSET}, {staged.stat().st_size / 1e6:.0f} MB")
        _check(predictor, staged, fp16)
        shutil.move(staged, folder / staged.name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variants", nargs="+", choices=list(MODEL_SPECS))
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "weights")
    args = parser.parse_args()
    for variant in args.variants:
        export(variant, args.weights_dir, args.revision, args.fp16)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

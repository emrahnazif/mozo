#!/usr/bin/env python3
"""Export Depth Anything 3 variants to ONNX and drop them into the local ``weights/`` tree.

Runs once, on a machine you control, and never ships.

    python tools/export/depth_anything_v3.py small base mono-large          # onnx-fp32, CPU
    python tools/export/depth_anything_v3.py small base mono-large --fp16   # onnx-fp16, CUDA

**The graph has dynamic height and width, and takes its shape-only tensors as inputs.** An image
reaches the network at 504 px on its long side and a multiple of 14 on the other, so the input
shape varies from image to image; a graph frozen at one shape would be wrong at every other (V2's
trap). Two things in the network are computed from the shape with Python floats -- the resampled
position grid of the backbone, and the UV embeddings of the any-view head -- and no exporter can
trace float arithmetic on a symbolic size. So the vendor computes them
(:meth:`DepthAnything3Net.shape_inputs`), the graph takes them beside the pixels, and the runner
computes them per shape exactly as the torch path does. Inputs, in order:

    pixels  1 x 3 x 14h x 14w     the normalised image
    grid    1 x C x h x w         the backbone's position grid, resampled for this shape
    uv0..3  1 x c_i x h x w       any-view models only: the head's UV embeddings per stage
    uv4     1 x f/2 x 14h x 14w   ... and at the output

h and w (patches) range over 3..36: an image more elongated than 12:1 has a side under 42 px at
the processing size and is refused by the graph runtimes (torch runs it).

Outputs: ``depth`` (1 x 14h x 14w), and ``sky`` for ``mono-large``. The sky fill and everything
after it run in the vendor on the outputs, as on the torch path.

``onnx-fp16`` (``--fp16``) is exported on CUDA under fp16 autocast, as upstream runs its own
inference, so the graph carries autocast's types -- and upstream's own choice to run the head in
fp32 (``torch.autocast(enabled=False)`` around it). A strongly typed TensorRT engine runs exactly
those types; it is checked through TensorRT (needs TensorRT 11 and a GPU).

Every export is checked against the torch model it came from, on the fixture photographs and
synthetic shapes in both orientations, before it is written; see :data:`FP32_TOLERANCE` and
:data:`FP16_TOLERANCE`.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from common import fixtures, require_one_file, variant_parser  # noqa: E402
from mozo.image import load_image  # noqa: E402
from mozo.runtimes import OnnxRunner, TensorRTRunner  # noqa: E402
from mozo.vendors.depth_anything_v3_deploy import MODEL_SPECS, Predictor  # noqa: E402
from mozo.vendors.depth_anything_v3_deploy.graph import (  # noqa: E402
    MAX_PATCHES, MIN_PATCHES, input_names, output_names, profile)
from mozo.weights import resolve  # noqa: E402

FAMILY = "depth_anything_v3"
OPSET = 18

#: The largest difference allowed between the fp32 graph and torch, relative to the depth map's
#: own maximum. Measured through ONNX Runtime on the CPU (MLAS against torch's MKL), on the
#: synthetic shapes and the fixture photograph: 2.7e-7 to 8.0e-5, the worst on the photograph
#: with ``mono-large`` (small 1.9e-5, base 3.4e-5). Five times the worst.
FP32_TOLERANCE = 4e-4

#: The largest mean relative difference over the map allowed between the fp16 engine and torch
#: fp32. TensorRT's fp16 sits farther from fp32 than torch autocast does: on the samples, 2.5e-4
#: to 4.5e-3 for small and base and up to 1.5e-2 for mono-large on its 504x42 strip, against
#: autocast's 8e-5 to 3.2e-3. So the bound is not tied to autocast. It catches a broken graph --
#: well under the model's own error, AbsRel 7-9% on NYUv2 -- and the bench measures what fp16
#: does to accuracy there: AbsRel moved by at most 1e-4 on all three variants.
FP16_TOLERANCE = 2e-2


def _samples() -> list[np.ndarray]:
    """The fixture photographs whole, and crops of the first at the extremes of the shape range.

    Crops of a photograph rather than noise: a depth network's output on noise is meaningless, and
    fp16's distance from fp32 on it measures nothing a user will see.
    """
    images = [load_image(str(p)) for p in fixtures()]
    photo = images[0]
    for h, w in [(480, 640), (640, 480), (900, 900), (100, 1200), (1200, 100)]:
        images.append(np.ascontiguousarray(photo[:h, :w]))
    return images


class _Graph(torch.nn.Module):
    """The network as the graph runs it: pixels and shape inputs in, raw outputs out."""

    def __init__(self, net: torch.nn.Module, sky: bool, autocast: bool) -> None:
        super().__init__()
        self.net, self.sky, self.autocast = net, sky, autocast

    def forward(self, pixels: torch.Tensor, *shape: torch.Tensor):
        with torch.autocast(pixels.device.type, dtype=torch.float16, enabled=self.autocast):
            out = self.net(pixels[:, None], list(shape))
        return tuple(out[k][:, 0].float() for k in (("depth", "sky") if self.sky else ("depth",)))


def _export(predictor: Predictor, destination: Path, autocast: bool) -> None:
    device = predictor.device
    graph = _Graph(predictor.model, predictor.spec.sky, autocast).eval()
    height, width = 378, 504
    pixels = torch.randn(1, 3, height, width, device=device)
    shape = predictor.model.shape_inputs(height, width, device)
    names = input_names(predictor.spec)
    h = torch.export.Dim("h", min=MIN_PATCHES, max=MAX_PATCHES)
    w = torch.export.Dim("w", min=MIN_PATCHES, max=MAX_PATCHES)
    patches, pixels_dims = {2: h, 3: w}, {2: 14 * h, 3: 14 * w}
    dims = [patches] * (len(names) - 1)
    if len(names) > 2:
        dims[-1] = pixels_dims  # uv4 is at the output's resolution
    torch.onnx.export(graph, (pixels, *shape), str(destination), dynamo=True, opset_version=OPSET,
                      input_names=names, output_names=list(output_names(predictor.spec)),
                      dynamic_shapes=(pixels_dims, tuple(dims)), external_data=False)
    require_one_file(destination)


def _check(predictor: Predictor, path: Path, autocast: bool) -> None:
    """Compare the graph with torch on every sample; refuse it if it is too far.

    fp32 runs through ONNX Runtime on the CPU and is held to the largest difference anywhere in
    the map. fp16 runs through TensorRT, which is what it is published for -- through ONNX
    Runtime's CPU kernels mono-large at 504x504 was 6e-3 mean relative, four times autocast's --
    and is held to the mean relative difference (autocast's printed beside it): isolated pixels
    move by percent in either runtime, and a maximum measures those rather than the model.
    """
    device = predictor.device
    if autocast:
        engines = Path(tempfile.mkdtemp(prefix="dav3-engines-"))
        trt = TensorRTRunner(path, device="cuda", engines=engines, shapes=profile(predictor.spec))

        def run(feed: list[torch.Tensor]) -> np.ndarray:
            return trt(*feed)[0][0].float().cpu().numpy()
    else:
        ort = OnnxRunner(path, device="cpu")

        def run(feed: list[torch.Tensor]) -> np.ndarray:
            return ort._session.run(None, {name: t.cpu().numpy()
                                           for name, t in zip(ort.inputs, feed)})[0][0]
    for i, rgb in enumerate(_samples()):
        pixels = predictor.preprocess(rgb)[:, 0].to(device)
        shape = predictor.model.shape_inputs(*pixels.shape[-2:], device)
        with torch.inference_mode():
            want = predictor.model(pixels[:, None], shape)["depth"][0, 0].cpu().numpy()
            got = run([pixels, *shape])
            worst = float(np.abs(got - want).max()) / float(np.abs(want).max())
            mean = float(np.mean(np.abs(got - want) / np.abs(want)))
            if autocast:
                with torch.autocast(device.type, dtype=torch.float16):
                    half = predictor.model(pixels[:, None], shape)["depth"][0, 0].float()
                half = half.cpu().numpy()
                reference = float(np.mean(np.abs(half - want) / np.abs(want)))
                limit = FP16_TOLERANCE
                measured, note = mean, f"mean rel {mean:.2e}; autocast {reference:.2e}, limit {limit:.2e}"
            else:
                limit = FP32_TOLERANCE
                measured, note = worst, f"limit {FP32_TOLERANCE:.0e}; mean rel {mean:.2e}"
        line = (f"  sample {i:2d} {tuple(pixels.shape[-2:])!s:>11}  max|d|/max {worst:.2e}  "
                f"({note})")
        print(line)
        if measured > limit:
            raise SystemExit(f"{line.strip()}: the graph is too far from torch. Not writing it.")
    if autocast:
        del trt
        shutil.rmtree(engines, ignore_errors=True)


def export(variant: str, revision: str | None, weights_dir: Path, fp16: bool) -> None:
    key = "onnx-fp16" if fp16 else "onnx-fp32"
    print(f"\n=== {FAMILY}/{variant} {key}")
    if fp16 and not torch.cuda.is_available():
        raise SystemExit("the fp16 graph is exported on CUDA, under fp16 autocast")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = resolve(FAMILY, variant, "torch-fp32", revision=revision)
    predictor = Predictor.from_pretrained(variant, weights=checkpoint,
                                          device="cuda" if fp16 else "cpu")
    destination = weights_dir / FAMILY / variant / checkpoint.parent.name / f"{key}.onnx"
    with tempfile.TemporaryDirectory() as scratch, torch.no_grad():
        staged = Path(scratch) / f"{key}.onnx"
        _export(predictor, staged, autocast=fp16)
        print(f"  exported  opset {OPSET}, patches {MIN_PATCHES}..{MAX_PATCHES} per side"
              + (", fp16 autocast" if fp16 else ""))
        _check(predictor, staged, autocast=fp16)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(staged, destination)
    print(f"  {key} {destination.stat().st_size / 1e6:.1f} MB")


def main() -> int:
    parser = variant_parser(__doc__, ROOT / "weights", required=True)
    parser.add_argument("--fp16", action="store_true", help="export onnx-fp16 (needs CUDA)")
    args = parser.parse_args()
    unknown = [v for v in args.variants if v not in MODEL_SPECS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}. Known: {list(MODEL_SPECS)}")
    for variant in args.variants:
        export(variant, None, args.weights_dir, args.fp16)
    print("\nMerge the new artifacts into mozo/manifest.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Export RTMO's dense network to ONNX and drop the graphs into the local ``weights/`` tree.

Runs once, on a machine you control, and never ships.

    python tools/export/rtmo.py tiny small medium large   # onnx-fp32
    python tools/export/rtmo.py small --fp16        # onnx-fp16 (needs CUDA)

**What the graph is.** :class:`~mozo.vendors.rtmo_deploy.network.RTMO`'s ``forward``: the
letterboxed BGR batch in, one row per grid cell out -- scores, decoded boxes, visibilities and pose
vectors. Everything data-dependent stays in the vendor: the score filter, the sort, NMS, and the
DCC, which runs only on the people NMS kept and so has no fixed shape to export. The DCC's weights
are the checkpoint's, which is why a graph runtime fetches the checkpoint too
(``registry.NEEDS``). Fixed input (1x3x640x640, 416 for ``tiny``): the vendor letterboxes every
frame to that square.

``onnx-fp32``  exported on the CPU, opset 17. Published to build a TensorRT engine from (``auto``
               takes ``tensorrt-fp32`` on CUDA where TensorRT 11+ is installed) and for serving
               without torch's forward.

``onnx-fp16``  ``--fp16``: measured, **not published**. Exported on CUDA under fp16 autocast
               (opset 18), so the graph carries
               autocast's types and a strongly typed TensorRT engine runs exactly those. **The
               grouped convolutions of the pose branch are kept in fp32**: TensorRT 11.5 has no
               fp16 kernel for a grouped convolution with an odd number of channels per group,
               which ``tiny`` (13) and ``medium`` (27) have, and refuses to build. For one rule
               across variants they are fp32 in all four. On 500 COCO images the engine moved
               visible joints by 0.7-1.0 px at p99 (fp32: 0.0001) and was faster than the fp32
               engine only on ``medium`` and ``large`` (graph 3.6/4.6 ms against 5.2/8.6 ms; a
               whole call 5% and 21% shorter), not on ``tiny`` and ``small``: not offered.

Each graph is verified before it is written, on the fixture photographs, through ONNX Runtime on
the CPU, against the torch model it came from: the same people above 0.5, and boxes, scores and
visible joints within the bounds below. A graph that fails is not written.
"""

from __future__ import annotations

import contextlib
import shutil
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from common import fixtures, require_one_file, variant_parser  # noqa: E402
from mozo.image import load_image  # noqa: E402
from mozo.runtimes import OnnxRunner  # noqa: E402
from mozo.vendors.rtmo_deploy import Predictor  # noqa: E402
from mozo.vendors.rtmo_deploy.network import ConvModule  # noqa: E402
from mozo.weights import resolve  # noqa: E402

INPUT_NAME = "images"
#: Named for what :meth:`Predictor.dense` returns, in its order.
OUTPUT_NAMES = ("scores", "boxes", "visibility", "pose")
OPSET = 17
FP16_OPSET = 18

THRESHOLD = 0.5
#: fp32 bounds. Measured through ONNX Runtime on the CPU against torch, 500 COCO images, all four
#: variants: every joint within 0.0024 px, scores within 5e-6, no person gained or lost.
FP32_PX, FP32_SCORE = 0.05, 1e-4
#: fp16 bounds, for the same check: visible joints moved up to 38 px on one person of 665
#: (``small``), so this only catches a broken graph, not fp16's ordinary imprecision.
FP16_PX, FP16_SCORE = 5.0, 0.02


def _compare(name: str, want: tuple, got: tuple, px: float, score: float) -> str:
    """Hold one fixture's decoded people (input pixels) from the graph against torch's."""
    boxes, scores, keypoints, visible = want
    if len(scores) != len(got[1]):
        raise SystemExit(f"{name}: torch found {len(scores)} people, the graph {len(got[1])}. "
                         "Not writing this artifact.")
    if not len(scores):
        return f"  {name:<24}   0 people"
    box = float((boxes - got[0]).abs().max())
    seen = visible > 0.5
    joint = float((keypoints - got[2]).abs().max(-1).values[seen].max()) if seen.any() else 0.0
    moved = float((scores - got[1]).abs().max())
    if box > px or joint > px or moved > score:
        raise SystemExit(f"{name}: boxes {box:.4f} px, visible joints {joint:.4f} px, scores "
                         f"{moved:.5f} (bounds {px} px, {score}). Not writing this artifact.")
    return (f"  {name:<24} {len(scores):>3} people   boxes {box:.4f} px   visible joints "
            f"{joint:.4f} px   scores {moved:.5f}")


def _grouped_in_fp32(model: torch.nn.Module) -> None:
    """Hold *model*'s grouped convolutions in fp32 under autocast. See the module docstring.

    Per instance, on the model being exported only: the reference predictor is left as it is.
    """
    for module in model.modules():
        if isinstance(module, ConvModule) and module.conv.groups > 1:
            original = module.forward

            def forward(x, original=original):
                with torch.autocast("cuda", enabled=False):
                    return original(x.float())
            module.forward = forward


def export(variant: str, revision: str | None, weights_dir: Path, fp16: bool) -> None:
    key = "onnx-fp16" if fp16 else "onnx-fp32"
    print(f"\n=== rtmo/{variant} {key}")
    checkpoint = resolve("rtmo", variant, "torch-fp32", revision=revision)
    reference = Predictor(checkpoint, variant, "cpu")
    side = reference.spec.input_size

    frames = []
    for path in fixtures():
        batch, _, _ = reference.preprocess(load_image(str(path)))
        frames.append((path.name, batch, reference.decode(reference.dense(batch), THRESHOLD)))

    if fp16:
        if not torch.cuda.is_available():
            raise SystemExit("the fp16 graph is exported on CUDA, under fp16 autocast")
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        model, device, opset, px, score = (Predictor(checkpoint, variant, "cuda").model, "cuda",
                                           FP16_OPSET, FP16_PX, FP16_SCORE)
        _grouped_in_fp32(model)
        context = torch.autocast("cuda", torch.float16)
    else:
        model, device, opset, px, score = reference.model, "cpu", OPSET, FP32_PX, FP32_SCORE
        context = contextlib.nullcontext()

    destination = weights_dir / "rtmo" / variant / checkpoint.parent.name / f"{key}.onnx"
    with tempfile.TemporaryDirectory() as scratch:
        staged = Path(scratch) / f"{key}.onnx"
        with context, torch.no_grad():
            torch.onnx.export(model, (torch.zeros(1, 3, side, side, device=device),), str(staged),
                              input_names=[INPUT_NAME], output_names=list(OUTPUT_NAMES),
                              opset_version=opset, dynamo=False)
        require_one_file(staged)
        print(f"  exported  {side}x{side}, opset {opset}, outputs {', '.join(OUTPUT_NAMES)}")

        runner = OnnxRunner(staged, device="cpu")
        for name, batch, want in frames:
            got = tuple(torch.from_numpy(a).float() for a in runner(batch.numpy()))
            print(_compare(name, want, reference.decode(got, THRESHOLD), px, score))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(staged, destination)
    print(f"  {key} {destination.stat().st_size / 1e6:.1f} MB")


def main() -> int:
    parser = variant_parser(__doc__, ROOT / "weights", required=True, revision=None)
    parser.add_argument("--fp16", action="store_true", help="export the onnx-fp16 graph (CUDA)")
    args = parser.parse_args()
    for variant in args.variants:
        export(variant, args.revision, args.weights_dir, args.fp16)
    print("\nRun tools/generate_manifest.py to pick these up.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

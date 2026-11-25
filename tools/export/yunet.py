#!/usr/bin/env python3
"""Export YuNet's network to ONNX and drop the graphs into the local ``weights/`` tree.

Runs once, on a machine you control, and never ships.

    python tools/export/yunet.py small base           # onnx-fp32
    python tools/export/yunet.py small base --fp16    # onnx-fp16 (needs CUDA)

**What the graph is.** :class:`~mozo.vendors.yunet_deploy.network.YuNet`'s ``forward``: the padded
BGR batch in, the twelve raw maps out (class, box, objectness and landmark logits/offsets at
strides 8, 16 and 32), named ``cls_8`` .. ``kps_32``. Unlike upstream's own ONNX export, nothing is
flattened or passed through a sigmoid inside the graph, so the decode is the vendor's on every
runtime. Height and width are dynamic (any multiple of 32): the vendor runs every image at its own
size.

``onnx-fp32``  exported on the CPU, opset 17.
``onnx-fp16``  ``--fp16``: exported on CUDA under fp16 autocast (opset 18), so a strongly typed
               TensorRT engine runs autocast's types.

Each graph is verified before it is written, through ONNX Runtime on the CPU, against the torch
model it came from, on the fixture photographs at three sizes (so the dynamic axes are exercised):
the same faces above 0.5, and boxes, landmarks and scores within the bounds below. A graph that
fails is not written. Whether a graph is *published* is decided by measurement; see the family's
PROVENANCE.

The export is byte-for-byte repeatable, **but not across ``MKL_CBWR``**: the exporter folds batch
norm into the convolutions on the CPU, and ``MKL_CBWR=AVX2,STRICT`` moves the folded weights by up
to 2.4e-07. The published graphs were exported without it; export them the same way to reproduce
their sha256.
"""

from __future__ import annotations

import contextlib
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from common import fixtures, require_one_file, variant_parser  # noqa: E402
from mozo.adapters.yunet import GRAPH_INPUT, GRAPH_OUTPUTS  # noqa: E402
from mozo.image import load_image  # noqa: E402
from mozo.runtimes import OnnxRunner  # noqa: E402
from mozo.vendors.yunet_deploy import Predictor  # noqa: E402
from mozo.vendors.yunet_deploy.postprocess import decode  # noqa: E402
from mozo.vendors.yunet_deploy.preprocess import to_batch  # noqa: E402
from mozo.weights import resolve  # noqa: E402

#: The names the adapter maps a graph's outputs by, so the two cannot drift apart.
INPUT_NAME, OUTPUT_NAMES = GRAPH_INPUT, GRAPH_OUTPUTS
OPSET = 17
FP16_OPSET = 18

THRESHOLD = 0.5
#: fp32 bounds, through ONNX Runtime on the CPU against torch. The measured worst case is in the
#: PROVENANCE; these only have to catch a broken graph.
FP32_PX, FP32_SCORE = 0.05, 1e-4
#: fp16 bounds, for the same check: catch a broken graph, not fp16's ordinary imprecision.
FP16_PX, FP16_SCORE = 3.0, 0.02


class Flat(torch.nn.Module):
    """The network with its four lists of three maps returned as one flat tuple of twelve."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        return tuple(m for maps in self.model(x) for m in maps)


def _frames(reference: Predictor):
    """Fixture photographs at their own size, halved, and at an odd size that needs padding."""
    for path in fixtures():
        image = load_image(str(path))
        height, width = image.shape[:2]
        for label, picture in (("full", image),
                               ("half", cv2.resize(image, (width // 2, height // 2))),
                               ("odd", cv2.resize(image, (777, 333)))):
            batch = to_batch(picture, reference.spec.size_divisor, torch.device("cpu"))
            want = decode(reference.network(batch), reference.spec.strides, THRESHOLD,
                          reference.spec.nms_threshold)
            yield f"{path.stem}/{label}", batch, want


def _compare(name: str, want: tuple, got: tuple, px: float, score: float) -> str:
    """Hold one image's faces from the graph against torch's."""
    boxes, scores, points = want
    if len(scores) != len(got[1]):
        raise SystemExit(f"{name}: torch found {len(scores)} faces, the graph {len(got[1])}. "
                         "Not writing this artifact.")
    if not len(scores):
        return f"  {name:<24}   0 faces"
    box = float((boxes - got[0]).abs().max())
    point = float((points - got[2]).abs().max())
    moved = float((scores - got[1]).abs().max())
    if box > px or point > px or moved > score:
        raise SystemExit(f"{name}: boxes {box:.4f} px, landmarks {point:.4f} px, scores "
                         f"{moved:.5f} (bounds {px} px, {score}). Not writing this artifact.")
    return (f"  {name:<24} {len(scores):>3} faces   boxes {box:.5f} px   landmarks {point:.5f} px"
            f"   scores {moved:.6f}")


def export(variant: str, revision: str | None, weights_dir: Path, fp16: bool) -> None:
    key = "onnx-fp16" if fp16 else "onnx-fp32"
    print(f"\n=== yunet/{variant} {key}")
    checkpoint = resolve("yunet", variant, "torch-fp32", revision=revision)
    reference = Predictor(checkpoint, variant, "cpu")
    frames = list(_frames(reference))

    if fp16:
        if not torch.cuda.is_available():
            raise SystemExit("the fp16 graph is exported on CUDA, under fp16 autocast")
        model, device, opset, px, score = (Predictor(checkpoint, variant, "cuda").model, "cuda",
                                           FP16_OPSET, FP16_PX, FP16_SCORE)
        context = torch.autocast("cuda", torch.float16)
    else:
        model, device, opset, px, score = reference.model, "cpu", OPSET, FP32_PX, FP32_SCORE
        context = contextlib.nullcontext()

    destination = weights_dir / "yunet" / variant / checkpoint.parent.name / f"{key}.onnx"
    with tempfile.TemporaryDirectory() as scratch:
        staged = Path(scratch) / f"{key}.onnx"
        axes = {INPUT_NAME: {2: "height", 3: "width"}}
        axes.update({name: {2: f"{name}_h", 3: f"{name}_w"} for name in OUTPUT_NAMES})
        with context, torch.no_grad():
            torch.onnx.export(Flat(model), (torch.zeros(1, 3, 640, 640, device=device),),
                              str(staged), input_names=[INPUT_NAME],
                              output_names=list(OUTPUT_NAMES), dynamic_axes=axes,
                              opset_version=opset, dynamo=False)
        require_one_file(staged)
        print(f"  exported  dynamic HxW, opset {opset}, {len(OUTPUT_NAMES)} outputs")

        runner = OnnxRunner(staged, device="cpu")
        graph = Predictor(None, variant, forward=lambda b: [torch.from_numpy(a)
                                                             for a in runner(b.numpy())])
        for name, batch, want in frames:
            got = decode(graph.network(batch), reference.spec.strides, THRESHOLD,
                         reference.spec.nms_threshold)
            print(_compare(name, want, got, px, score))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(staged, destination)
    print(f"  {key} {destination.stat().st_size / 1e3:.0f} KB")


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

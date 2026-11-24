#!/usr/bin/env python3
"""Export OWLv2's vision tower to ONNX and drop it into the local ``weights/`` tree.

This runs once, on a machine you control, and never ships. The graph is published for one reason:
it is what mozo builds a TensorRT engine from on the user's machine, because an engine is specific
to the GPU and TensorRT version that built it and cannot be published itself.

    python tools/export/owlv2.py base-ensemble
    python tools/export/owlv2.py base-ensemble base large-ensemble large

**The vision tower only**: pixels in, one feature per patch out -- the class-token merge and the
final layer norm included, the heads not. That is 89% of a cold call (``tools/bench/owlv2.py``), and
the half TensorRT speeds up. The text tower is cached per vocabulary and the heads are a few small
matmuls, so both stay in the torch checkpoint, which every load fetches beside this graph
(``registry.NEEDS``). SAM 3's graph is the same shape of artifact, for the same reason.

``onnx-fp32``    ``pixels`` ``(1, 3, S, S)`` normalised -> ``patches`` ``(1, P, width)``, opset 17.
                 Fixed at batch 1 and at the variant's own side, because the adapter runs one
                 image at a time at the published resolution and an engine built for one shape is
                 the fastest one TensorRT makes.

Every export is checked against the torch model it came from before it is written, on CPU, through
ONNX Runtime: the patch features, and then the detections the torch heads make from them, against
the same heads on torch's own features. A graph that disagrees is not published.

**What the graph is not for.** mozo does not run it through ONNX Runtime: it was measured twice as
slow as torch on the CPU, because the attention over 3,601 tokens traces to MatMul, Softmax, MatMul
where torch dispatches one fused kernel (``PROVENANCE.md``, under Export). ``registry.EXECUTES``
says so, and asking for ``onnx-fp32`` by name says it is published but not run by this family.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
# This file is called owlv2.py, so its own directory would shadow the package it imports.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from common import fixtures, require_one_file, variant_parser  # noqa: E402
from mozo.image import load_image  # noqa: E402
from mozo.runtimes import OnnxRunner  # noqa: E402
from mozo.vendors.owlv2_deploy.config import SPECS  # noqa: E402
from mozo.vendors.owlv2_deploy.image import preprocess, to_original  # noqa: E402
from mozo.vendors.owlv2_deploy.predictor import Detector  # noqa: E402
from mozo.weights import resolve  # noqa: E402
from tools.fetch.owlv2 import REVISION  # noqa: E402

INPUT_NAME = "pixels"
OUTPUT_NAME = "patches"
OPSET = 17

#: Phrases the detections are compared under. Ordinary objects, so the fixture yields real
#: detections rather than near-threshold noise.
VOCABULARY = ["person", "car", "bicycle", "dog", "bag", "building", "tree", "sign"]

#: What the export may move. Measured on the fixture through ONNX Runtime on the CPU, for the
#: four variants: features 1.9e-04 to 2.7e-03, boxes at most 0.08 px over each variant's 50 best
#: patches, scores at most 1.5e-05.
#: The tolerances sit above that with room, and far below what a caller could see.
FEATURE_TOLERANCE = 5e-3
BOX_TOLERANCE_PX = 1.0
SCORE_TOLERANCE = 1e-3


class _VisionGraph(torch.nn.Module):
    """``OwlV2.encode_image`` as a module, so it traces alone."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixels):
        return self.model.encode_image(pixels)


def _export(model, size: int, destination: Path) -> None:
    torch.onnx.export(
        _VisionGraph(model),
        (torch.zeros(1, 3, size, size),),
        str(destination),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        opset_version=OPSET,
        dynamo=False,
    )


def _candidates(model, patches, queries, query_mask, shape):
    """Every patch's box in source pixels, score and label, from the torch heads."""
    logits, boxes, _ = model.detect(patches, queries, query_mask)
    best = logits[0].max(dim=-1)
    return to_original(boxes[0], shape), torch.sigmoid(best.values), best.indices


def _compare(name: str, want_patches, want, got_patches, got) -> str:
    """Compare one image between torch and the graph, patch by patch.

    OWLv2 returns one candidate per patch, so the same index is the same candidate on both sides,
    and no matching is needed. Boxes and labels are compared on the patches torch scores above
    0.1 -- the ones a caller could see -- and on its 50 best whatever they score, so a variant
    that finds nothing in the fixture under this vocabulary (``large`` does not) is still
    checked. Scores are compared on every patch.

    Raises:
        SystemExit: If features, a box, a score or a label moved beyond tolerance.
    """
    (wb, ws, wl), (gb, gs, gl) = want, got
    seen = ws > 0.1
    seen[ws.topk(50).indices] = True
    feature = float((want_patches - got_patches).abs().max())
    score = float((ws - gs).abs().max())
    box = float((wb[seen] - gb[seen]).abs().max())
    flips = int((wl[seen] != gl[seen]).sum())
    if feature > FEATURE_TOLERANCE:
        raise SystemExit(f"{name}: patch features moved {feature:.2e}. Not writing this artifact.")
    if box > BOX_TOLERANCE_PX or score > SCORE_TOLERANCE or flips:
        raise SystemExit(f"{name}: a box moved {box:.3f} px, scores {score:.2e}, {flips} labels "
                         "changed. Not writing this artifact.")
    return (f"  {name:<24} {int(seen.sum()):>4} patches   features {feature:.2e}   "
            f"boxes {box:.4f} px   scores {score:.2e}")


def export_variant(variant: str, revision: str | None, weights_dir: Path) -> None:
    """Export one variant, verify it against its own torch detections, and place it in the tree."""
    print(f"\n=== owlv2/{variant}")
    checkpoint = resolve("owlv2", variant, revision=revision)
    detector = Detector(checkpoint, variant, device="cpu")
    model, size = detector.model, detector.image_size
    queries, query_mask = detector.encode_text(VOCABULARY)

    reference = []
    for path in fixtures():
        image = load_image(str(path))
        pixels = preprocess(image, size)
        patches = detector.encode_pixels(pixels)
        shape = image.shape[:2]
        reference.append((path.name, shape, pixels, patches,
                          _candidates(model, patches, queries, query_mask, shape)))

    destination = weights_dir / "owlv2" / variant / checkpoint.parent.name / "onnx-fp32.onnx"
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Staged in a directory of its own beside the destination: a verification failure must not
    # leave a bad artifact where the manifest generator would pick it up, and a passing one is
    # moved into place rather than copied.
    with tempfile.TemporaryDirectory(dir=destination.parent) as scratch:
        staged = Path(scratch) / "onnx-fp32.onnx"
        _export(model, size, staged)
        print(f"  exported  {size}x{size}, opset {OPSET}, {INPUT_NAME} -> {OUTPUT_NAME}, batch 1")
        require_one_file(staged)

        runner = OnnxRunner(staged, device="cpu")
        for name, shape, pixels, want_patches, want in reference:
            got_patches = torch.from_numpy(runner(pixels.numpy())[0])
            got = _candidates(model, got_patches, queries, query_mask, shape)
            print(_compare(name, want_patches, want, got_patches, got))

        os.replace(staged, destination)

    print(f"  onnx-fp32 {destination.stat().st_size / 1e6:.1f} MB")


def main() -> int:
    parser = variant_parser(__doc__, ROOT / "weights", required=True, revision=REVISION)
    args = parser.parse_args()

    unknown = [v for v in args.variants if v not in SPECS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}. Known: {list(SPECS)}")

    for variant in args.variants:
        export_variant(variant, args.revision, args.weights_dir)

    print(f"\n{len(args.variants)} graphs written. Run tools/generate_manifest.py to publish them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

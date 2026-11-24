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

``onnx-fp16``    ``pixels`` ``(1, 3, S, S)`` normalised fp32 -> ``patches`` ``(1, P, width)`` fp32,
                 opset 18. Fixed at batch 1 and at the variant's own side, because the adapter runs
                 one image at a time at the published resolution and an engine built for one
                 shape is the fastest one TensorRT makes.

**fp16 is in the graph's own types**: it is exported from the tower after ``half_trunk()``, so it
carries exactly ``torch-fp16``'s precision split -- embeddings and blocks fp16, both layer norms
fp32. TensorRT 11 builds strongly typed engines only and has no fp16 switch, so an fp16 engine
needs an fp16 graph; it is also half the download of an fp32 one. Exported through
``torch.export`` on CUDA, where the tower runs in fp16.

Every export is checked before it is written, on CPU, through ONNX Runtime. The reference is the
fp32 model, and the yardstick is ``torch-fp16`` itself: on each fixture, the graph's detections may
be no farther from fp32 than ``torch-fp16``'s are, with a 2x margin -- boxes on the patches a caller
could see, scores on every patch, and no more label flips. Comparing patch features directly says
nothing here: fp16 alone moves them by about 0.5. A graph that fails is not published.

**What the graph is not for.** mozo does not run it through ONNX Runtime: it was measured twice as
slow as torch on the CPU, because the attention over 3,601 tokens traces to MatMul, Softmax, MatMul
where torch dispatches one fused kernel (``PROVENANCE.md``, under Export). ``registry.EXECUTES``
says so, and asking for ``onnx-fp16`` by name says it is published but not run by this family.
"""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
# This file is called owlv2.py, so its own directory would shadow the package it imports.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from common import fixtures, require_one_file, variant_parser  # noqa: E402
from mozo.image import load_image  # noqa: E402
from mozo.runtimes import OnnxRunner, free_device_memory  # noqa: E402
from mozo.vendors.owlv2_deploy.config import SPECS  # noqa: E402
from mozo.vendors.owlv2_deploy.image import preprocess, to_original  # noqa: E402
from mozo.vendors.owlv2_deploy.predictor import Detector  # noqa: E402
from mozo.weights import resolve  # noqa: E402
from tools.fetch.owlv2 import REVISION  # noqa: E402

INPUT_NAME = "pixels"
OUTPUT_NAME = "patches"
OPSET = 18
ARTIFACT = "onnx-fp16"

#: Phrases the detections are compared under. Ordinary objects, so the fixture yields real
#: detections rather than near-threshold noise.
VOCABULARY = ["person", "car", "bicycle", "dog", "bag", "building", "tree", "sign"]

#: How much farther from fp32 the graph may be than ``torch-fp16`` is, and a floor for when
#: ``torch-fp16`` happens to land almost exactly. Measured on the fixture through ONNX Runtime on
#: the CPU, graph against torch-fp16, both against fp32: boxes 0.18-2.27 px against 0.54-2.10 on
#: the patches above 0.1 (11.2-24.8 against 3.1-14.5 over the 50 best, where the box head is
#: unstable even in fp32); scores 0.0010-0.0043 against 0.0004-0.0050; no label flips on either.
MARGIN = 2.0
BOX_FLOOR_PX = 1.0
SCORE_FLOOR = 1e-3


class _VisionGraph(torch.nn.Module):
    """``OwlV2.encode_image`` as a module, so it traces alone."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixels):
        return self.model.encode_image(pixels)


def _export(model, size: int, destination: Path) -> None:
    with torch.no_grad():
        torch.onnx.export(
            _VisionGraph(model).eval(),
            (torch.zeros(1, 3, size, size, device="cuda"),),
            str(destination),
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            opset_version=OPSET,
            dynamo=True,
            external_data=False,  # one file: an artifact is one file
        )


def _candidates(model, patches, queries, query_mask, shape):
    """Every patch's box in source pixels, score and label, from the torch heads."""
    logits, boxes, _ = model.detect(patches, queries, query_mask)
    best = logits[0].max(dim=-1)
    return to_original(boxes[0], shape), torch.sigmoid(best.values), best.indices


@dataclass
class _Drift:
    """How far one image's detections are from fp32's."""

    box: float
    score: float
    flips: int

    @classmethod
    def between(cls, want, got) -> "_Drift":
        """OWLv2 returns one candidate per patch, so the same index is the same candidate on both
        sides and no matching is needed. Boxes and labels on the patches fp32 scores above 0.1 --
        the ones a caller could see -- and on its 50 best whatever they score, so a variant that
        finds nothing in the fixture under this vocabulary (``large`` does not) is still checked.
        Scores on every patch."""
        (wb, ws, wl), (gb, gs, gl) = want, got
        seen = ws > 0.1
        seen[ws.topk(50).indices] = True
        return cls(float((wb[seen] - gb[seen]).abs().max()), float((ws - gs).abs().max()),
                   int((wl[seen] != gl[seen]).sum()))


def _compare(name: str, half: _Drift, graph: _Drift) -> str:
    """Hold the graph to ``torch-fp16``'s own distance from fp32.

    Raises:
        SystemExit: If the graph is farther from fp32 than :data:`MARGIN` times torch-fp16.
    """
    box_limit = max(MARGIN * half.box, BOX_FLOOR_PX)
    score_limit = max(MARGIN * half.score, SCORE_FLOOR)
    if graph.box > box_limit or graph.score > score_limit or graph.flips > half.flips:
        raise SystemExit(
            f"{name}: the graph is farther from fp32 than torch-fp16 is -- boxes "
            f"{graph.box:.2f} px (limit {box_limit:.2f}), scores {graph.score:.4f} (limit "
            f"{score_limit:.4f}), {graph.flips} label flips against {half.flips}. Not writing "
            "this artifact.")
    return (f"  {name:<24} boxes {graph.box:6.2f} px (torch-fp16 {half.box:6.2f})   "
            f"scores {graph.score:.4f} ({half.score:.4f})   flips {graph.flips}")


def export_variant(variant: str, revision: str | None, weights_dir: Path) -> None:
    """Export one variant, verify it against torch-fp16's distance from fp32, and place it."""
    print(f"\n=== owlv2/{variant}")
    checkpoint = resolve("owlv2", variant, revision=revision)
    reference = Detector(checkpoint, variant, device="cpu")
    half = Detector(checkpoint, variant, device="cuda", half_trunk=True)
    model, size = reference.model, reference.image_size
    queries, query_mask = reference.encode_text(VOCABULARY)

    def candidates(patches, shape):
        return _candidates(model, patches.float().cpu(), queries, query_mask, shape)

    images = []
    with torch.no_grad():
        for path in fixtures():
            image = load_image(str(path))
            pixels, shape = preprocess(image, size), image.shape[:2]
            want = candidates(reference.encode_pixels(pixels), shape)
            images.append((path.name, shape, pixels, want,
                           _Drift.between(want, candidates(half.encode_pixels(pixels.cuda()),
                                                           shape))))

    destination = weights_dir / "owlv2" / variant / checkpoint.parent.name / f"{ARTIFACT}.onnx"
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Staged in a directory of its own beside the destination: a verification failure must not
    # leave a bad artifact where the manifest generator would pick it up, and a passing one is
    # moved into place rather than copied.
    with tempfile.TemporaryDirectory(dir=destination.parent) as scratch:
        staged = Path(scratch) / f"{ARTIFACT}.onnx"
        _export(half.model, size, staged)
        del half
        free_device_memory()
        print(f"  exported  {size}x{size}, opset {OPSET}, {INPUT_NAME} -> {OUTPUT_NAME}, batch 1")
        require_one_file(staged)

        runner = OnnxRunner(staged, device="cpu")
        for name, shape, pixels, want, half_drift in images:
            got = candidates(torch.from_numpy(runner(pixels.numpy())[0]), shape)
            print(_compare(name, half_drift, _Drift.between(want, got)))

        os.replace(staged, destination)

    print(f"  {ARTIFACT} {destination.stat().st_size / 1e6:.1f} MB")


def main() -> int:
    parser = variant_parser(__doc__, ROOT / "weights", required=True, revision=REVISION)
    args = parser.parse_args()

    unknown = [v for v in args.variants if v not in SPECS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}. Known: {list(SPECS)}")

    if not torch.cuda.is_available():
        raise SystemExit("the fp16 tower is exported on CUDA, where it runs in fp16")
    torch.backends.cuda.matmul.allow_tf32 = False
    for variant in args.variants:
        export_variant(variant, args.revision, args.weights_dir)

    print(f"\n{len(args.variants)} graphs written. Run tools/generate_manifest.py to publish them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

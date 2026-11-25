#!/usr/bin/env python3
"""Export DINOv2 backbones to ONNX and drop them into the local ``weights/`` tree.

Runs once, on a machine you control, and never ships.

    python tools/export/dinov2.py small base large small-reg base-reg
    python tools/export/dinov2.py small --fp16          # on CUDA

**One fixed shape, 1x3x224x224.** Preprocessing always produces the central 224x224 crop, and
the adapter runs one image per forward pass (a vector must not depend on its batch-mates), so
nothing else is ever fed. A fixed shape is also the only honest one: the position embedding is
interpolated from the trained 37x37 grid to 16x16 with Python arithmetic, which tracing bakes in
as constants, so a "dynamic" graph would be wrong at every other size.

The graph is the backbone alone, pixels to the final-norm class token (``cls``). Preprocessing and
the L2 normalisation stay in Python, the same code on every runtime.

**No xFormers path exists to trace.** The vendored package dropped upstream's xFormers branches,
so the export needs no ``XFORMERS_DISABLED``; this script refuses to run if ``xformers`` was
imported anyway.

``onnx-fp32``  exported from the torch model on the CPU. Checked on the fixtures through ONNX
               Runtime against torch: see :data:`FP32_TOLERANCE`.
``onnx-fp16``  ``--fp16``: exported on CUDA under fp16 autocast, so the graph carries autocast's
               types -- matrix products in fp16, norms and softmax in fp32 -- for a strongly typed
               TensorRT engine to run exactly those. Checked against torch fp32 on the CPU: its
               cosine to fp32 may fall short of torch autocast's own by :data:`FP16_MARGIN` at most.

``giant`` is not exported: at 4.5 GB in fp32 (2.3 GB in fp16) it is past protobuf's 2 GB limit,
and an artifact is one file.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
# This file is called dinov2.py; its own directory must not shadow anything it imports.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
# Exports from the tree you just built, not from the published bucket.
os.environ.setdefault("MOZO_BASE_URL", f"file://{ROOT / 'weights'}")

from common import fixtures, require_one_file, variant_parser  # noqa: E402
from mozo.image import load_image  # noqa: E402
from mozo.runtimes import OnnxRunner  # noqa: E402
from mozo.vendors.dinov2_deploy import SPECS  # noqa: E402
from mozo.vendors.dinov2_deploy.checkpoint import load_network  # noqa: E402
from mozo.vendors.dinov2_deploy.config import CROP  # noqa: E402
from mozo.vendors.dinov2_deploy.image import preprocess  # noqa: E402
from mozo.weights import resolve  # noqa: E402

INPUT_NAME = "pixels"
OUTPUT_NAME = "cls"
OPSET = 17
FP16_OPSET = 18

#: The variants a graph can be published for: everything under protobuf's 2 GB.
GRAPH_VARIANTS = tuple(variant for variant in SPECS if variant != "giant")

#: Largest ``max|delta|`` allowed between the fp32 graph (ONNX Runtime, CPU) and torch on a
#: normalised vector. Measured worst on the fixtures, all five variants: see the plan's Results.
FP32_TOLERANCE = 1e-5

#: How much farther from fp32 (1 - cosine) the fp16 graph may be than torch autocast is, with a
#: floor for when autocast happens to land almost exactly.
FP16_MARGIN = 2.0
FP16_FLOOR = 1e-4


def images() -> list[np.ndarray]:
    """The repository's photographs, plus odd-margin noise and a flat image. A short check of what
    is written; ``tools/verify/dinov2.py`` and ``tools/bench/dinov2.py`` measure more."""
    rng = np.random.default_rng(0)
    return [load_image(str(p)) for p in fixtures()] + [
        rng.integers(0, 256, (256, 343, 3), dtype=np.uint8), np.zeros((224, 224, 3), np.uint8)]


class _Backbone(torch.nn.Module):
    """The network behind a one-argument forward, with its 224x224 position embedding frozen.

    Upstream's forward is ``forward(*args, **kwargs)``, which the tracer flattens into a second
    graph input, ``masks``. And the register models resize the position embedding with antialiased
    bicubic, which ONNX has no operator for; at a fixed input size that resize is a constant
    anyway, so it is computed here once, by the model's own code, and the graph adds the result.
    """

    def __init__(self, network) -> None:
        super().__init__()
        self.network = network
        grid = (CROP // network.patch_size) ** 2
        with torch.no_grad():
            tokens = torch.zeros(1, 1 + grid, network.embed_dim, device=network.pos_embed.device)
            self.register_buffer("pos", network.interpolate_pos_encoding(tokens, CROP, CROP).clone())

    def _frozen(self, x, w, h):
        assert w == h == CROP, f"the graph is fixed at {CROP}x{CROP}, not {w}x{h}"
        return self.pos

    def forward(self, pixels):
        # The network reads its position embedding through this one method; for the trace it
        # reads the frozen one.
        self.network.interpolate_pos_encoding = self._frozen
        try:
            return self.network(pixels)
        finally:
            del self.network.interpolate_pos_encoding


def _export(network, destination: Path, fp16: bool) -> None:
    """fp32 on the CPU at opset 17; fp16 under CUDA autocast at opset 18, which can export its casts."""
    dummy = torch.randn(1, 3, CROP, CROP, device="cuda" if fp16 else "cpu")
    with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=fp16):
        torch.onnx.export(_Backbone(network).eval(), (dummy,), str(destination),
                          input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
                          opset_version=FP16_OPSET if fp16 else OPSET, dynamo=False)


def _vectors(run, pixels: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([F.normalize(torch.as_tensor(run(x[None])).float(), dim=1, p=2)
                      for x in pixels])


def export(variant: str, revision: str | None, weights_dir: Path, fp16: bool) -> None:
    if variant not in GRAPH_VARIANTS:
        raise SystemExit(f"dinov2/{variant}: no graph -- past protobuf's 2 GB limit as one file")
    key = "onnx-fp16" if fp16 else "onnx-fp32"
    print(f"\n=== dinov2/{variant} {key}")
    checkpoint = resolve("dinov2", variant, "torch-fp32", revision=revision)
    destination = weights_dir / "dinov2" / variant / checkpoint.parent.name / f"{key}.onnx"
    reference = load_network(SPECS[variant], checkpoint, "cpu")
    pixels = [preprocess(image) for image in images()]
    with torch.inference_mode():
        want = _vectors(reference, pixels)

    if fp16:
        if not torch.cuda.is_available():
            raise SystemExit("the fp16 graph is exported on CUDA, under fp16 autocast")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        network = reference.to("cuda")  # ``want`` is already on the host; no second load

        def autocast(x):
            with torch.autocast("cuda", torch.float16):
                return network(x.cuda()).float().cpu()
        with torch.inference_mode():
            budget = float(1 - (_vectors(autocast, pixels) * want).sum(1).min())
    else:
        network = reference

    if "xformers" in sys.modules:
        raise SystemExit("xformers was imported; the vendored package must not need it")
    with tempfile.TemporaryDirectory() as scratch:
        staged = Path(scratch) / f"{key}.onnx"
        _export(network, staged, fp16)
        require_one_file(staged)
        runner = OnnxRunner(staged, device="cpu")
        got = _vectors(lambda x: runner(x.numpy())[0], pixels)
        if fp16:
            drift = float(1 - (got * want).sum(1).min())
            limit = max(FP16_MARGIN * budget, FP16_FLOOR)
            print(f"  1 - cosine to torch fp32: graph {drift:.2e}, torch autocast {budget:.2e} "
                  f"(limit {limit:.2e})")
            if drift > limit:
                raise SystemExit("the fp16 graph is farther from fp32 than autocast is; not writing it")
        else:
            delta = float((got - want).abs().max())
            print(f"  max|delta| against torch on {len(pixels)} images: {delta:.2e} "
                  f"(limit {FP32_TOLERANCE:.0e})")
            if delta > FP32_TOLERANCE:
                raise SystemExit("the graph disagrees with torch; not writing it")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(staged, destination)
    print(f"  {key} {destination.stat().st_size / 1e6:.1f} MB -> {destination}")


def main() -> int:
    parser = variant_parser(__doc__, ROOT / "weights", required=True, revision=None)
    parser.add_argument("--fp16", action="store_true", help="export the onnx-fp16 graph (CUDA)")
    args = parser.parse_args()
    for variant in args.variants:
        export(variant, args.revision, args.weights_dir, args.fp16)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

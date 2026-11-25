#!/usr/bin/env python3
"""Check the vendored DEIM against the authors' repository, module by module, and mozo against
the vendor.

    python tools/verify/deim.py --upstream DEIM/                   # every variant, the fixtures
    python tools/verify/deim.py --upstream DEIM/ --coco val2017/   # plus 16 COCO photographs
    python tools/verify/deim.py --upstream DEIM/ --variant small photo.jpg
    python tools/verify/deim.py --upstream DEIM/ --falsify all     # prove the gate can fail
    python tools/verify/deim.py --no-reference                     # vendor vs mozo only

``DEIM/`` is a clone of ``github.com/ShihuaHuang95/DEIM`` at commit ``09d35d5`` (see the vendor's
PROVENANCE); it needs ``torch``, ``torchvision``, ``pyyaml``, ``scipy``, ``tensorboard``,
``faster-coco-eval`` and ``calflops`` importable, because importing upstream's ``engine`` imports
its training code. So this does not run in the test suite.

Three paths over the same checkpoint and the same decoded pixels:

1. **upstream**, run by ``tools/verify/deim_reference.py`` in a fresh process per variant (its
   docstring says why it must be fresh): what ``tools/inference/torch_inf.py`` does;
2. **the vendor**, ``mozo.vendors.deim_deploy``;
3. **mozo**, ``ModelManager().get_model("deim", variant, runtime="torch-fp32")``: registry,
   weights, adapter, PixelFlow.

Upstream against the vendor is compared exactly: the network input, the encoder's outputs, the
raw logits and boxes and all 300 post-processed detections with ``torch.equal``, and every
module's output in call order by the sha256 of its bytes (stricter than ``torch.equal``: it also
tells ``-0.0`` from ``0.0``). The first module that differs is named. The vendor against mozo is
compared after both go through PixelFlow's rounding (``tests/conftest.py::as_pixelflow_reports``),
at two thresholds.

CPU only, and under ``MKL_CBWR=AVX2,STRICT`` (set below, inherited by the reference process):
that is where the comparison is defined. See the comment at the setting.

``--falsify NAME`` perturbs one thing in the vendor and expects the gate to fail at the stage that
thing reaches and at no earlier one. See :data:`FALSIFY`.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Before torch loads MKL, and inherited by the reference process. Without it the comparison is not
# defined: MKL's sgemm picks its kernel by the buffers' alignment, which differs from process to
# process, so the same model on the same pixels gave a different encoder attention output in about
# one process in ten on an idle machine (medium: 1 of 12, logits up to ~5e-4 apart). Strict
# conditional-numerical-reproducibility mode makes the result independent of alignment. It sets
# the same arithmetic on both sides; it does not make either side match the other.
os.environ.setdefault("MKL_CBWR", "AVX2,STRICT")

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "verify"))
os.environ.setdefault("MOZO_BASE_URL", f"file://{ROOT / 'weights'}")

import mozo  # noqa: E402
from common import fixtures  # noqa: E402
from conftest import as_pixelflow_reports  # noqa: E402
from deim_reference import record_digests  # noqa: E402
from mozo.image import load_image  # noqa: E402
from mozo.vendors.deim_deploy import SPECS, load, postprocess, preprocess  # noqa: E402
from mozo.weights import resolve  # noqa: E402

FAMILY = "deim"
VARIANTS = tuple(SPECS)
THRESHOLDS = (0.4, 0.05)
STAGES = ("input", "modules", "encoder", "logits", "boxes", "detections", "mozo")

#: Module outputs that differ in *structure*, not arithmetic. Upstream's ``TransformerDecoder``
#: returns six stacked tensors for its training losses; the vendor's returns the two the eval path
#: reads, unstacked. Its values reach the comparison anyway, as the parent's ``pred_logits`` and
#: ``pred_boxes``.
STRUCTURAL = {"decoder.decoder"}


def _synthetic(photo: np.ndarray) -> dict[str, np.ndarray]:
    """The awkward cases: nothing to find, a tiny image, extreme aspect ratios both ways, an image
    already at the network's size (no resize), an upscale, and noise at an odd size."""
    rng = np.random.default_rng(0)
    return {
        "blank": np.full((480, 640, 3), 114, np.uint8),
        "black": np.zeros((640, 640, 3), np.uint8),
        "tiny": np.ascontiguousarray(photo[::97, ::61]),
        "wide": np.ascontiguousarray(photo[500:620]),
        "tall": np.ascontiguousarray(photo[:, 900:1000]),
        "exact-640": np.ascontiguousarray(photo[:640, :640]),
        "upscale": np.ascontiguousarray(photo[100:300, 200:500]),
        "noise": rng.integers(0, 256, (333, 517, 3), dtype=np.uint8),
    }


def images(paths: list[Path], coco: Path | None, count: int) -> dict[str, np.ndarray]:
    """Named RGB arrays: the photographs, the synthetic cases, and *count* COCO images."""
    found = {p.stem: load_image(str(p)) for p in paths}
    found.update(_synthetic(next(iter(found.values()))))
    if coco is not None:
        rng = np.random.default_rng(0)
        files = sorted(coco.glob("*.jpg"))
        for i in sorted(rng.choice(len(files), count, replace=False)):
            found[f"coco-{files[i].stem}"] = load_image(str(files[i]))
    return found


def run_reference(upstream: Path, variant: str, checkpoint: Path, inputs: Path, work: Path) -> dict:
    out = work / f"reference-{variant}.pt"
    subprocess.run(
        [sys.executable, str(ROOT / "tools" / "verify" / "deim_reference.py"),
         "--upstream", str(upstream), "--config", str(upstream / SPECS[variant].config),
         "--checkpoint", str(checkpoint), "--inputs", str(inputs), "--out", str(out)],
        check=True)
    return torch.load(out, weights_only=False)


def vendor_stages(model, rgb: np.ndarray) -> dict:
    store, handles = record_digests(model)
    encoder_out: list = []
    handles.append(model.encoder.register_forward_hook(
        lambda _m, _i, out: encoder_out.append([t.clone() for t in out])))
    batch, size = preprocess(rgb)
    with torch.inference_mode():
        outputs = model(batch)
    labels, boxes, scores = postprocess(outputs, size)
    for handle in handles:
        handle.remove()
    return {"input": batch, "encoder": encoder_out[0], "pred_logits": outputs["pred_logits"],
            "pred_boxes": outputs["pred_boxes"], "labels": labels, "boxes": boxes,
            "scores": scores, "digests": dict(store)}


def _equal(a, b) -> bool:
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)


def first_divergent_module(vendor: dict, reference: dict) -> str | None:
    """The first module, in the vendor's call order, whose output bytes differ from upstream's.

    Every vendor module must exist upstream; upstream may have modules the vendor never calls
    (the ``nn.Identity`` it applies for a pre-score nothing reads)."""
    for name, calls in vendor.items():
        if name in STRUCTURAL:
            continue
        if reference.get(name) != calls:
            return name
    return None


def compare(vendor: dict, reference: dict) -> dict[str, bool]:
    module = first_divergent_module(vendor["digests"], reference["digests"])
    return {
        "input": _equal(vendor["input"], reference["input"]),
        "modules": module is None,
        "encoder": _equal(vendor["encoder"], reference["encoder"]),
        "logits": _equal(vendor["pred_logits"], reference["pred_logits"]),
        "boxes": _equal(vendor["pred_boxes"], reference["pred_boxes"]),
        "detections": all(_equal(vendor[k], reference[k]) for k in ("labels", "boxes", "scores")),
        "_module": module,
    }


def mozo_matches(model, rgb: np.ndarray, vendor: dict) -> bool:
    """mozo's result against the vendor's 300 detections, filtered and rounded the same way."""
    for threshold in THRESHOLDS:
        keep = vendor["scores"][0] > threshold
        want = as_pixelflow_reports(vendor["boxes"][0][keep].numpy(),
                                    vendor["scores"][0][keep].numpy(),
                                    vendor["labels"][0][keep].numpy())
        rows = model.predict(rgb, threshold=threshold).to_dict()
        got_boxes = np.array([r["bbox"] for r in rows], np.float64).reshape(-1, 4)
        got_scores = np.array([r["confidence"] for r in rows], np.float64)
        got_ids = np.array([r["class_id"] for r in rows])
        if not (np.array_equal(got_boxes, want[0]) and np.array_equal(got_scores, want[1])
                and np.array_equal(got_ids, vendor["labels"][0][keep].numpy())):
            return False
    return True


# -- falsification ---------------------------------------------------------------------------

@contextlib.contextmanager
def _patch(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def _resize_bicubic():
    from PIL import Image
    real = Image.Image.resize
    return _patch(Image.Image, "resize",
                  lambda self, size, resample=None, *a, **k: real(self, size, Image.BICUBIC))


def _scale_reciprocal():
    """``x * (1 / 255)`` instead of ``x / 255``: the same number in exact arithmetic, not in
    float32. Patched on ``Tensor.div``, which only the preprocessing calls with 255."""
    original = torch.Tensor.div

    def multiply_by_reciprocal(self, other, *a, **k):
        if isinstance(other, (int, float)) and other == 255:
            return self * (1 / 255)
        return original(self, other, *a, **k)
    return _patch(torch.Tensor, "div", multiply_by_reciprocal)


def _encoder_gelu_tanh():
    import torch.nn as nn
    import mozo.vendors.deim_deploy.utils as utils
    real = utils.get_activation
    return _patch(utils, "get_activation",
                  lambda act, inplace=True: nn.GELU(approximate="tanh") if act == "gelu"
                  else real(act, inplace))


def _decoder_layernorm_eps():
    import torch.nn as nn
    import mozo.vendors.deim_deploy.decoder as decoder
    return _patch(decoder.nn, "LayerNorm", lambda d, *a, **k: nn.modules.normalization.LayerNorm(d, eps=1e-6))


def _weighting_training_form():
    """W(n) in float32 throughout, as upstream's training path builds it, instead of the deploy
    form's doubles. Reaches only the boxes' distribution integral."""
    import mozo.vendors.deim_deploy.decoder as decoder

    def training_form(reg_max, up, reg_scale):
        upper_bound1 = abs(up[0]) * abs(reg_scale)
        upper_bound2 = abs(up[0]) * abs(reg_scale) * 2
        step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
        left = [-(step) ** i + 1 for i in range(reg_max // 2 - 1, 0, -1)]
        right = [(step) ** i - 1 for i in range(1, reg_max // 2)]
        values = [-upper_bound2] + left + [torch.zeros_like(up[0][None])] + right + [upper_bound2]
        return torch.cat([v.reshape(1) for v in values], 0)
    return _patch(decoder, "weighting_function", training_form)


def _box_scale_swapped():
    import mozo.vendors.deim_deploy.postprocess as post
    real = post.torchvision.ops.box_convert
    return _patch(post.torchvision.ops, "box_convert",
                  lambda boxes, in_fmt, out_fmt: real(boxes, in_fmt, out_fmt)[..., [1, 0, 3, 2]])


#: name -> (perturbation, the first stage it must fail at). ``modules`` comes before the tensors
#: it contains, so a perturbation inside the network fails there first; the falsifier also checks
#: that the stage *it* reaches fails, not merely that something did.
FALSIFY = {
    "resize": (_resize_bicubic, "input", "input"),
    "scale": (_scale_reciprocal, "input", "input"),
    "gelu": (_encoder_gelu_tanh, "modules", "encoder"),
    "layernorm-eps": (_decoder_layernorm_eps, "modules", "logits"),
    "weighting": (_weighting_training_form, "modules", "boxes"),
    "box-scale": (_box_scale_swapped, "detections", "detections"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="*", type=Path)
    parser.add_argument("--upstream", type=Path, help="DEIM clone at the pinned commit")
    parser.add_argument("--variant", action="append", choices=VARIANTS)
    parser.add_argument("--coco", type=Path, help="a COCO image folder to add photographs from")
    parser.add_argument("--coco-count", type=int, default=16)
    parser.add_argument("--falsify", choices=[*FALSIFY, "all"])
    parser.add_argument("--no-reference", action="store_true")
    args = parser.parse_args()
    if not args.no_reference and args.upstream is None:
        parser.error("--upstream is required unless --no-reference")

    torch.set_grad_enabled(False)
    named = images(args.images or fixtures(), args.coco, args.coco_count)
    variants = args.variant or list(VARIANTS)
    print(f"torch {torch.__version__}, {len(named)} images, variants {variants}")
    failures = 0

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        inputs = work / "inputs.npz"
        np.savez(inputs, **named)
        for variant in variants:
            checkpoint = resolve(FAMILY, variant, "torch-fp32")
            reference = None
            if not args.no_reference:
                reference = run_reference(args.upstream, variant, checkpoint, inputs, work)
                if variant == variants[0]:
                    print("upstream:", reference["versions"])
                reference = reference["results"]

            if args.falsify:
                names = list(FALSIFY) if args.falsify == "all" else [args.falsify]
                for name in names:
                    perturb, first, reaches = FALSIFY[name]
                    with perturb():
                        model = load(variant, checkpoint)
                        got = {k: compare(vendor_stages(model, rgb), reference[k]) for k, rgb in named.items()}
                    failed = [s for s in STAGES[:-1] if not all(g[s] for g in got.values())]
                    caught = bool(failed) and failed[0] == first and reaches in failed
                    module = next((g["_module"] for g in got.values() if g["_module"]), None)
                    print(f"  falsify {variant:<14} {name:<14} {'CAUGHT' if caught else 'MISSED'}: "
                          f"fails at {failed or 'nothing'}, first module {module}")
                    failures += not caught
                continue

            model = load(variant, checkpoint)
            served = mozo.ModelManager().get_model(FAMILY, variant, runtime="torch-fp32", device="cpu")
            rows = []
            for key, rgb in named.items():
                vendor = vendor_stages(model, rgb)
                result = compare(vendor, reference[key]) if reference else {}
                result["mozo"] = mozo_matches(served, rgb, vendor)
                ok = all(v for k, v in result.items() if not k.startswith("_"))
                failures += not ok
                count = int((vendor["scores"][0] > THRESHOLDS[0]).sum())
                rows.append(ok)
                if not ok:
                    print(f"  FAIL {variant} {key}: {result}")
                print(f"  {variant:<14} {key:<22} {count:>3} dets  "
                      + " ".join(f"{s}={'ok' if result[s] else 'FAIL'}" for s in STAGES if s in result))
            print(f"{variant}: {sum(rows)}/{len(rows)} images exact")

    print("PASS" if not failures else f"FAIL ({failures})")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

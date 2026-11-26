#!/usr/bin/env python3
"""Check the vendored DEIM against the authors' repository, module by module, and mozo against
the vendor.

    python tools/verify/deim.py --upstream DEIM/ --device cuda     # the primary gate, on a GPU
    python tools/verify/deim.py --upstream DEIM/ --device cuda --coco val2017/ \\
        --reference-cache refs/                                     # plus 16 COCO photographs
    python tools/verify/deim.py --device cuda --coco val2017/ --reference-cache refs/ \\
        --falsify all                                               # prove the gate can fail
    python tools/verify/deim.py --upstream DEIM/                   # the CPU gate (the default)
    python tools/verify/deim.py --upstream DEIM/ --variant small photo.jpg
    python tools/verify/deim.py --no-reference                     # vendor vs mozo only

``DEIM/`` is a clone of ``github.com/ShihuaHuang95/DEIM`` at commit ``09d35d5`` (see the vendor's
PROVENANCE); it needs ``torch``, ``torchvision``, ``pyyaml``, ``scipy``, ``tensorboard``,
``faster-coco-eval``, ``calflops`` and ``transformers`` importable, because importing upstream's
``engine`` imports its training code. So this does not run in the test suite.

Three paths over the same checkpoint and the same decoded pixels, in one process per variant
(several variants run as one child process each: upstream's config registry leaks between configs,
see ``deim_reference.py``):

1. **upstream**, ``deim_reference.Upstream``: what ``tools/inference/torch_inf.py`` does on the
   device, decode included;
2. **the vendor**, ``mozo.vendors.deim_deploy``, run the same way: batch made on the host, network
   and decode on the device;
3. **mozo**, ``ModelManager().get_model("deim", variant, runtime="torch-fp32", device=...)``:
   registry, weights, adapter, PixelFlow.

Upstream against the vendor is compared exactly: the network input, the encoder's outputs, the
raw logits and boxes and all 300 post-processed detections, and every module's output in call
order by the sha256 of its bytes (stricter than ``torch.equal``: it also tells ``-0.0`` from
``0.0``). The first module that differs is named. The vendor against mozo is compared exactly on
the arrays mozo hands to PixelFlow, and again on PixelFlow's rounded rows
(``tests/conftest.py::as_pixelflow_reports``), at two thresholds.

``--device cuda`` is the primary gate: one GPU, torch's default CUDA settings, which mozo leaves
alone (cuDNN may use TF32 for convolutions; matmul TF32, ``cudnn.benchmark`` and deterministic
algorithms off). Each side first runs every image twice and must repeat itself bit for bit.
``--dtype`` is fp32 only: DEIM's only torch runtime is ``torch-fp32`` (``auto`` on CUDA is a
TensorRT engine, measured by parity and accuracy instead), and upstream has no half inference path.

On the CPU, ``MKL_CBWR=AVX2,STRICT`` is forced (set below): that is where the CPU comparison is
defined. See the comment at the setting.

``--reference-cache DIR`` keeps upstream's records per variant (the input and encoder outputs as
digests, module outputs as digests, the raw outputs and 300 detections whole), read when present,
computed and saved when not. ``--falsify NAME`` perturbs one thing in the vendor and expects the
gate to fail at the stage that thing reaches and at no earlier one, after an unperturbed run has
passed against the same reference. See :data:`FALSIFY`; five of them perturb only what a GPU does.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, NamedTuple
from unittest import mock

# Before torch loads MKL, and inherited by the per-variant processes. Without it the comparison
# is not defined: MKL's sgemm picks its kernel by the buffers' alignment, which differs from process to
# process, so the same model on the same pixels gave a different encoder attention output in about
# one process in ten on an idle machine (medium: 1 of 12, logits up to ~5e-4 apart). Strict
# conditional-numerical-reproducibility mode makes the result independent of alignment. It sets
# the same arithmetic on both sides; it does not make either side match the other.
os.environ["MKL_CBWR"] = "AVX2,STRICT"  # forced: a weaker mode left in the environment would undefine the gate

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
from deim_reference import digest, record_digests  # noqa: E402
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


def _cache_form(record: dict) -> dict:
    """What a reference keeps of one image: the network input and the encoder's outputs as their
    digests (dtype, shape, sha256 of the bytes), the small raw outputs and all 300 detections
    whole. The full intermediates of 25 images run to gigabytes."""
    return {**record, "input": digest(record["input"]), "encoder": digest(record["encoder"])}


def _same_record(a: dict, b: dict) -> bool:
    """Whether two records of one side, in cache form, hold the same bytes at every stage."""
    return (a["input"] == b["input"] and a["encoder"] == b["encoder"]
            and a["digests"] == b["digests"]
            and all(_equal(a[k], b[k]) for k in ("pred_logits", "pred_boxes", "labels", "boxes", "scores")))


def _reference_key(args, named: dict) -> dict:
    """What a cached reference was made for. A cache made for anything else is refused."""
    device = torch.device(args.device)
    return {"device": args.device, "dtype": args.dtype, "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "images": {k: hashlib.sha256(v.tobytes()).hexdigest() + str(v.shape) for k, v in named.items()}}


def reference_for(variant: str, args, named: dict, checkpoint: Path) -> dict:
    """Upstream's records for every image, in cache form: read from ``--reference-cache`` when it
    holds this variant, otherwise computed here (on CUDA each image twice, which must repeat bit
    for bit) and saved there."""
    key = _reference_key(args, named)
    path = args.reference_cache / f"{variant}.pt" if args.reference_cache else None
    if path is not None and path.exists():
        cached = torch.load(path, weights_only=False)
        if cached["for"] != key:
            raise SystemExit(f"{path} was made for another device, dtype, torch, GPU or image set")
        print(f"upstream: read from {path} ({cached['versions']}, commit {cached['commit']})")
        return cached["results"]
    if args.upstream is None:
        raise SystemExit(f"no cached reference for {variant} and no --upstream to compute one")

    from deim_reference import Upstream, versions
    device = torch.device(args.device)
    upstream = Upstream(args.upstream, args.upstream / SPECS[variant].config, checkpoint, device)
    commit = subprocess.run(["git", "-C", str(args.upstream), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or "?"
    print(f"upstream: {versions()}, commit {commit}")
    results, unrepeated = {}, []
    for name, rgb in named.items():
        results[name] = _cache_form(upstream(rgb))
        if device.type == "cuda" and not _same_record(results[name], _cache_form(upstream(rgb))):
            unrepeated.append(name)
    if device.type == "cuda":
        print(f"  upstream repeats itself on {len(named) - len(unrepeated)}/{len(named)} images"
              + (f"; not on {unrepeated}" if unrepeated else ""))
        if unrepeated:
            raise SystemExit("upstream is not deterministic here, so no comparison means anything")
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"for": key, "results": results, "versions": versions(), "commit": commit}, path)
        print(f"upstream: saved to {path}")
    return results


def vendor_stages(model, rgb: np.ndarray, device: torch.device) -> dict:
    """The vendor run the way upstream's script runs: batch and size made on the host and moved
    to the device, network and decode there. Every stage is copied back to the host (exact)."""
    store, handles = record_digests(model)
    encoder_out: list = []
    handles.append(model.encoder.register_forward_hook(
        lambda _m, _i, out: encoder_out.append([t.detach().to("cpu", copy=True) for t in out])))
    batch, size = preprocess(rgb)
    batch, size = batch.to(device), size.to(device)
    with torch.inference_mode():
        outputs = model(batch)
        labels, boxes, scores = postprocess(outputs, size)
    for handle in handles:
        handle.remove()
    return {"input": batch.cpu(), "encoder": encoder_out[0],
            "pred_logits": outputs["pred_logits"].cpu(), "pred_boxes": outputs["pred_boxes"].cpu(),
            "labels": labels.cpu(), "boxes": boxes.cpu(), "scores": scores.cpu(),
            "digests": dict(store)}


def _equal(a, b) -> bool:
    """Tensors, lists of them, or a cached digest of either on one side (compared by digest)."""
    live = (torch.Tensor, list)
    if not isinstance(a, live) or not isinstance(b, live):
        return (digest(a) if isinstance(a, live) else a) == (digest(b) if isinstance(b, live) else b)
    if isinstance(a, list):
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
    """mozo's result against the vendor's 300 detections, at each threshold, twice: exactly, on
    the arrays mozo hands to PixelFlow, and on the rows PixelFlow gives back, after the vendor's
    arrays go through the same rounding. The rows alone could not see a last-bit difference
    (PixelFlow keeps two or three decimals), such as a decode on the host instead of the device."""
    import pixelflow as pf

    handed: list = []
    real = pf.detections.from_arrays

    def spy(**arrays):
        handed.append(arrays)
        return real(**arrays)

    for threshold in THRESHOLDS:
        keep = vendor["scores"][0] > threshold
        exact = {"boxes": vendor["boxes"][0][keep].numpy(), "scores": vendor["scores"][0][keep].numpy(),
                 "class_ids": vendor["labels"][0][keep].numpy()}
        want = as_pixelflow_reports(exact["boxes"], exact["scores"], exact["class_ids"])
        with mock.patch.object(pf.detections, "from_arrays", spy):
            rows = model.predict(rgb, threshold=threshold).to_dict()
        if not all(handed[-1][k].dtype == v.dtype and np.array_equal(handed[-1][k], v)
                   for k, v in exact.items()):
            return False
        got_boxes = np.array([r["bbox"] for r in rows], np.float64).reshape(-1, 4)
        got_scores = np.array([r["confidence"] for r in rows], np.float64)
        got_ids = np.array([r["class_id"] for r in rows])
        if not (np.array_equal(got_boxes, want[0]) and np.array_equal(got_scores, want[1])
                and np.array_equal(got_ids, exact["class_ids"])):
            return False
    return True


# -- falsification ---------------------------------------------------------------------------

def _patch(obj, name, value):
    return mock.patch.object(obj, name, value)


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
    import mozo.vendors.deim_deploy.hybrid_encoder as encoder
    real = encoder.get_activation
    return _patch(encoder, "get_activation",
                  lambda act, inplace=True: nn.GELU(approximate="tanh") if act == "gelu"
                  else real(act, inplace))


def _decoder_layernorm_eps():
    """LayerNorm eps 1e-5 -> 1e-6 in the decoder only, through a stand-in for its ``nn``."""
    import types

    import torch.nn as nn
    import mozo.vendors.deim_deploy.decoder as decoder
    stand_in = types.SimpleNamespace(**{k: getattr(nn, k) for k in dir(nn) if not k.startswith("__")})
    stand_in.LayerNorm = lambda d, *a, **k: nn.LayerNorm(d, eps=1e-6)
    return _patch(decoder, "nn", stand_in)


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
    import torchvision.ops

    real = torchvision.ops.box_convert
    return _patch(torchvision.ops, "box_convert",
                  lambda boxes, in_fmt, out_fmt: real(boxes, in_fmt, out_fmt)[..., [1, 0, 3, 2]])


def _frozen_bn_as_batchnorm():
    """``FrozenBatchNorm2d`` computed as eval-mode ``F.batch_norm``: the same statistics, the
    arithmetic the other checkpoints use. Reaches only ``xlarge-obj365``'s backbone."""
    import torch.nn.functional as F
    import mozo.vendors.deim_deploy.hgnetv2 as hgnetv2

    def batch_norm(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                            False, 0.0, self.eps)
    return _patch(hgnetv2.FrozenBatchNorm2d, "forward", batch_norm)


@contextlib.contextmanager
def _setting(obj, name, value):
    """A torch backend flag set for the vendor's run alone (upstream ran before, unperturbed)."""
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def _cudnn_tf32_off():
    """cuDNN convolutions in full fp32, where torch's default (and mozo's) lets them use TF32."""
    return _setting(torch.backends.cudnn, "allow_tf32", False)


def _matmul_tf32_on():
    """cuBLAS matmuls in TF32, where torch's default (and mozo's) keeps them in full fp32. The
    backbone is all convolutions, so the first matmul is in the encoder."""
    return _setting(torch.backends.cuda.matmul, "allow_tf32", True)


def _channels_last():
    """The vendor's batch in channels-last memory, values equal (the input stage still passes):
    cuDNN picks other kernels for the other layout."""
    real = preprocess

    def nhwc(*args, **kwargs):
        batch, size = real(*args, **kwargs)
        return batch.contiguous(memory_format=torch.channels_last), size
    return _patch(sys.modules[__name__], "preprocess", nhwc)


def _attention_sdpa():
    """``need_weights=False`` on every ``nn.MultiheadAttention``, which sends it through
    ``scaled_dot_product_attention`` and whichever backend torch picks (head sizes 16 and 32).
    Upstream calls it with the default ``need_weights=True``: the unfused ``baddbmm`` + softmax
    path, which never reaches SDPA, so no SDPA backend is in use to perturb."""
    import torch.nn as nn

    real = nn.MultiheadAttention.forward

    def fused(self, query, key, value, *args, **kwargs):
        return real(self, query, key, value, *args, **{**kwargs, "need_weights": False})
    return _patch(nn.MultiheadAttention, "forward", fused)


def _decode_on_host():
    """The decode on the host, as mozo's adapter did before the GPU gate, while upstream decodes
    on the device its network ran on. CUDA's ``sigmoid`` is not the CPU's in the last bit."""
    real = postprocess
    return _patch(sys.modules[__name__], "postprocess",
                  lambda outputs, size, *a, **k: real({n: t.cpu() for n, t in outputs.items()},
                                                      size.cpu(), *a, **k))


class Falsification(NamedTuple):
    perturb: Callable
    first: str      # the first stage it must fail at (``modules`` precedes the tensors it holds)
    reaches: str    # the stage its effect is aimed at, which must fail too
    applies: Callable = lambda spec: True   # variants it can reach; skipped elsewhere
    cuda_only: bool = False                 # a perturbation of what only a GPU does


FALSIFY = {
    "resize": Falsification(_resize_bicubic, "input", "input"),
    "scale": Falsification(_scale_reciprocal, "input", "input"),
    "gelu": Falsification(_encoder_gelu_tanh, "modules", "encoder"),
    "layernorm-eps": Falsification(_decoder_layernorm_eps, "modules", "logits"),
    "weighting": Falsification(_weighting_training_form, "modules", "boxes"),
    "box-scale": Falsification(_box_scale_swapped, "detections", "detections"),
    "frozen-bn": Falsification(_frozen_bn_as_batchnorm, "modules", "encoder",
                               lambda spec: spec.freeze_norm),
    "tf32-off": Falsification(_cudnn_tf32_off, "modules", "encoder", cuda_only=True),
    "matmul-tf32": Falsification(_matmul_tf32_on, "modules", "encoder", cuda_only=True),
    "channels-last": Falsification(_channels_last, "modules", "encoder", cuda_only=True),
    "attention-sdpa": Falsification(_attention_sdpa, "modules", "encoder", cuda_only=True),
    "decode-host": Falsification(_decode_on_host, "detections", "detections", cuda_only=True),
}


def environment(device: torch.device) -> str:
    """torch, CUDA, cuDNN, the GPU, its driver and the settings in force: what a result holds for."""
    if device.type == "cpu":
        return f"torch {torch.__version__}, CPU, MKL_CBWR={os.environ.get('MKL_CBWR', '')}"
    driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader",
                             "-i", str(device.index or 0)], capture_output=True, text=True)
    return (f"torch {torch.__version__}, CUDA {torch.version.cuda}, cuDNN "
            f"{torch.backends.cudnn.version()}, {torch.cuda.get_device_name(device)}, driver "
            f"{driver.stdout.strip() or 'unknown'}; cudnn.allow_tf32 "
            f"{torch.backends.cudnn.allow_tf32}, matmul.allow_tf32 "
            f"{torch.backends.cuda.matmul.allow_tf32}, cudnn.benchmark "
            f"{torch.backends.cudnn.benchmark}, deterministic "
            f"{torch.are_deterministic_algorithms_enabled()}")


def _without_variants(argv: list[str]) -> list[str]:
    out, skip = [], False
    for arg in argv:
        if skip:
            skip = False
        elif arg == "--variant":
            skip = True
        elif not arg.startswith("--variant="):
            out.append(arg)
    return out


def falsify(variant: str, args, named: dict, checkpoint: Path, reference: dict) -> int:
    """An unperturbed run against the reference first, which must pass (so a falsifier's failure
    is its perturbation, not the reference), then each falsifier that applies here."""
    device = torch.device(args.device)

    def failing():
        model = load(variant, checkpoint).to(device)
        got = {k: compare(vendor_stages(model, rgb, device), reference[k]) for k, rgb in named.items()}
        failed = [s for s in STAGES[:-1] if not all(g[s] for g in got.values())]
        return failed, next((g["_module"] for g in got.values() if g["_module"]), None)

    failed, module = failing()
    print(f"  unperturbed    {variant:<14} {'PASS' if not failed else 'FAIL'}: fails at "
          f"{failed or 'nothing'}, first module {module}")
    if failed:
        return 1
    names = list(FALSIFY) if args.falsify == "all" else [args.falsify]
    failures = 0
    for name in names:
        perturb, first, reaches, applies, cuda_only = FALSIFY[name]
        if not applies(SPECS[variant]) or (cuda_only and device.type != "cuda"):
            continue
        with perturb():
            failed, module = failing()
        caught = bool(failed) and failed[0] == first and reaches in failed
        print(f"  falsify {variant:<14} {name:<14} {'CAUGHT' if caught else 'MISSED'}: "
              f"fails at {failed or 'nothing'}, first module {module}")
        failures += not caught
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="*", type=Path)
    parser.add_argument("--upstream", type=Path, help="DEIM clone at the pinned commit")
    parser.add_argument("--variant", action="append", choices=VARIANTS)
    parser.add_argument("--coco", type=Path, help="a COCO image folder to add photographs from")
    parser.add_argument("--coco-count", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--falsify", choices=[*FALSIFY, "all"])
    parser.add_argument("--reference-cache", type=Path,
                        help="a folder of upstream's records per variant: read when present, "
                             "computed and saved there when not")
    parser.add_argument("--no-reference", action="store_true")
    args = parser.parse_args()
    if not args.no_reference and args.upstream is None and args.reference_cache is None:
        parser.error("--upstream (or a --reference-cache) is required unless --no-reference")
    if args.falsify and args.no_reference:
        parser.error("--falsify compares against upstream, so it needs --upstream")
    if args.dtype != "fp32":
        parser.error("DEIM runs through torch in fp32 only: mozo publishes no torch half runtime "
                     "for it (autocast was slower than fp32 on CUDA for every variant), and "
                     "upstream's inference script has no half-precision path")
    if args.falsify and FALSIFY.get(args.falsify, FALSIFY["resize"]).cuda_only and args.device != "cuda":
        parser.error(f"--falsify {args.falsify} perturbs what only a GPU does: use --device cuda")

    variants = args.variant or list(VARIANTS)
    if len(variants) > 1:
        # One variant per process: upstream's config registry leaks between configs (see
        # deim_reference.py), and upstream, the vendor and mozo share the process they compare in.
        base = [sys.executable, __file__, *_without_variants(sys.argv[1:])]
        failed = [v for v in variants if subprocess.run([*base, "--variant", v]).returncode]
        print("PASS" if not failed else f"FAIL ({', '.join(failed)})")
        return 1 if failed else 0

    variant = variants[0]
    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    named = images(args.images or fixtures(), args.coco, args.coco_count)
    print(environment(device))
    print(f"{variant}: {len(named)} images, {args.dtype}")
    checkpoint = resolve(FAMILY, variant, "torch-fp32")
    reference = None if args.no_reference else reference_for(variant, args, named, checkpoint)

    if args.falsify:
        failures = falsify(variant, args, named, checkpoint, reference)
        print("PASS" if not failures else f"FAIL ({failures})")
        return 0 if not failures else 1

    model = load(variant, checkpoint).to(device)
    served = mozo.ModelManager().get_model(FAMILY, variant, runtime="torch-fp32", device=args.device)
    failures, rows, unrepeated, outputs = 0, [], [], 0
    for key, rgb in named.items():
        vendor = vendor_stages(model, rgb, device)
        outputs += sum(len(c) for n, c in vendor["digests"].items() if n not in STRUCTURAL)
        if device.type == "cuda" and not _same_record(_cache_form(vendor),
                                                      _cache_form(vendor_stages(model, rgb, device))):
            unrepeated.append(key)
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
    if device.type == "cuda":
        print(f"  vendor repeats itself on {len(named) - len(unrepeated)}/{len(named)} images"
              + (f"; not on {unrepeated}" if unrepeated else ""))
        failures += len(unrepeated)
    print(f"{variant}: {sum(rows)}/{len(rows)} images exact; {outputs} module outputs, and per "
          f"image the input, encoder, logits, boxes, 300 detections and mozo at {len(THRESHOLDS)} "
          f"thresholds")
    print("PASS" if not failures else f"FAIL ({failures})")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

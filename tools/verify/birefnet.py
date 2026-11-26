#!/usr/bin/env python3
"""Verify the vendored BiRefNet against the authors' Hugging Face remote code.

    python tools/verify/birefnet.py general --weights weights/birefnet/general/<rev>
    python tools/verify/birefnet.py lite --weights ... --images /path/to/photos
    python tools/verify/birefnet.py general --weights ... --no-falsify
    python tools/verify/birefnet.py general --weights ... --device cuda --reference-cache /tmp/ref

Needs, beside mozo: ``transformers``, ``timm``, ``einops``, ``kornia`` and ``safetensors``
(upstream's remote code imports all of them) and the Hub, or an ``HF_HOME`` that already holds
the pinned revision. None of them is a dependency of mozo.

Checks, in increasing strength:

1. **Standalone** -- the vendored tree imports nothing outside stdlib, ``torch``,
   ``torchvision``, ``numpy``, ``cv2`` and ``PIL``, and never ``mozo``.
2. **The einops rewrite** -- ``image2patches`` against ``einops.rearrange`` itself, at every grid
   the decoder uses (32, 16, 8, 4, 1) and on a non-square input.
3. **Structural** -- the published checkpoint holds upstream's tensors (dtype kept) minus the
   twelve training-only ones, and loads strictly into the vendored network.
4. **Every stage, end to end, on each image** -- the preprocessed tensor against torchvision's
   own transforms; the output of every backbone block, patch merge and stage norm (both passes),
   the squeeze block, every patch conv, deformable conv, ASPP, decoder, lateral and attention
   block, by module name against upstream's module of the same name; the logits; the alpha
   against ``ToPILImage`` + ``resize``; the cut-out against the model card's ``putalpha`` and,
   with ``refine``, against ``handler.py``'s own ``refine_foreground``; mozo's adapter, when the
   weights resolve through ``mozo.weights``.
5. **Falsification** -- perturbations of the vendored code, each of which must fail at the stage
   it reaches. Run by default; ``--no-falsify`` skips them.

Every comparison is ``torch.equal`` / ``np.array_equal``: this is an extraction.

**Device and precision.** ``--device cpu`` (the default) gates float32 on the CPU. ``--device
cuda`` gates, on the GPU, every precision a CUDA torch runtime of mozo uses: ``fp32``
(``torch-fp32``) and ``fp16`` (``torch-fp16``, which ``auto`` picks on CUDA without TensorRT, and
always for ``hr``) -- float16 **autocast** around the network, mozo's mechanism. ``--dtype``
picks one. ``--dtype half`` gates the authors' own CUDA path instead, which no mozo runtime uses:
the model card's and ``handler.py``'s ``model.half()`` and ``input.half()``, against
``Predictor(dtype=torch.float16)``. Both sides always run in this one process, on the same device,
under the same precision mechanism, with torch's global settings (TF32, cuDNN) as mozo leaves
them: nothing is flipped to get a match. ``--repeat`` (2 on CUDA) runs each side that many times
per image and requires every run byte-identical to the first.

**The reference is pinned to float32 weights and eager attention.** ``from_pretrained`` is given
``dtype=torch.float32``: transformers 5 defaults to ``dtype="auto"``, which loads the
float16-stored checkpoints (``general``, ``hr``) as a float16 model. The upstream module's own
``SDPA_enabled`` is ``False``. Parameters are cloned after loading so both sides compute from
freshly allocated tensors (``VENDORING.md`` §4). Run with ``MKL_CBWR=AVX2,STRICT`` when the two
sides run in different processes; here they share one. ``handler.py``'s
``torch.set_float32_matmul_precision('high')`` is not applied to either side: mozo never sets it.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
# Named for the package it verifies: its own directory leaves sys.path first.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from common import fixtures  # noqa: E402
from fetch.birefnet import CHECKPOINTS, TRAINING_ONLY  # noqa: E402
from mozo.image import load_image  # noqa: E402
from mozo.vendors.birefnet_deploy import MODEL_SPECS, Predictor  # noqa: E402
from mozo.vendors.birefnet_deploy import blocks, image as vendor_image, network, swin  # noqa: E402
from mozo.vendors.birefnet_deploy import predictor as vendor_predictor  # noqa: E402

VENDOR = ROOT / "mozo" / "vendors" / "birefnet_deploy"
ALLOWED_THIRD_PARTY = {"torch", "torchvision", "numpy", "cv2", "PIL"}

#: Modules whose every output is compared, by type. Leaf convolutions inside these are covered
#: by the block that contains them; the deformable conv is named because it is the one operator
#: an exporter replaces.
STAGES = (swin.PatchEmbed, swin.SwinTransformerBlock, swin.PatchMerging, torch.nn.LayerNorm,
          blocks.DeformableConv2d, blocks.ASPPDeformable, blocks.BasicDecBlk, blocks.BasicLatBlk,
          blocks.SimpleConvs, torch.nn.Sequential)

#: Precision -> how both sides run. ``fp16`` is mozo's ``torch-fp16`` runtime (autocast around the
#: network); ``half`` is upstream's own ``model.half()`` path, gated only on request.
PRECISIONS = ("fp32", "fp16", "half")


# --------------------------------------------------------------------------------------------
# 1, 2

def check_standalone() -> list[str]:
    problems = []
    for path in sorted(VENDOR.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom) and not node.level
                     else [])
            for name in names:
                root = name.split(".")[0]
                if root == "mozo" or (root not in ALLOWED_THIRD_PARTY
                                      and root not in sys.stdlib_module_names):
                    problems.append(f"{path.name}: imports {name}")
    return problems


def check_rewrite() -> list[tuple[str, bool]]:
    from einops import rearrange

    results = []
    for (H, W), grid in [((1024, 1024), g) for g in (32, 16, 8, 4, 1)] + [((2048, 2048), 32),
                                                                          ((96, 160), 4)]:
        x = torch.randn(2, 3, H, W)
        ref = torch.empty(1, 1, H // grid, W // grid)
        ours = network.image2patches(x, ref)
        theirs = rearrange(x, 'b c (hg h) (wg w) -> b (c hg wg) h w', hg=grid, wg=grid)
        results.append((f"image2patches {H}x{W} grid {grid}", torch.equal(ours, theirs)))
    return results


# --------------------------------------------------------------------------------------------
# the reference

def load_upstream(variant: str, device: str = "cpu"):
    """Upstream's model at the pinned commit, float32, parameters re-allocated, on *device*; and
    its files."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForImageSegmentation

    repo, commit = CHECKPOINTS[variant].repo, CHECKPOINTS[variant].commit
    model = AutoModelForImageSegmentation.from_pretrained(
        repo, revision=commit, trust_remote_code=True, dtype=torch.float32).eval()
    with torch.no_grad():
        for tensor in list(model.parameters()) + list(model.buffers()):
            tensor.data = tensor.data.clone()
    model.to(device)
    safetensors = hf_hub_download(repo, "model.safetensors", revision=commit)
    handler = hf_hub_download(repo, "handler.py", revision=commit)
    return model, safetensors, handler


def handler_functions(path: str) -> dict:
    """``refine_foreground`` and the two estimator functions out of upstream's ``handler.py``.

    Only those three definitions are executed: the module itself, imported, would call
    ``torch.set_float32_matmul_precision('high')`` and build an endpoint.
    """
    import cv2

    tree = ast.parse(Path(path).read_text())
    wanted = {"refine_foreground", "FB_blur_fusion_foreground_estimator_2",
              "FB_blur_fusion_foreground_estimator"}
    module = ast.Module(body=[n for n in tree.body
                              if isinstance(n, ast.FunctionDef) and n.name in wanted],
                        type_ignores=[])
    scope = {"np": np, "cv2": cv2, "Image": Image}
    exec(compile(module, path, "exec"), scope)
    return scope


def upstream_pipeline(model, rgb: np.ndarray, resolution: int, refine_fn, device: str = "cpu",
                      precision: str = "fp32"):
    """The model card's snippet and ``handler.py``'s refine, run as upstream runs them.

    The transforms run on the host and the tensor moves to *device* afterwards, as in the card;
    ``half`` casts it as the card does (``.to('cuda').half()``), ``fp16`` runs the forward under
    autocast. The sigmoid runs where the logits are, in their dtype, before ``.cpu()``.
    """
    from torchvision import transforms

    image = Image.fromarray(rgb)
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    pixels = transform(image).unsqueeze(0)
    x = pixels.to(device)
    if precision == "half":
        x = x.half()
    with torch.no_grad(), torch.autocast(torch.device(device).type, dtype=torch.float16,
                                         enabled=precision == "fp16"):
        logits = model(x)[-1]
    pred_pil = transforms.ToPILImage()(logits.sigmoid().cpu()[0].squeeze())
    mask = pred_pil.resize(image.size)
    plain = image.copy()
    plain.putalpha(mask)
    refined = refine_fn(image, pred_pil)
    refined.putalpha(pred_pil.resize(image.size))
    return pixels, logits.cpu(), np.asarray(mask), np.asarray(plain), np.asarray(refined)


# --------------------------------------------------------------------------------------------
# 4

def record(model, names: set[str]):
    """Hook every module in *names*; return (outputs by name in call order, remove).

    Each output is copied to the host when it is produced: a snapshot no later in-place op can
    touch, and GPU memory stays free for ``hr``'s 2048x2048 activations."""
    seen: dict[str, list] = {}
    handles = []
    for name, module in model.named_modules():
        if name in names:
            handles.append(module.register_forward_hook(
                lambda m, i, o, name=name: seen.setdefault(name, []).append(
                    (o[0] if isinstance(o, tuple) else o).detach().to("cpu", copy=True))))
    return seen, lambda: [h.remove() for h in handles]


def stage_names(ours) -> list[str]:
    return [name for name, module in ours.model.named_modules()
            if name and isinstance(module, STAGES)]


#: ``--swap-models``: the two networks, of which only the one running stays on the GPU. ``hr``
#: needs 9.3 GB for one 2048x2048 forward and does not fit on a 12 GB card beside a second Swin-L.
RESIDENT: dict = {}


def resident(model) -> None:
    """Park every other registered network on the host and bring *model* to the GPU."""
    if not RESIDENT:
        return
    for other in RESIDENT.values():
        if other is not model:
            other.to("cpu")
    # The parked network's blocks go back to the driver: a 2048 forward wants 3.75 GB in one
    # piece, which the allocator's cached fragments cannot supply.
    torch.cuda.empty_cache()
    model.to("cuda")


def run_upstream(upstream, rgb, resolution, refine_fn, device, precision, names):
    """One upstream pass with every stage recorded: (outputs by name, pipeline results)."""
    resident(upstream)
    seen, remove = record(upstream, set(names))
    try:
        return seen, upstream_pipeline(upstream, rgb, resolution, refine_fn, device, precision)
    finally:
        remove()


def run_ours(ours, rgb, names):
    """One pass through mozo's own ``Predictor.logits`` -- the production call, autocast and
    device placement included -- with every stage recorded: (outputs, host logits, alpha).

    The alpha is post-processed from the logits where they were produced, as ``matte`` does."""
    resident(ours.model)
    seen, remove = record(ours.model, set(names))
    try:
        logits = ours.logits(rgb)
    finally:
        remove()
    return seen, logits.cpu(), vendor_image.postprocess(logits, rgb.shape[:2])


def stage_diff(names, a_seen, b_seen) -> tuple[int, int, str]:
    """(calls compared, mismatches, first mismatching stage) between two recordings."""
    calls = mismatched = 0
    first = ""
    for name in names:
        for a, b in zip(a_seen.get(name, []), b_seen.get(name, [])):
            calls += 1
            if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
                mismatched += 1
                first = first or name
        if len(a_seen.get(name, [])) != len(b_seen.get(name, [])):
            mismatched += 1
            first = first or f"{name} (call count)"
    return calls, mismatched, first


def tensors_equal(a, b) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)


def compare(variant, upstream, ours, rgb, refine_fn, names, device="cpu", precision="fp32",
            repeat=1) -> list[tuple[str, bool, str, int]]:
    """Every stage for one image: (stage, equal, detail, comparisons).

    With *repeat* > 1 each side runs that many times and every later run must equal its first,
    stage by stage: the determinism check (``VENDORING.md`` §2.7)."""
    resolution = MODEL_SPECS[variant].resolution
    up_seen, (up_pixels, up_logits, up_mask, up_plain, up_refined) = run_upstream(
        upstream, rgb, resolution, refine_fn, device, precision, names)

    rows = []
    pixels = vendor_image.preprocess(rgb, resolution)
    rows.append(("pixels", torch.equal(pixels, up_pixels), "", 1))

    our_seen, logits, alpha = run_ours(ours, rgb, names)
    calls, mismatched, first = stage_diff(names, our_seen, up_seen)
    rows.append((f"{calls} module outputs", mismatched == 0,
                 f"first: {first}" if mismatched else "", calls))
    rows.append(("logits", tensors_equal(logits, up_logits), "", 1))
    rows.append(("alpha", np.array_equal(alpha, up_mask), "", 1))
    resident(ours.model)
    rows.append(("cutout", np.array_equal(ours.cutout(rgb), up_plain), "", 1))
    rows.append(("cutout refine", np.array_equal(ours.cutout(rgb, refine=True), up_refined), "", 1))

    for run in range(1, repeat):
        seen, (_, again_logits, again_mask, _, _) = run_upstream(
            upstream, rgb, resolution, refine_fn, device, precision, names)
        calls, mismatched, first = stage_diff(names, seen, up_seen)
        ok = mismatched == 0 and tensors_equal(again_logits, up_logits) \
            and np.array_equal(again_mask, up_mask)
        rows.append((f"upstream run {run + 1} = run 1", ok, f"first: {first}" if not ok else "",
                     calls + 2))
        del seen
        seen, again_logits, again_alpha = run_ours(ours, rgb, names)
        calls, mismatched, first = stage_diff(names, seen, our_seen)
        ok = mismatched == 0 and tensors_equal(again_logits, logits) \
            and np.array_equal(again_alpha, alpha)
        rows.append((f"mozo run {run + 1} = run 1", ok, f"first: {first}" if not ok else "",
                     calls + 2))
        del seen
    return rows


def synthetic() -> list[tuple[str, np.ndarray]]:
    """The awkward inputs: tiny, extreme aspect, flat, exactly the network's size."""
    rng = np.random.default_rng(0)
    return [
        ("1x1", np.full((1, 1, 3), 127, np.uint8)),
        ("3x2000 strip", rng.integers(0, 256, (3, 2000, 3), dtype=np.uint8)),
        ("flat grey 300x200", np.full((200, 300, 3), 128, np.uint8)),
        ("noise 1024x1024", rng.integers(0, 256, (1024, 1024, 3), dtype=np.uint8)),
    ]


# --------------------------------------------------------------------------------------------
# 5

def falsify(variant, upstream, ours, rgb, refine_fn, names, device="cpu", precision="fp32",
            cache: Path | None = None) -> list[tuple[str, bool, str]]:
    """Each perturbation must make the first failing stage the one it reaches.

    The reference never changes between perturbations, so it is computed once -- or read from
    *cache*, written the first time, keyed by variant, device, precision, image and versions."""
    import torch.nn.functional as F
    import torchvision

    def set_attr(obj, name, value):
        original = getattr(obj, name)
        setattr(obj, name, value)
        return lambda: setattr(obj, name, original)

    def each(kind, attr, change):
        touched = [(m, getattr(m, attr)) for m in ours.model.modules() if isinstance(m, kind)]
        for m, v in touched:
            setattr(m, attr, change(v))
        return lambda: [setattr(m, attr, v) for m, v in touched]

    def flag(obj, name, value):
        return set_attr(obj, name, value)

    real_patches, real_resize = network.image2patches, network._resize
    real_deform = blocks.DeformableConv2d.deform
    real_postprocess = vendor_image.postprocess
    real_preprocess = vendor_predictor.preprocess
    resolution = MODEL_SPECS[variant].resolution

    def rounded_postprocess(logits, size):
        small = (logits.sigmoid().cpu()[0].squeeze() * 255).round().byte().numpy()
        return np.asarray(Image.fromarray(small, mode="L").resize(size[::-1], Image.BICUBIC))

    def float32_postprocess(logits, size):  # the branch's postprocess before the GPU gate
        small = logits.float().sigmoid().cpu()[0].squeeze().mul(255).byte().numpy()
        return np.asarray(Image.fromarray(small, mode="L").resize(size[::-1], Image.BICUBIC))

    def half_model():
        """Upstream's ``.half()`` in place of mozo's autocast, on mozo's side only. The float32
        weights are restored from a copy: ``.float()`` would not undo the rounding."""
        saved = {k: v.clone() for k, v in ours.model.state_dict().items()}
        ours.model.half()
        state = (ours.dtype, ours.autocast)
        ours.dtype, ours.autocast = torch.float16, False

        def restore():
            ours.model.float().load_state_dict(saved)
            ours.dtype, ours.autocast = state
        return restore

    cases = [
        ("normalisation mean 0.485 -> 0.486", "pixels",
         lambda: set_attr(vendor_image, "MEAN", (0.486, 0.456, 0.406))),
        ("resize BILINEAR -> BICUBIC", "pixels",
         lambda: set_attr(Image, "BILINEAR", Image.BICUBIC)),
        (f"resolution {resolution} -> {resolution + 32}", "pixels",
         lambda: set_attr(ours, "resolution", resolution + 32)),
        ("LayerNorm eps 1e-5 -> 1e-6", "bb.patch_embed.norm",
         lambda: each(torch.nn.LayerNorm, "eps", lambda e: 1e-6)),
        ("image2patches order (c, wg, hg)", "decoder.ipt_blk5",
         lambda: set_attr(network, "image2patches", lambda x, ref: real_patches(
             x.transpose(-1, -2), ref.transpose(-1, -2)).transpose(-1, -2))),
        ("align_corners True -> False", "bb.patch_embed.norm (2nd pass)",
         lambda: set_attr(network, "_resize", lambda x, size: F.interpolate(
             x, size=size, mode="bilinear", align_corners=False))),
        ("deformable offsets negated", "squeeze_module.0.dec_att.aspp1.atrous_conv",
         lambda: set_attr(blocks.DeformableConv2d, "deform",
                          lambda self, x, o, m: real_deform(self, x, -o, m))),
        ("modulator 2*sigmoid -> sigmoid", "squeeze_module.0.dec_att.aspp1.atrous_conv",
         lambda: set_attr(blocks.DeformableConv2d, "deform",
                          lambda self, x, o, m: real_deform(self, x, o, m / 2))),
        ("alpha truncation -> rounding", "alpha",
         lambda: set_attr(vendor_image, "postprocess", rounded_postprocess)),
        ("refine radius 90 -> 91", "cutout refine",
         lambda: set_attr(vendor_image.refine_foreground, "__defaults__", (91,))),
    ]
    if precision == "fp32":
        # Below float16's resolution: a relative change of 1e-7 rounds away in a float16 product,
        # and BatchNorm runs in float16 under autocast, where eps 1e-4 vs 1e-5 is lost the same
        # way. The float16 run replaces them with the two float16-specific cases below.
        cases += [
            ("Swin attention scale x 1.0000001", "bb.layers.0.blocks.0.norm2",
             lambda: each(swin.WindowAttention, "scale", lambda s: s * 1.0000001)),
            ("BatchNorm eps 1e-5 -> 1e-4", "squeeze_module.0.dec_att.aspp1.atrous_conv",
             lambda: each(torch.nn.BatchNorm2d, "eps", lambda e: 1e-4)),
        ]
    if precision == "fp16":
        cases += [
            ("sigmoid in float32 (pre-GPU-gate postprocess)", "alpha",
             lambda: set_attr(vendor_image, "postprocess", float32_postprocess)),
            ("upstream's .half() instead of autocast", "bb.patch_embed.norm",
             half_model),
        ]
    if device == "cuda":
        # The production settings, flipped: each changes the kernels on the GPU only.
        cases += [
            ("input channels-last (cuDNN layout)", "bb.patch_embed.norm",
             lambda: set_attr(vendor_predictor, "preprocess", lambda *a, **k: real_preprocess(
                 *a, **k).contiguous(memory_format=torch.channels_last))),
        ]
        if precision == "fp32":
            # The patch embedding's 4x4 conv over 3 channels gets no TF32 kernel from cuDNN, and
            # Swin is all matmuls, so cuDNN's TF32 first shows in the first decoder convolution.
            cases += [
                ("cuDNN TF32 off (production: on)", "squeeze_module.0.dec_att.aspp1.atrous_conv",
                 lambda: flag(torch.backends.cudnn, "allow_tf32", False)),
                ("matmul TF32 on (production: off)", "bb.layers.0.blocks.0.norm2",
                 lambda: flag(torch.backends.cuda.matmul, "allow_tf32", True)),
            ]

    image_key = hashlib.sha256(rgb.tobytes()).hexdigest()[:12]
    path = cache and cache / (f"{variant}-{device}-{precision}-{image_key}-torch{torch.__version__}"
                              f"-tv{torchvision.__version__}.pt")
    if path and path.exists():
        up_seen, up = torch.load(path, weights_only=False)
        print(f"    reference read from {path}")
    else:
        up_seen, up = run_upstream(upstream, rgb, resolution, refine_fn, device, precision, names)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save((up_seen, up), path)
            print(f"    reference written to {path}")

    def observe() -> tuple[str, str]:
        pixels = vendor_image.preprocess(rgb, ours.resolution)
        if pixels.shape != up[0].shape or not torch.equal(pixels, up[0]):
            return "pixels", "shape" if pixels.shape != up[0].shape else \
                f"{(pixels - up[0]).abs().max().item():.1e}"
        our_seen, logits, alpha = run_ours(ours, rgb, names)
        # Upstream's call order, so "first" means first in the forward.
        for name in names:
            for k, (a, b) in enumerate(zip(our_seen.get(name, []), up_seen.get(name, []))):
                if not tensors_equal(a, b):
                    delta = (a.float() - b.float()).abs().max().item() if a.shape == b.shape else 0
                    return name + (" (2nd pass)" if k else ""), \
                        f"{delta:.1e}" + ("" if a.dtype == b.dtype else f" {a.dtype}/{b.dtype}")
        if not tensors_equal(logits, up[1]):
            return "logits", f"{(logits.float() - up[1].float()).abs().max().item():.1e}"
        if not np.array_equal(alpha, up[2]):
            return "alpha", f"{np.abs(alpha.astype(int) - up[2]).max()}"
        from mozo.vendors.birefnet_deploy.image import refine_foreground
        refined = np.dstack([refine_foreground(rgb, alpha), alpha])
        if not np.array_equal(refined, up[4]):
            return "cutout refine", f"{np.abs(refined.astype(int) - up[4]).max()}"
        return "nothing", "-"

    rows = []
    for label, expected, apply in cases:
        restore = apply()
        try:
            moved, delta = observe()
        finally:
            restore()
        rows.append((label, moved == expected, f"{moved} {delta}"))
    assert network.image2patches is real_patches and network._resize is real_resize
    assert blocks.DeformableConv2d.deform is real_deform and vendor_image.postprocess is real_postprocess
    assert vendor_predictor.preprocess is real_preprocess
    return rows


def ordered_names(ours, rgb) -> list[str]:
    """Stage modules in the order the forward first calls them."""
    wanted = set(stage_names(ours))
    order: list[str] = []
    handles = [m.register_forward_hook(lambda m, i, o, n=n: order.append(n) if n not in order else None)
               for n, m in ours.model.named_modules() if n in wanted]
    resident(ours.model)
    ours.logits(rgb[:64, :64])
    for h in handles:
        h.remove()
    return order


def environment(device: str) -> str:
    """torch, CUDA, cuDNN, GPU, driver and the global settings both sides run under."""
    lines = [f"torch {torch.__version__}  CUDA {torch.version.cuda}  "
             f"cuDNN {torch.backends.cudnn.version()}"]
    if device == "cuda":
        import subprocess

        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader",
                                 f"--id={torch.cuda.current_device()}"],
                                capture_output=True, text=True).stdout.strip() or "?"
        lines.append(f"GPU {torch.cuda.get_device_name()}  driver {driver}")
        lines.append(f"matmul allow_tf32 {torch.backends.cuda.matmul.allow_tf32}  "
                     f"cudnn allow_tf32 {torch.backends.cudnn.allow_tf32}  "
                     f"cudnn benchmark {torch.backends.cudnn.benchmark}  "
                     f"cudnn deterministic {torch.backends.cudnn.deterministic}  "
                     f"deterministic algorithms {torch.are_deterministic_algorithms_enabled()}  "
                     f"float32 matmul precision {torch.get_float32_matmul_precision()}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variant", choices=list(CHECKPOINTS))
    parser.add_argument("--weights", type=Path, required=True,
                        help="revision directory holding torch-fp32.pth, or the file itself")
    parser.add_argument("--images", type=Path, default=None,
                        help="directory of photographs; defaults to the shared fixtures")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=PRECISIONS, action="append", default=None,
                        help="precision(s) to gate; default fp32 on cpu, fp32 and fp16 on cuda")
    parser.add_argument("--repeat", type=int, default=None,
                        help="runs per side per image, all byte-identical; default 2 on cuda, 1 on cpu")
    parser.add_argument("--reference-cache", type=Path, default=None,
                        help="directory for the falsifiers' reference run, reused across runs")
    parser.add_argument("--swap-models", action="store_true",
                        help="keep only the running network on the GPU (always on for hr on cuda)")
    parser.add_argument("--no-falsify", action="store_true")
    args = parser.parse_args()
    device = args.device
    precisions = args.dtype or (["fp32"] if device == "cpu" else ["fp32", "fp16"])
    # ``half`` converts both models in place, so it runs last.
    precisions = sorted(dict.fromkeys(precisions), key=PRECISIONS.index)
    repeat = args.repeat or (2 if device == "cuda" else 1)

    import torchvision
    import transformers

    torch.manual_seed(0)
    print(f"variant {args.variant}: {CHECKPOINTS[args.variant].repo} @ "
          f"{CHECKPOINTS[args.variant].commit}")
    print(f"torch {torch.__version__}  torchvision {torchvision.__version__}  "
          f"transformers {transformers.__version__}  device {device}  "
          f"precisions {', '.join(precisions)}  repeat {repeat}  "
          f"MKL_CBWR={os.environ.get('MKL_CBWR', '(unset)')}")
    print(environment(device))

    problems = check_standalone()
    print(f"\n[1] standalone: {'ok' if not problems else 'FAILED'}")
    for p in problems:
        print(f"    {p}")

    rewrite = check_rewrite()
    print(f"[2] image2patches vs einops: {sum(ok for _, ok in rewrite)}/{len(rewrite)} exact")

    from safetensors.torch import load_file

    upstream, safetensors, handler = load_upstream(args.variant, device)
    refine_fn = handler_functions(handler)["refine_foreground"]
    checkpoint = args.weights / "torch-fp32.pth" if args.weights.is_dir() else args.weights
    published = torch.load(checkpoint, map_location="cpu", weights_only=True)
    source = load_file(safetensors)
    expected = {k: v for k, v in source.items() if not k.startswith(TRAINING_ONLY)}
    structural = (published.keys() == expected.keys()
                  and all(published[k].dtype == v.dtype and torch.equal(published[k], v)
                          for k, v in expected.items())
                  and len(source) - len(expected) == 12)
    base = Predictor.from_pretrained(args.variant, checkpoint, device=device)
    print(f"[3] structural: {len(published)} tensors = upstream's {len(source)} minus "
          f"{len(source) - len(expected)} training-only, dtypes kept, strict load: "
          f"{'ok' if structural else 'FAILED'}")
    del published, source, expected

    paths = fixtures() if args.images is None else sorted(
        p for p in args.images.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    inputs = [(p.name, load_image(str(p))) for p in paths] + synthetic()
    if device == "cuda" and (args.swap_models or args.variant == "hr"):
        RESIDENT.update(upstream=upstream, ours=base.model)
    names = ordered_names(base, inputs[0][1])
    # The falsifiers run on the shared fixture whatever --images says: an image whose matte is
    # nearly saturated can hide a perturbation of the last steps (a float32 sigmoid of float16
    # logits moves no pixel of COCO 000000000139 for `lite`, and 1.5M of the fixture's).
    shared = fixtures()[0]
    falsify_rgb = next((rgb for label, rgb in inputs if label == shared.name), None)
    if falsify_rgb is None:
        falsify_rgb = load_image(str(shared))
    good = not problems and all(ok for _, ok in rewrite) and structural

    for precision in precisions:
        if precision == "half":
            upstream.half()
            ours = Predictor(base.model, device=device, dtype=torch.float16)
        else:
            ours = Predictor(base.model, device=device, autocast=precision == "fp16")
        total = 0
        print(f"\n[4] every stage ({len(names)} stage modules), {device} {precision}:")
        for label, rgb in inputs:
            rows = compare(args.variant, upstream, ours, rgb, refine_fn, names, device, precision,
                           repeat)
            ok = all(r[1] for r in rows)
            good &= ok
            total += sum(r[3] for r in rows)
            detail = "; ".join(f"{s} FAILED {d}" for s, e, d, _ in rows if not e)
            print(f"    {'PASS' if ok else 'FAIL'}  {label:40s} {len(rows)} checks"
                  f"{'  ' + detail if detail else ''}")

        runtime = {"fp32": "torch-fp32", "fp16": "torch-fp16"}.get(precision)
        if runtime:
            try:
                from mozo.adapters.birefnet import BiRefNetPredictor
                for model in (RESIDENT and (upstream, base.model)) or ():
                    model.to("cpu")
                adapter = BiRefNetPredictor(args.variant, device=device, runtime=runtime)
            except Exception as error:  # weights not resolvable through mozo.weights here
                print(f"    skip  mozo adapter: {type(error).__name__}: {str(error)[:100]}")
            else:
                label, rgb = inputs[0]
                # One forward a side: the cut-out's fourth channel is the matte.
                theirs, used = adapter.cutout(rgb), adapter.runtime
                del adapter
                if device == "cuda":
                    torch.cuda.empty_cache()
                resident(ours.model)
                ok = np.array_equal(theirs, ours.cutout(rgb))
                good &= ok
                total += 1
                print(f"    {'PASS' if ok else 'FAIL'}  mozo adapter (registry -> weights, "
                      f"{used}), {label}")
        print(f"    {total} comparisons, {device} {precision}")

        if not args.no_falsify:
            print(f"[5] falsification, {device} {precision} -- each must fail, at its own stage")
            rows = falsify(args.variant, upstream, ours, falsify_rgb, refine_fn, names, device,
                           precision, args.reference_cache)
            for label, ok, moved in rows:
                print(f"    {'caught' if ok else 'MISSED'}  {label:44s} {moved}")
            good &= all(ok for _, ok, _ in rows)
            print(f"    {sum(ok for _, ok, _ in rows)}/{len(rows)} caught")

    print(f"\n{'PASS' if good else 'FAIL'}")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Verify the vendored BiRefNet against the authors' Hugging Face remote code.

    python tools/verify/birefnet.py general --weights weights/birefnet/general/<rev>
    python tools/verify/birefnet.py lite --weights ... --images /path/to/photos
    python tools/verify/birefnet.py general --weights ... --no-falsify

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

**The reference is pinned to CPU, float32 and eager attention.** ``from_pretrained`` is given
``dtype=torch.float32``: transformers 5 defaults to ``dtype="auto"``, which loads the
float16-stored checkpoints (``general``, ``hr``) as a float16 model. The upstream module's own
``SDPA_enabled`` is ``False``. Parameters are cloned after loading so both sides compute from
freshly allocated tensors (``VENDORING.md`` §4). Run with ``MKL_CBWR=AVX2,STRICT`` when the two
sides run in different processes; here they share one.
"""

from __future__ import annotations

import argparse
import ast
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

VENDOR = ROOT / "mozo" / "vendors" / "birefnet_deploy"
ALLOWED_THIRD_PARTY = {"torch", "torchvision", "numpy", "cv2", "PIL"}

#: Modules whose every output is compared, by type. Leaf convolutions inside these are covered
#: by the block that contains them; the deformable conv is named because it is the one operator
#: an exporter replaces.
STAGES = (swin.PatchEmbed, swin.SwinTransformerBlock, swin.PatchMerging, torch.nn.LayerNorm,
          blocks.DeformableConv2d, blocks.ASPPDeformable, blocks.BasicDecBlk, blocks.BasicLatBlk,
          blocks.SimpleConvs, torch.nn.Sequential)


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

def load_upstream(variant: str):
    """Upstream's model at the pinned commit, float32, parameters re-allocated; and its files."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForImageSegmentation

    repo, commit = CHECKPOINTS[variant].repo, CHECKPOINTS[variant].commit
    model = AutoModelForImageSegmentation.from_pretrained(
        repo, revision=commit, trust_remote_code=True, dtype=torch.float32).eval()
    with torch.no_grad():
        for tensor in list(model.parameters()) + list(model.buffers()):
            tensor.data = tensor.data.clone()
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


def upstream_pipeline(model, rgb: np.ndarray, resolution: int, refine_fn):
    """The model card's snippet and ``handler.py``'s refine, run as upstream runs them."""
    from torchvision import transforms

    image = Image.fromarray(rgb)
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    pixels = transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(pixels)[-1]
    pred_pil = transforms.ToPILImage()(logits.sigmoid().cpu()[0].squeeze())
    mask = pred_pil.resize(image.size)
    plain = image.copy()
    plain.putalpha(mask)
    refined = refine_fn(image, pred_pil)
    refined.putalpha(pred_pil.resize(image.size))
    return pixels, logits, np.asarray(mask), np.asarray(plain), np.asarray(refined)


# --------------------------------------------------------------------------------------------
# 4

def record(model, names: set[str]):
    """Hook every module in *names*; return (outputs by name in call order, remove)."""
    seen: dict[str, list] = {}
    handles = []
    for name, module in model.named_modules():
        if name in names:
            handles.append(module.register_forward_hook(
                lambda m, i, o, name=name: seen.setdefault(name, []).append(
                    o[0] if isinstance(o, tuple) else o)))
    return seen, lambda: [h.remove() for h in handles]


def stage_names(ours) -> list[str]:
    return [name for name, module in ours.model.named_modules()
            if name and isinstance(module, STAGES)]


def compare(variant, upstream, ours, rgb, refine_fn, names) -> list[tuple[str, bool, str]]:
    """Every stage for one image: (stage, equal, detail)."""
    resolution = MODEL_SPECS[variant].resolution
    up_seen, remove = record(upstream, set(names))
    up_pixels, up_logits, up_mask, up_plain, up_refined = upstream_pipeline(
        upstream, rgb, resolution, refine_fn)
    remove()

    rows = []
    pixels = vendor_image.preprocess(rgb, resolution)
    rows.append(("pixels", torch.equal(pixels, up_pixels), ""))

    our_seen, remove = record(ours.model, set(names))
    with torch.inference_mode():
        logits = ours.model(pixels)
    remove()
    calls = mismatched = 0
    first = ""
    for name in names:
        for a, b in zip(our_seen.get(name, []), up_seen.get(name, [])):
            calls += 1
            if a.shape != b.shape or not torch.equal(a, b):
                mismatched += 1
                first = first or name
        if len(our_seen.get(name, [])) != len(up_seen.get(name, [])):
            mismatched += 1
            first = first or f"{name} (call count)"
    rows.append((f"{calls} module outputs", mismatched == 0,
                 f"first: {first}" if mismatched else ""))
    del our_seen, up_seen

    rows.append(("logits", torch.equal(logits, up_logits), ""))
    alpha = vendor_image.postprocess(logits, rgb.shape[:2])
    rows.append(("alpha", np.array_equal(alpha, up_mask), ""))
    rows.append(("cutout", np.array_equal(ours.cutout(rgb), up_plain), ""))
    rows.append(("cutout refine", np.array_equal(ours.cutout(rgb, refine=True), up_refined), ""))
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

def falsify(variant, upstream, ours, rgb, refine_fn, names) -> list[tuple[str, bool, str]]:
    """Each perturbation must make the first failing stage the one it reaches."""
    import torch.nn.functional as F

    def set_attr(obj, name, value):
        original = getattr(obj, name)
        setattr(obj, name, value)
        return lambda: setattr(obj, name, original)

    def each(kind, attr, change):
        touched = [(m, getattr(m, attr)) for m in ours.model.modules() if isinstance(m, kind)]
        for m, v in touched:
            setattr(m, attr, change(v))
        return lambda: [setattr(m, attr, v) for m, v in touched]

    real_patches, real_resize = network.image2patches, network._resize
    real_deform = blocks.DeformableConv2d.deform
    real_postprocess = vendor_image.postprocess
    resolution = MODEL_SPECS[variant].resolution

    def rounded_postprocess(logits, size):
        small = (logits.float().sigmoid()[0].squeeze() * 255).round().byte().numpy()
        return np.asarray(Image.fromarray(small, mode="L").resize(size[::-1], Image.BICUBIC))

    cases = (
        ("normalisation mean 0.485 -> 0.486", "pixels",
         lambda: set_attr(vendor_image, "MEAN", (0.486, 0.456, 0.406))),
        ("resize BILINEAR -> BICUBIC", "pixels",
         lambda: set_attr(Image, "BILINEAR", Image.BICUBIC)),
        (f"resolution {resolution} -> {resolution + 32}", "pixels",
         lambda: set_attr(ours, "resolution", resolution + 32)),
        ("Swin attention scale x 1.0000001", "bb.layers.0.blocks.0.norm2",
         lambda: each(swin.WindowAttention, "scale", lambda s: s * 1.0000001)),
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
        ("BatchNorm eps 1e-5 -> 1e-4", "squeeze_module.0.dec_att.aspp1.atrous_conv",
         lambda: each(torch.nn.BatchNorm2d, "eps", lambda e: 1e-4)),
        ("alpha truncation -> rounding", "alpha",
         lambda: set_attr(vendor_image, "postprocess", rounded_postprocess)),
        ("refine radius 90 -> 91", "cutout refine",
         lambda: set_attr(vendor_image.refine_foreground, "__defaults__", (91,))),
    )

    # The reference is perturbation-free, so it is computed once.
    up_seen, remove = record(upstream, set(names))
    up = upstream_pipeline(upstream, rgb, resolution, refine_fn)
    remove()

    def observe() -> tuple[str, str]:
        pixels = vendor_image.preprocess(rgb, ours.resolution)
        if pixels.shape != up[0].shape or not torch.equal(pixels, up[0]):
            return "pixels", "shape" if pixels.shape != up[0].shape else \
                f"{(pixels - up[0]).abs().max().item():.1e}"
        our_seen, remove = record(ours.model, set(names))
        with torch.inference_mode():
            logits = ours.model(pixels)
        remove()
        # Upstream's call order, so "first" means first in the forward.
        for name in names:
            for k, (a, b) in enumerate(zip(our_seen.get(name, []), up_seen.get(name, []))):
                if not torch.equal(a, b):
                    return name + (" (2nd pass)" if k else ""), f"{(a - b).abs().max().item():.1e}"
        alpha = vendor_image.postprocess(logits, rgb.shape[:2])
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
    return rows


def ordered_names(ours, rgb) -> list[str]:
    """Stage modules in the order the forward first calls them."""
    wanted = set(stage_names(ours))
    order: list[str] = []
    handles = [m.register_forward_hook(lambda m, i, o, n=n: order.append(n) if n not in order else None)
               for n, m in ours.model.named_modules() if n in wanted]
    with torch.inference_mode():
        ours.model(vendor_image.preprocess(rgb[:64, :64], ours.resolution))
    for h in handles:
        h.remove()
    return order


# --------------------------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variant", choices=list(CHECKPOINTS))
    parser.add_argument("--weights", type=Path, required=True,
                        help="revision directory holding torch-fp32.pth, or the file itself")
    parser.add_argument("--images", type=Path, default=None,
                        help="directory of photographs; defaults to the shared fixtures")
    parser.add_argument("--no-falsify", action="store_true")
    args = parser.parse_args()

    import torchvision
    import transformers

    torch.manual_seed(0)
    print(f"variant {args.variant}: {CHECKPOINTS[args.variant].repo} @ "
          f"{CHECKPOINTS[args.variant].commit}")
    print(f"torch {torch.__version__}  torchvision {torchvision.__version__}  "
          f"transformers {transformers.__version__}  device cpu, float32  "
          f"MKL_CBWR={os.environ.get('MKL_CBWR', '(unset)')}")

    problems = check_standalone()
    print(f"\n[1] standalone: {'ok' if not problems else 'FAILED'}")
    for p in problems:
        print(f"    {p}")

    rewrite = check_rewrite()
    print(f"[2] image2patches vs einops: {sum(ok for _, ok in rewrite)}/{len(rewrite)} exact")

    from safetensors.torch import load_file

    upstream, safetensors, handler = load_upstream(args.variant)
    refine_fn = handler_functions(handler)["refine_foreground"]
    checkpoint = args.weights / "torch-fp32.pth" if args.weights.is_dir() else args.weights
    published = torch.load(checkpoint, map_location="cpu", weights_only=True)
    source = load_file(safetensors)
    expected = {k: v for k, v in source.items() if not k.startswith(TRAINING_ONLY)}
    structural = (published.keys() == expected.keys()
                  and all(published[k].dtype == v.dtype and torch.equal(published[k], v)
                          for k, v in expected.items())
                  and len(source) - len(expected) == 12)
    ours = Predictor.from_pretrained(args.variant, checkpoint, device="cpu")
    print(f"[3] structural: {len(published)} tensors = upstream's {len(source)} minus "
          f"{len(source) - len(expected)} training-only, dtypes kept, strict load: "
          f"{'ok' if structural else 'FAILED'}")

    paths = fixtures() if args.images is None else sorted(
        p for p in args.images.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    inputs = [(p.name, load_image(str(p))) for p in paths] + synthetic()
    names = ordered_names(ours, inputs[0][1])

    print(f"[4] every stage ({len(names)} stage modules):")
    good = not problems and all(ok for _, ok in rewrite) and structural
    for label, rgb in inputs:
        rows = compare(args.variant, upstream, ours, rgb, refine_fn, names)
        ok = all(r[1] for r in rows)
        good &= ok
        detail = "; ".join(f"{s} FAILED {d}" for s, e, d in rows if not e)
        print(f"    {'PASS' if ok else 'FAIL'}  {label:40s} {len(rows)} stages"
              f"{'  ' + detail if detail else ''}")

    try:
        from mozo.adapters.birefnet import BiRefNetPredictor
        adapter = BiRefNetPredictor(args.variant, device="cpu", runtime="torch-fp32")
    except Exception as error:  # weights not resolvable through mozo.weights here
        print(f"    skip  mozo adapter: {type(error).__name__}: {str(error)[:100]}")
    else:
        label, rgb = inputs[0]
        # One forward a side: the cut-out's fourth channel is the matte.
        ok = np.array_equal(adapter.cutout(rgb), ours.cutout(rgb))
        good &= ok
        print(f"    {'PASS' if ok else 'FAIL'}  mozo adapter (registry -> weights), {label}")

    if not args.no_falsify:
        print("[5] falsification -- each must fail, at its own stage")
        rows = falsify(args.variant, upstream, ours, inputs[0][1], refine_fn, names)
        for label, ok, moved in rows:
            print(f"    {'caught' if ok else 'MISSED'}  {label:36s} {moved}")
        good &= all(ok for _, ok, _ in rows)

    print(f"\n{'PASS' if good else 'FAIL'}")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Verify the vendored EoMT against the authors' code, stage by stage, exactly.

    MKL_CBWR=AVX2,STRICT python -I tools/verify/eomt.py --upstream /path/to/eomt \\
        --authors /path/to/bin-dir [--mozo] [--witness] [--coco /path/to/val2017 --count 20] \\
        [variants...]

The reference is ``tue-mps/eomt`` at commit ``aa017c3`` (MIT) on timm 1.0.15: its ``EoMT`` network
built as the authors' ``inference.ipynb`` builds it (``models.vit.ViT`` + ``models.eomt.EoMT``,
``masked_attn_enabled=False``) inside the Lightning module of the checkpoint's task, filled from the
**authors' own** ``pytorch_model.bin`` (``--authors``, downloaded and sha256-checked there if
absent), and driven the way ``eval_step`` and the notebook drive it, last layer only:
``resize_and_pad_imgs_instance_panoptic`` -> network -> ``F.interpolate`` ->
``revert_resize_and_pad_logits_instance_panoptic`` -> ``to_per_pixel_preds_panoptic`` for panoptic;
``window_imgs_semantic`` -> network -> ``F.interpolate`` -> ``to_per_pixel_logits_semantic`` ->
``revert_window_logits_semantic`` -> argmax for semantic. Compared on the CPU, ``torch.equal``:

1. **checkpoint** -- mozo's published file against the authors' (every key, every tensor; the
   loss's ``criterion.empty_weight`` is the one key not carried).
2. **pixels**, **patch_embed**, the input of every block (**block.N**, queries included from where
   they enter), the final norm's input (**final**), **class_logits**, **mask_logits**.
3. **segments** -- the post-processed partition: the panoptic segment-id map and each segment's
   class, or the semantic class map. mozo's memory-bounded post-processing (kept queries only;
   class maps upsampled in slices) against upstream's all-queries, all-classes path.
4. **score** (``--witness``) -- each panoptic segment's score against ``transformers``' slow
   pipeline (``use_fast=False``), which reproduces upstream's partition and reports the score
   upstream computes but does not return; rounded to the 6 places it reports.
5. **mozo** (``--mozo``) -- ``EomtPredictor`` through the registry and the weights tree: its
   ``Detections`` (masks, classes, confidences) against the vendor's segments, on arrays and file
   paths.

**Set ``MKL_CBWR=AVX2,STRICT``.** MKL picks kernels by buffer alignment, so two CPU torch processes
can differ in the last bit without it (VENDORING §5).

``--falsify NAME`` perturbs one thing in the vendor and expects the gate to fail first at the stage
named; ``--falsify list`` names them, ``--falsify all`` runs every one (each in a fresh process)
and fails unless each was caught where expected.
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.stdout.reconfigure(line_buffering=True)
# This checks a tree you just built; reaching for the published bucket would verify other bytes.
os.environ.setdefault("MOZO_BASE_URL", f"file://{ROOT / 'weights'}")
os.environ.setdefault("WANDB_MODE", "disabled")  # upstream's Lightning module imports wandb

import warnings  # noqa: E402

warnings.filterwarnings("ignore", message=".*is an instance of `nn.Module`.*")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from common import download_verified, hub_lfs_sha256s  # noqa: E402
from common import fixtures as photographs  # noqa: E402
from fetch.eomt import COMMIT, COMMITS  # noqa: E402

from mozo.vendors.eomt_deploy import SPECS, Segmenter  # noqa: E402
from mozo.vendors.eomt_deploy import image as vimage  # noqa: E402
from mozo.vendors.eomt_deploy import postprocess as vpost  # noqa: E402

#: The versions parity was established against: upstream's own requirements.txt. Printed.
PINNED = {"torch": "2.7.0", "torchvision": "0.22.0", "timm": "1.0.15"}

#: Perturbations of the vendor: name -> (what, the stage expected to fail first, default variant).
FALSIFIERS = {
    "bicubic": ("PIL bicubic instead of bilinear", "pixels", "coco-panoptic-small"),
    "floor": ("int() instead of round() for the resized size", "pixels", "coco-panoptic-small"),
    "pad_topleft": ("padding on the top and left", "pixels", "coco-panoptic-small"),
    "registers_first": ("register tokens before the class token", "block.0", "coco-panoptic-small"),
    "ln_eps": ("block 1 norm2 eps 1e-6 -> 1e-5", "block.2", "coco-panoptic-small"),
    "layerscale": ("block 3 ls2 skipped", "block.4", "coco-panoptic-small"),
    "eager": ("manual softmax attention instead of SDPA", "block.1", "coco-panoptic-small"),
    "queries_early": ("queries enter one block early", "block.8", "coco-panoptic-small"),
    "mask_head_tanh": ("mask head GELU -> tanh approximation", "mask_logits", "coco-panoptic-small"),
    "threshold": ("panoptic mask_thresh 0.8 -> 0.7", "segments", "coco-panoptic-small"),
    "overlap": ("panoptic overlap_thresh 0.8 -> 0.5", "segments", "coco-panoptic-small"),
    "no_count": ("window overlap summed, not averaged", "segments", "ade20k-semantic-large"),
    "argmax_early": ("argmax before the upsample to the picture", "segments",
                     "ade20k-semantic-large"),
}


class Gate:
    """Exact comparisons, with the first failing stage remembered."""

    def __init__(self) -> None:
        self.count = 0
        self.failures: list[tuple[str, float]] = []

    @property
    def failed(self) -> bool:
        return bool(self.failures)

    def check(self, stage: str, ours, theirs, label: str = "") -> bool:
        self.count += 1
        ours, theirs = torch.as_tensor(ours), torch.as_tensor(theirs)
        if ours.shape == theirs.shape and torch.equal(ours, theirs):
            return True
        delta = (float((ours.double() - theirs.double()).abs().max())
                 if ours.shape == theirs.shape and ours.numel() else float("inf"))
        self.failures.append((stage, delta))
        print(f"    FAIL {stage:14s} {label:28s} max|delta| = {delta:.3e}"
              + ("" if ours.shape == theirs.shape else f" shapes {tuple(ours.shape)} vs {tuple(theirs.shape)}"))
        return False


def load_upstream(checkout: Path):
    """Import the reference from *checkout* and say which commit it is."""
    try:
        head = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        head = "unknown (not a git checkout)"
    if head != COMMIT:
        print(f"WARNING: upstream checkout is at {head}, the reference commit is {COMMIT}")
    sys.path.insert(0, str(checkout))
    import models.eomt  # noqa: F401
    import models.vit  # noqa: F401
    import training.mask_classification_panoptic  # noqa: F401
    import training.mask_classification_semantic  # noqa: F401
    sys.path.remove(str(checkout))
    return sys.modules


def authors_checkpoint(variant: str, directory: Path) -> Path:
    """The authors' ``pytorch_model.bin`` for *variant*, downloaded into *directory* if absent."""
    spec = SPECS[variant]
    commit = COMMITS[variant]
    target = directory / spec.repository.split("/")[1] / "pytorch_model.bin"
    sha256 = hub_lfs_sha256s(spec.repository, commit)["pytorch_model.bin"]
    download_verified(f"https://huggingface.co/{spec.repository}/resolve/{commit}/pytorch_model.bin",
                      target, sha256, label=variant, width=24, timeout=900)
    return target


def reference(variant: str, modules, original: Path):
    """Upstream's Lightning module for *variant*, filled from the authors' file, as the notebook
    loads it (``load_state_dict(strict=False)``; nothing may be missing or unexpected)."""
    spec = SPECS[variant]
    size = (spec.image_size, spec.image_size)
    backbone = {384: "vit_small", 768: "vit_base", 1024: "vit_large"}[spec.width]
    # ``ckpt_path`` set so timm does not download DINOv2's own weights: every one is overwritten.
    encoder = modules["models.vit"].ViT(img_size=size, backbone_name=f"{backbone}_patch14_reg4_dinov2",
                                        ckpt_path="authors-checkpoint")
    network = modules["models.eomt"].EoMT(encoder=encoder, num_classes=spec.classes,
                                          num_q=spec.queries, num_blocks=spec.query_blocks,
                                          masked_attn_enabled=False)
    if spec.task == "panoptic":
        lit = modules["training.mask_classification_panoptic"].MaskClassificationPanoptic(
            network=network, img_size=size, num_classes=spec.classes,
            stuff_classes=list(spec.stuff), attn_mask_annealing_enabled=False)
    else:
        lit = modules["training.mask_classification_semantic"].MaskClassificationSemantic(
            network=network, img_size=size, num_classes=spec.classes,
            attn_mask_annealing_enabled=False)
    state = torch.load(original, map_location="cpu", weights_only=True)
    incompatible = lit.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise SystemExit(f"reference load: {incompatible}")
    return lit.eval(), state


def upstream_answer(lit, spec, rgb: np.ndarray):
    """Upstream's pixels, last-layer logits and post-processed map for one image."""
    timg = torch.from_numpy(rgb.copy()).permute(2, 0, 1)
    size = tuple(timg.shape[-2:])
    with torch.inference_mode():
        if spec.task == "semantic":
            pixels, origins = lit.window_imgs_semantic([timg])
        else:
            pixels = lit.resize_and_pad_imgs_instance_panoptic([timg])
        mask_layers, class_layers = lit(pixels)
        mask_logits, class_logits = mask_layers[-1], class_layers[-1]
        up = F.interpolate(mask_logits, lit.img_size, mode="bilinear")
        if spec.task == "semantic":
            crop = lit.to_per_pixel_logits_semantic(up, class_logits)
            answer = lit.revert_window_logits_semantic(crop, origins, [size])[0].argmax(0)
        else:
            up = lit.revert_resize_and_pad_logits_instance_panoptic(up, [size])
            answer = lit.to_per_pixel_preds_panoptic(up, class_logits, lit.stuff_classes,
                                                     lit.mask_thresh, lit.overlap_thresh)[0]
    return pixels, mask_logits, class_logits, answer


def fixtures(coco: Path | None, count: int, seed: int = 0) -> list[tuple[str, np.ndarray]]:
    """RGB images: the repo's photographs, awkward synthetic cases, and random COCO val."""
    frames = [(p.name, np.asarray(Image.open(p).convert("RGB"))) for p in photographs()]
    example = Image.fromarray(frames[0][1])
    frames += [
        ("portrait", np.ascontiguousarray(np.asarray(example).transpose(1, 0, 2))),
        ("square-640", np.asarray(example)[:640, 200:840].copy()),     # no pad, no resize
        ("odd-1000x701", np.asarray(example.resize((1000, 701)))),      # round != floor
        ("tiny-37x50", np.asarray(example.resize((37, 50)))),           # upsampled
        ("wide-2000x120", np.asarray(example.resize((2000, 120)))),     # many windows
        ("black-320x240", np.zeros((240, 320, 3), np.uint8)),
        ("grey", np.asarray(example.convert("L").convert("RGB"))),
    ]
    if coco is not None and count:
        names = sorted(glob.glob(str(coco / "*.jpg")))
        for path in random.Random(seed).sample(names, count):
            frames.append((Path(path).name, np.asarray(Image.open(path).convert("RGB"))))
    return frames


def pre_hooks(model: torch.nn.Module, names: dict[str, str]) -> tuple[dict, list]:
    """Record each named module's first input (the last call), keyed by stage."""
    seen: dict[str, torch.Tensor] = {}
    handles = []
    modules = dict(model.named_modules())
    for stage, name in names.items():
        def hook(_module, inputs, stage=stage):
            seen[stage] = inputs[0].detach().clone()
        handles.append(modules[name].register_forward_pre_hook(hook))
    return seen, handles


def stage_names(spec, prefix: str) -> dict[str, str]:
    """stage -> the module whose input is that stage: each block's first norm sees the block's
    input (queries included from where they enter), the backbone's final norm the last block's
    output."""
    names = {f"block.{i}": f"{prefix}encoder.backbone.blocks.{i}.norm1" for i in range(spec.depth)}
    names["final"] = f"{prefix}encoder.backbone.norm"
    return names


def compare_variant(variant: str, modules, authors: Path, frames, gate: Gate,
                    falsifier: str | None, witness: bool) -> Segmenter:
    """Every stage for one variant. Returns the vendor's segmenter, for the mozo path."""
    from mozo.weights import resolve

    spec = SPECS[variant]
    original = authors_checkpoint(variant, authors)
    published = resolve("eomt", variant, "torch-fp32")

    segmenter = Segmenter(published, spec, device="cpu")
    network = segmenter.network()
    lit, theirs_state = reference(variant, modules, original)

    # --- the checkpoint: the published file, as the vendor loaded it, against the authors' ---
    theirs = {k.removeprefix("network."): v for k, v in theirs_state.items()
              if k.startswith("network.")}
    ours_state = network.state_dict()
    extra = set(theirs_state) - {f"network.{k}" for k in theirs} - {"criterion.empty_weight"}
    if set(theirs) != set(ours_state) or extra:
        gate.failures.append(("checkpoint", float("inf")))
        print(f"    FAIL checkpoint keys differ: {sorted(set(theirs) ^ set(ours_state) | extra)[:5]}")
    for key in sorted(theirs.keys() & ours_state.keys()):
        gate.check("checkpoint", ours_state[key], theirs[key], key)
    del ours_state, theirs, theirs_state

    if falsifier:
        perturb(falsifier, network)

    ours_seen, ours_hooks = pre_hooks(network, stage_names(spec, ""))
    theirs_seen, theirs_hooks = pre_hooks(lit, stage_names(spec, "network."))
    # The patch embedding's output is block 0's input before the position embedding and tokens.
    for model, seen, hooks in ((network, ours_seen, ours_hooks),
                               (lit.network, theirs_seen, theirs_hooks)):
        hooks.append(model.encoder.backbone.patch_embed.register_forward_hook(
            lambda _m, _i, out, seen=seen: seen.__setitem__("patch_embed", out.detach().clone())))

    hf = _witness(variant) if witness and spec.task == "panoptic" else None
    for index, (label, rgb) in enumerate(frames):
        size = tuple(rgb.shape[:2])
        their_pixels, their_masks, their_classes, answer = upstream_answer(lit, spec, rgb)
        with torch.inference_mode():
            pixels, mask_logits, class_logits, origins = segmenter.raw(rgb)
        gate.check("pixels", pixels, their_pixels, label)
        for stage in ["patch_embed", *stage_names(spec, "")]:
            gate.check(stage, ours_seen[stage], theirs_seen[stage], label)
        gate.check("class_logits", class_logits, their_classes, label)
        gate.check("mask_logits", mask_logits, their_masks, label)

        # --- post-processing, the vendor's memory-bounded path from the vendor's own logits ---
        with torch.inference_mode():
            if spec.task == "semantic":
                labels = vpost.semantic(mask_logits, class_logits, origins, size, spec.image_size)
                gate.check("segments", labels, answer, label)
            else:
                segments, found = vpost.panoptic(mask_logits[0], class_logits[0], size,
                                                 spec.image_size, spec.stuff)
                gate.check("segments", segments, answer[..., 1], label)
                classes = torch.full(size, spec.classes, dtype=torch.long)
                for segment_id, class_id, _ in found:
                    classes[segments == segment_id] = class_id
                gate.check("segments", classes, answer[..., 0], label + " classes")
                if hf is not None and index < 3:
                    gate.check("score", torch.tensor([round(s, 6) for _, _, s in found]),
                               torch.tensor(hf(rgb)), label)

    for handle in ours_hooks + theirs_hooks:
        handle.remove()
    return segmenter


def _witness(variant: str):
    """``transformers``' slow pipeline for *variant*: rgb -> its segments' scores, in id order."""
    from huggingface_hub import snapshot_download
    from transformers import AutoImageProcessor, EomtForUniversalSegmentation

    spec = SPECS[variant]
    path = snapshot_download(spec.repository, revision=COMMITS[variant],
                             allow_patterns=["*.json", "model.safetensors"])
    model = EomtForUniversalSegmentation.from_pretrained(path, attn_implementation="sdpa").eval()
    processor = AutoImageProcessor.from_pretrained(path, use_fast=False)

    def scores(rgb: np.ndarray) -> list[float]:
        with torch.inference_mode():
            inputs = processor(images=Image.fromarray(rgb), return_tensors="pt")
            out = model(pixel_values=inputs["pixel_values"])
            info = processor.post_process_panoptic_segmentation(
                out, target_sizes=[rgb.shape[:2]])[0]["segments_info"]
        return [s["score"] for s in sorted(info, key=lambda s: s["id"])]
    return scores


def perturb(name: str, network) -> None:
    """Apply one falsifier to the vendor (and only the vendor)."""
    backbone = network.encoder.backbone
    if name == "bicubic":
        def resize(image, height, width):
            out = Image.fromarray(image).resize((width, height), Image.BICUBIC)
            return torch.from_numpy(np.array(out)).permute(2, 0, 1)
        vimage._resize = resize
    elif name == "floor":
        def pad_size(size, side):
            factor = min(side / size[0], side / size[1])
            return [int(s * factor) for s in size]
        vimage.pad_size = pad_size
    elif name == "pad_topleft":
        import mozo.vendors.eomt_deploy.predictor as vpred

        def resize_and_pad(image, side):  # the binding the segmenter calls
            new_h, new_w = vimage.pad_size(image.shape[:2], side)
            resized = vimage._resize(image, new_h, new_w)
            return F.pad(resized, (side - new_w, 0, side - new_h, 0))[None]
        vpred.resize_and_pad = resize_and_pad
    elif name == "registers_first":
        def _pos_embed(x):
            x = x + backbone.pos_embed
            return torch.cat([backbone.reg_token.expand(x.shape[0], -1, -1),
                              backbone.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        backbone._pos_embed = _pos_embed
    elif name == "ln_eps":
        backbone.blocks[1].norm2.eps = 1e-5
    elif name == "layerscale":
        backbone.blocks[3].ls2 = torch.nn.Identity()
    elif name == "eager":
        def eager(self, x):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            attn = ((q * self.head_dim ** -0.5) @ k.transpose(-2, -1)).softmax(dim=-1)
            return self.proj((attn @ v).transpose(1, 2).reshape(B, N, C))
        for block in backbone.blocks:
            block.attn.forward = eager.__get__(block.attn)
    elif name == "queries_early":
        network.num_blocks += 1
    elif name == "mask_head_tanh":
        network.mask_head[1] = torch.nn.GELU(approximate="tanh")
    elif name in ("threshold", "overlap"):
        original_panoptic = vpost.panoptic
        setting = {"threshold": 0.7} if name == "threshold" else {"overlap": 0.5}
        vpost.panoptic = lambda *a, **k: original_panoptic(*a, **{**k, **setting})
    elif name in ("no_count", "argmax_early"):
        def semantic(mask_logits, class_logits, origins, size, side):
            mask_logits = F.interpolate(mask_logits, (side, side), mode="bilinear")
            crop = torch.einsum("bqhw, bqc -> bchw", mask_logits.sigmoid(),
                                class_logits.softmax(dim=-1)[..., :-1])
            h, w = vimage.window_size(size, side)
            sums, counts = torch.zeros((crop.shape[1], h, w)), torch.zeros((1, h, w))
            for c, (start, end) in zip(crop, origins):
                window = (slice(None), slice(start, end)) if size[0] > size[1] else \
                    (slice(None), slice(None), slice(start, end))
                sums[window] += c
                counts[window] += 1
            if name == "no_count":  # the overlap summed, not averaged
                return F.interpolate(sums[None], size, mode="bilinear")[0].argmax(0)
            labels = (sums / counts).argmax(0)[None, None].float()  # argmax, then upsample
            return F.interpolate(labels, size, mode="nearest")[0, 0].long()
        vpost.semantic = semantic
    else:
        raise SystemExit(f"unknown falsifier {name!r}; known: {sorted(FALSIFIERS)}")


def against_mozo(variant: str, vendor: Segmenter, frames, paths: list[Path], gate: Gate) -> None:
    """mozo's adapter, through the registry and the weights tree, against the vendor's segments
    (which the stages above hold to upstream)."""
    import mozo
    from mozo.image import load_image
    from pixelflow.validators import round_to_decimal

    model = mozo.get_model(f"eomt/{variant}", device="cpu")
    if model.runtime != "torch-fp32":
        print(f"    note: mozo chose {model.runtime}; comparing that")
    inputs = [(label, rgb, rgb) for label, rgb in frames]
    inputs += [(p.name + " (path)", str(p), load_image(str(p))) for p in paths]
    for label, given, rgb in inputs:
        want = vendor.segment(rgb)
        got = model.predict(given)
        masks = (np.stack([d.masks[0] for d in got]) if len(got)
                 else np.zeros((0, *rgb.shape[:2]), bool))
        gate.check("mozo", torch.from_numpy(masks), torch.from_numpy(want.masks), label)
        gate.check("mozo", torch.tensor([d.class_id for d in got], dtype=torch.long),
                   torch.from_numpy(want.class_ids), label + " classes")
        # nan stands for None (a semantic segment's confidence) on both sides.
        expected = ([float("nan")] * len(want.class_ids) if want.scores is None
                    else [round_to_decimal(float(s)) for s in want.scores])
        confidences = [float("nan") if d.confidence is None else d.confidence for d in got]
        gate.check("mozo", torch.tensor(confidences, dtype=torch.float64).nan_to_num(-1.0),
                   torch.tensor(expected, dtype=torch.float64).nan_to_num(-1.0),
                   label + " confidences")


def versions() -> str:
    import timm
    import torchvision

    return (f"torch {torch.__version__}, torchvision {torchvision.__version__}, "
            f"timm {timm.__version__} (pinned {PINNED})")


def falsify_all(argv: list[str]) -> int:
    """Run every falsifier in a process of its own and report which were caught where."""
    base = [a for i, a in enumerate(argv)
            if not a.startswith("--falsify") and not (i and argv[i - 1] == "--falsify")]
    missed = []
    for name in FALSIFIERS:
        run = subprocess.run([sys.executable, "-I", __file__, *base, "--falsify", name],
                             capture_output=True, text=True)
        verdict = [line for line in run.stdout.splitlines() if line.startswith("falsifier")]
        print(verdict[-1] if verdict else f"falsifier {name!r}: no verdict\n{run.stderr[-2000:]}")
        if not verdict or "CAUGHT at the expected stage" not in verdict[-1]:
            missed.append(name)
    print(f"\n{len(FALSIFIERS) - len(missed)} of {len(FALSIFIERS)} falsifiers caught where expected"
          + (f"; missed: {missed}" if missed else ""))
    return 1 if missed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variants", nargs="*", help=f"default: all of {list(SPECS)}")
    parser.add_argument("--upstream", type=Path, required=True,
                        help=f"a tue-mps/eomt checkout at {COMMIT[:7]}")
    parser.add_argument("--authors", type=Path, required=True,
                        help="directory holding the authors' pytorch_model.bin files (filled if empty)")
    parser.add_argument("--mozo", action="store_true", help="also compare mozo's adapter")
    parser.add_argument("--witness", action="store_true",
                        help="also check panoptic scores against transformers' slow pipeline")
    parser.add_argument("--coco", type=Path, default=None, help="a directory of .jpg to sample")
    parser.add_argument("--count", type=int, default=10, help="how many --coco images")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--falsify", default=None, help="a falsifier name, 'list' or 'all'")
    args = parser.parse_args()

    if args.falsify == "list":
        for name, (what, stage, variant) in FALSIFIERS.items():
            print(f"{name:16s} {what}  [expected at: {stage}; on {variant}]")
        return 0
    if args.falsify == "all":
        return falsify_all(sys.argv[1:])

    if os.environ.get("MKL_CBWR") != "AVX2,STRICT":
        print("WARNING: MKL_CBWR is not AVX2,STRICT; CPU results may differ in the last bit")
    torch.set_num_threads(args.threads)
    variants = args.variants or ([FALSIFIERS[args.falsify][2]] if args.falsify else list(SPECS))
    print(f"EoMT gate: {versions()}, upstream {COMMIT[:7]}, CPU, {args.threads} threads")
    modules = load_upstream(args.upstream)
    frames = fixtures(args.coco, args.count)
    print(f"{len(frames)} images: {', '.join(label for label, _ in frames[:12])}"
          + (", ..." if len(frames) > 12 else ""))

    gate = Gate()
    for variant in variants:
        before = (gate.count, len(gate.failures))
        print(f"\n[{variant}]")
        vendor = compare_variant(variant, modules, args.authors, frames, gate,
                                    falsifier=args.falsify, witness=args.witness)
        if args.mozo and not args.falsify:
            against_mozo(variant, vendor, frames, photographs(), gate)
        count, failed = gate.count - before[0], len(gate.failures) - before[1]
        print(f"  {variant}: {count - failed} of {count} comparisons exact")

    if args.falsify:
        what, expected, _ = FALSIFIERS[args.falsify]
        first = gate.failures[0][0] if gate.failures else None
        where = ("CAUGHT at the expected stage" if first == expected
                 else f"CAUGHT, but first at {first!r} (expected {expected!r})" if first
                 else "NOT CAUGHT")
        stages = sorted({stage for stage, _ in gate.failures})
        print(f"falsifier {args.falsify!r} ({what}): {where}; "
              f"{len(gate.failures)} of {gate.count} comparisons failed, stages {stages}")
        return 0 if first == expected else 1

    print(f"\n{gate.count - len(gate.failures)} of {gate.count} comparisons exact"
          + ("" if not gate.failed else f"; {len(gate.failures)} FAILED"))
    return 1 if gate.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

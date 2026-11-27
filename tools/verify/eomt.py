#!/usr/bin/env python3
"""Verify the vendored EoMT against the authors' code, stage by stage, exactly.

    # CPU, float32 (the secondary gate; needs no GPU):
    MKL_CBWR=AVX2,STRICT python -I tools/verify/eomt.py --upstream /path/to/eomt \\
        --authors /path/to/bin-dir [--mozo] [--witness] [--coco /path/to/val2017 --count 20] \\
        [variants...]
    # CUDA, float32 and fp16 autocast (the primary gate):
    python -I tools/verify/eomt.py --device cuda --upstream ... --authors ... --mozo \\
        --coco ... --count 20 [--dtype fp32] [--dtype fp16] [--reference-cache DIR]

The reference is ``tue-mps/eomt`` at commit ``aa017c3`` (MIT) on timm 1.0.15: its ``EoMT`` network
built as the authors' ``inference.ipynb`` builds it (``models.vit.ViT`` + ``models.eomt.EoMT``,
``masked_attn_enabled=False``) inside the Lightning module of the checkpoint's task, filled from the
**authors' own** ``pytorch_model.bin`` (``--authors``, downloaded and sha256-checked there if
absent), and driven the way the notebook's ``infer_panoptic`` / ``infer_semantic`` drive it, last
layer only, under ``torch.no_grad()``: ``resize_and_pad_imgs_instance_panoptic`` -> network ->
``F.interpolate`` -> ``revert_resize_and_pad_logits_instance_panoptic`` ->
``to_per_pixel_preds_panoptic`` for panoptic; ``window_imgs_semantic`` -> network ->
``F.interpolate`` -> ``to_per_pixel_logits_semantic`` -> ``revert_window_logits_semantic`` -> argmax
for semantic. Compared with ``torch.equal`` (dtype included):

1. **checkpoint** -- mozo's published file against the authors' (every key, every tensor; the
   loss's ``criterion.empty_weight`` is the one key not carried). **requires_grad** -- every
   parameter's flag against upstream's (on CUDA it can change which matmul kernel runs).
2. **pixels**, **layout** (the strides of the patch embedding's input: cuDNN picks kernels by
   memory layout), **patch_embed**, the input of every block (**block.N**, queries included from
   where they enter), the final norm's input (**final**), **class_logits**, **mask_logits**.
3. **segments** -- the post-processed partition: the panoptic segment-id map and each segment's
   class, or the semantic class map. mozo's memory-bounded post-processing (kept queries only;
   class maps upsampled in slices) against upstream's all-queries, all-classes path.
4. **score** (``--witness``, CPU only) -- each panoptic segment's score against ``transformers``'
   slow pipeline (``use_fast=False``), which reproduces upstream's partition and reports the score
   upstream computes but does not return; rounded to the 6 places it reports.
5. **mozo** (``--mozo``) -- ``EomtPredictor`` through the registry and the weights tree, with the
   runtime of the precision under test: the arrays its segmenter hands to
   ``pf.detections.from_arrays`` (masks, classes, float32 scores, exactly) and the ``Detections``
   that come out, against the vendor's segments, on arrays and file paths.
6. **determinism** (``--repeat``, 2 on CUDA) -- each side run again on every picture, every stage
   byte-identical to its first run (``VENDORING.md`` §2.7).

**Device and precision.** ``--device cpu`` (the default) gates float32 on the CPU. ``--device
cuda`` gates float32 and fp16 by default (``--dtype`` picks): ``fp16`` is CUDA float16 autocast,
which ``torch-fp16`` -- ``auto`` on CUDA -- runs. Upstream's own half path is the same mechanism
over the same span: the notebook wraps network *and* post-processing in
``autocast(dtype=torch.float16)``, and its Lightning ``16-mixed`` evaluation does the same, so the
reference runs that way and so does mozo. Both sides run in this one process, on one GPU, under
torch's default global settings, which mozo's runtime leaves alone (printed on every run).

**Set ``MKL_CBWR=AVX2,STRICT``** for CPU runs. MKL picks kernels by buffer alignment, so two CPU
torch processes can differ in the last bit without it (VENDORING §5).

``--falsify NAME`` perturbs one thing in the vendor (or one global setting, after the reference has
run) and expects the gate to fail first at the stage named; ``--falsify list`` names them,
``--falsify all`` runs every one that applies to the device and precisions (each in a fresh
process) and fails unless each was caught where expected. ``--reference-cache DIR`` keeps the
reference's per-stage digests there, so the falsifiers run upstream once per precision and fixture
set.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import random
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import NamedTuple

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

PRECISIONS = ("fp32", "fp16")

ALL, CUDA = ("cpu", "cuda"), ("cuda",)

#: Perturbations: name -> (what, the stage expected to fail first, default variant, the precisions
#: and the devices it applies to).
FALSIFIERS = {
    "bicubic": ("PIL bicubic instead of bilinear", "pixels", "coco-panoptic-small", PRECISIONS, ALL),
    "floor": ("int() instead of round() for the resized size", "pixels", "coco-panoptic-small",
              PRECISIONS, ALL),
    "pad_topleft": ("padding on the top and left", "pixels", "coco-panoptic-small", PRECISIONS, ALL),
    "registers_first": ("register tokens before the class token", "block.0", "coco-panoptic-small",
                        PRECISIONS, ALL),
    "ln_eps": ("block 1 norm2 eps 1e-6 -> 1e-5", "block.2", "coco-panoptic-small", PRECISIONS, ALL),
    "layerscale": ("block 3 ls2 skipped", "block.4", "coco-panoptic-small", PRECISIONS, ALL),
    "eager": ("manual softmax attention instead of SDPA", "block.1", "coco-panoptic-small",
              PRECISIONS, ALL),
    "queries_early": ("queries enter one block early", "block.8", "coco-panoptic-small",
                      PRECISIONS, ALL),
    "mask_head_tanh": ("mask head GELU -> tanh approximation", "mask_logits", "coco-panoptic-small",
                       PRECISIONS, ALL),
    "threshold": ("panoptic mask_thresh 0.8 -> 0.7", "segments", "coco-panoptic-small",
                  PRECISIONS, ALL),
    "overlap": ("panoptic overlap_thresh 0.8 -> 0.5", "segments", "coco-panoptic-small",
                PRECISIONS, ALL),
    "no_count": ("window overlap summed, not averaged", "segments", "ade20k-semantic-large",
                 PRECISIONS, ALL),
    "argmax_early": ("argmax before the upsample to the picture", "segments",
                     "ade20k-semantic-large", PRECISIONS, ALL),
    "requires_grad_off": ("parameters frozen (upstream's all require grad)", "requires_grad",
                          "coco-panoptic-small", PRECISIONS, ALL),
    "inference_build": ("network loaded inside inference mode (mozo's adapter before the GPU gate)",
                        "requires_grad", "ade20k-semantic-large", PRECISIONS, ALL),
    # GPU only: the production settings, flipped. Each changes kernels on the GPU only.
    "channels_last": ("network input channels-last (cuDNN layout)", "layout", "coco-panoptic-small",
                      PRECISIONS, CUDA),
    "sdpa_math": ("SDPA pinned to the math backend (production: torch's pick)", "block.1",
                  "coco-panoptic-small", PRECISIONS, CUDA),
    "tf32_matmul_on": ("matmul TF32 on (production: off)", "block.1", "coco-panoptic-small",
                       ("fp32",), CUDA),
    "cudnn_tf32_off": ("cuDNN TF32 off (production: on)", "mask_logits", "coco-panoptic-small",
                       ("fp32",), CUDA),
    # Semantic only: on panoptic the float16 upsample never flipped a ``>= 0.5`` or the argmax
    # over score x mask on the 28 pictures (0 of 758 comparisons failed), so it cannot be caught.
    "fp32_postprocess": ("post-processing on float32 logits outside autocast (mozo before the GPU "
                         "gate)", "segments", "ade20k-semantic-large", ("fp16",), CUDA),
}

#: Falsifiers that flip a process-wide setting: applied only after the reference has run.
GLOBAL = {"tf32_matmul_on", "cudnn_tf32_off"}


class Digest(NamedTuple):
    """A tensor's shape, dtype and sha256 of its bytes: what the reference cache keeps."""
    shape: tuple
    dtype: str
    sha256: str


def digest(tensor: torch.Tensor) -> Digest:
    t = tensor.detach().cpu().contiguous()
    return Digest(tuple(t.shape), str(t.dtype), hashlib.sha256(t.numpy().tobytes()).hexdigest())


class Gate:
    """Exact comparisons, with the first failing stage remembered."""

    def __init__(self) -> None:
        self.count = 0
        self.failures: list[tuple[str, float]] = []
        self.stages: dict[str, int] = {}

    @property
    def failed(self) -> bool:
        return bool(self.failures)

    def check(self, stage: str, ours, theirs, label: str = "") -> bool:
        self.count += 1
        self.stages[stage] = self.stages.get(stage, 0) + 1
        ours = torch.as_tensor(ours)
        if isinstance(theirs, Digest):
            if digest(ours) == theirs:
                return True
            self.failures.append((stage, float("nan")))
            print(f"    FAIL {stage:14s} {label:28s} digest differs from the cached reference")
            return False
        theirs = torch.as_tensor(theirs)
        if theirs.device != ours.device:
            ours, theirs = ours.cpu(), theirs.cpu()
        same = ours.shape == theirs.shape and ours.dtype == theirs.dtype
        if same and torch.equal(ours, theirs):
            return True
        delta = (float((ours.double() - theirs.double()).abs().max())
                 if ours.shape == theirs.shape and ours.numel() else float("inf"))
        self.failures.append((stage, delta))
        print(f"    FAIL {stage:14s} {label:28s} max|delta| = {delta:.3e}"
              + ("" if ours.shape == theirs.shape else f" shapes {tuple(ours.shape)} vs {tuple(theirs.shape)}")
              + ("" if ours.dtype == theirs.dtype else f" dtypes {ours.dtype} vs {theirs.dtype}"))
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
    loads it (``load_state_dict(strict=False)``; nothing may be missing or unexpected). On the CPU;
    the caller moves it."""
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


def autocast(device: str, precision: str):
    """fp16 autocast on CUDA, as the notebook spells it; nothing for fp32."""
    return (torch.autocast(device_type="cuda", dtype=torch.float16) if precision == "fp16"
            else nullcontext())


def upstream_answer(lit, spec, rgb: np.ndarray, device: str, precision: str) -> dict:
    """Upstream's pixels, last-layer logits and post-processed map for one image, as the
    notebook's ``infer_panoptic`` / ``infer_semantic`` compute them."""
    timg = torch.from_numpy(rgb.copy()).permute(2, 0, 1)
    with torch.no_grad(), autocast(device, precision):
        imgs = [timg.to(device)]
        size = tuple(imgs[0].shape[-2:])
        if spec.task == "semantic":
            pixels, origins = lit.window_imgs_semantic(imgs)
        else:
            pixels = lit.resize_and_pad_imgs_instance_panoptic(imgs)
        mask_layers, class_layers = lit(pixels)
        mask_logits, class_logits = mask_layers[-1], class_layers[-1]
        up = F.interpolate(mask_logits, lit.img_size, mode="bilinear")
        if spec.task == "semantic":
            crop = lit.to_per_pixel_logits_semantic(up, class_logits)
            del up  # read once; 1.8 GB for a 17-window picture, which upstream's revert then needs
            answer = lit.revert_window_logits_semantic(crop, origins, [size])[0].argmax(0)
            out = {"segments": answer}
        else:
            up = lit.revert_resize_and_pad_logits_instance_panoptic(up, [size])
            answer = lit.to_per_pixel_preds_panoptic(up, class_logits, lit.stuff_classes,
                                                     lit.mask_thresh, lit.overlap_thresh)[0]
            out = {"segments": answer[..., 1], "segment classes": answer[..., 0]}
    return {"pixels": pixels, "class_logits": class_logits, "mask_logits": mask_logits, **out}


def vendor_answer(segmenter: Segmenter, spec, rgb: np.ndarray) -> tuple[dict, list]:
    """The vendor's pixels, logits and partition for one image, in the order and under the
    precision :meth:`Segmenter.segment` runs them, and the panoptic ``found`` list."""
    size = tuple(rgb.shape[:2])
    with torch.inference_mode(), segmenter.precision():
        pixels, mask_logits, class_logits, origins = segmenter.raw(rgb)
        out = {"pixels": pixels, "class_logits": class_logits, "mask_logits": mask_logits}
        if spec.task == "semantic":
            out["segments"] = vpost.semantic(mask_logits, class_logits, origins, size,
                                             spec.image_size)
            return out, []
        segments, found = vpost.panoptic(mask_logits[0], class_logits[0], size, spec.image_size,
                                         spec.stuff)
        classes = torch.full(size, spec.classes, dtype=torch.long, device=segments.device)
        for segment_id, class_id, _ in found:
            classes[segments == segment_id] = class_id
    return {**out, "segments": segments, "segment classes": classes}, found


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


def record_stages(model: torch.nn.Module, spec, prefix: str) -> tuple[dict, list]:
    """Record, on every forward of *model*: each named stage module's first input (the last
    call), the patch embedding's output, and the strides of the patch embedding's input. Kept on
    the host (an exact copy): a 17-window semantic picture's stages are ~2 GB a run."""
    seen: dict[str, torch.Tensor] = {}
    modules = dict(model.named_modules())
    handles = []
    for stage, name in stage_names(spec, prefix).items():
        def hook(_module, inputs, stage=stage):
            seen[stage] = inputs[0].detach().to("cpu", copy=True)
        handles.append(modules[name].register_forward_pre_hook(hook))
    embed = modules[f"{prefix}encoder.backbone.patch_embed"]
    handles.append(embed.register_forward_hook(
        lambda _m, _i, out: seen.__setitem__("patch_embed", out.detach().to("cpu", copy=True))))
    handles.append(embed.proj.register_forward_pre_hook(
        lambda _m, inputs: seen.__setitem__("layout", torch.tensor(inputs[0].stride()))))
    return seen, handles


def stage_names(spec, prefix: str) -> dict[str, str]:
    """stage -> the module whose input is that stage: each block's first norm sees the block's
    input (queries included from where they enter), the backbone's final norm the last block's
    output."""
    names = {f"block.{i}": f"{prefix}encoder.backbone.blocks.{i}.norm1" for i in range(spec.depth)}
    names["final"] = f"{prefix}encoder.backbone.norm"
    return names


def stage_order(spec) -> list[str]:
    """Every per-picture stage, in the order the pipeline produces them."""
    return ["pixels", "layout", "patch_embed", *stage_names(spec, ""), "class_logits",
            "mask_logits", "segments", "segment classes"]


def requires_grad(module: torch.nn.Module) -> tuple[list[str], torch.Tensor]:
    """Each parameter's name, and whether it requires grad and is an inference tensor."""
    named = sorted(module.named_parameters())
    return [n for n, _ in named], torch.tensor([(p.requires_grad, p.is_inference())
                                                for _, p in named])


def free(device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()


def cache_path(cache: Path | None, variant, device, precision, label, rgb) -> Path | None:
    if cache is None:
        return None
    import timm

    key = hashlib.sha256(rgb.tobytes()).hexdigest()[:12]
    return cache / (f"{variant}-{device}-{precision}-{label}-{key}-torch{torch.__version__}"
                    f"-timm{timm.__version__}-{COMMIT[:7]}.pt")


#: Variants whose two sides do not fit on a 12 GB card together: upstream's semantic revert keeps
#: 150 full-size class maps twice (sums and counts), ~8 GB for the 2000x120 picture's 17 windows.
SWAP = {"ade20k-semantic-large"}


class Reference:
    """Upstream for one variant: the module (on the CPU until a picture needs it on *device*),
    each picture's stages (tensors, or digests from *cache*). With *swap*, *other* (the vendor's
    network) waits on the host while upstream runs, and upstream waits there otherwise."""

    def __init__(self, variant, modules, original, device, cache, swap=False):
        self.variant, self.spec, self.device, self.cache = variant, SPECS[variant], device, cache
        self.lit, self.state = reference(variant, modules, original)
        self._on_device = False
        self.swap, self.other = swap, None
        self.seen, self.hooks = record_stages(self.lit, self.spec, "network.")

    def module(self):
        if not self._on_device:
            self.lit.to(self.device)
            self._on_device = True
        return self.lit

    def run(self, rgb, precision) -> dict:
        if self.swap and self.other is not None:
            self.other.to("cpu")
            free(self.device)
        try:
            out = upstream_answer(self.module(), self.spec, rgb, self.device, precision)
        finally:
            if self.swap:
                self.lit.to("cpu")
                self._on_device = False
                free(self.device)
                if self.other is not None:
                    self.other.to(self.device)
        # On the host (exact copies): upstream's semantic revert of the 17-window picture comes
        # within ~1 GB of a 12 GB card, so nothing of an earlier run may stay on the GPU.
        out = {k: v.to("cpu", copy=True) for k, v in out.items()}
        free(self.device)
        return {**{k: self.seen[k] for k in ("layout", "patch_embed", *stage_names(self.spec, ""))},
                **out}

    def digests(self, label, rgb, precision) -> dict:
        """The picture's stages as digests, from the cache if there, else run and cached."""
        path = cache_path(self.cache, self.variant, self.device, precision, label, rgb)
        if path is not None and path.exists():
            return torch.load(path, weights_only=False)
        stages = {k: digest(v) for k, v in self.run(rgb, precision).items()}
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(stages, path)
        return stages

    def close(self):
        for handle in self.hooks:
            handle.remove()
        del self.lit
        free(self.device)


def compare_variant(variant: str, modules, authors: Path, frames, gate: Gate, device: str,
                    precisions: list[str], falsifier: str | None, witness: bool, repeat: int,
                    cache: Path | None, use_mozo: bool, swap: bool = False) -> None:
    """Every stage for one variant, in each precision."""
    from mozo.weights import resolve

    spec = SPECS[variant]
    original = authors_checkpoint(variant, authors)
    published = resolve("eomt", variant, "torch-fp32")
    swap = device == "cuda" and (swap or variant in SWAP)
    ref = Reference(variant, modules, original, device, cache, swap)

    segmenter = Segmenter(published, spec, device=device)
    if falsifier == "inference_build":  # mozo before the GPU gate: built inside inference mode
        from mozo.vendors.eomt_deploy.checkpoint import load_network

        with torch.inference_mode():
            segmenter._network = load_network(spec, published, device)
    network = segmenter.network()
    ref.other = network
    if swap:
        print("  models swapped: only the running side's network is on the GPU")

    # --- the checkpoint: the published file, as the vendor loaded it, against the authors' ---
    theirs = {k.removeprefix("network."): v for k, v in ref.state.items()
              if k.startswith("network.")}
    ours_state = network.state_dict()
    extra = set(ref.state) - {f"network.{k}" for k in theirs} - {"criterion.empty_weight"}
    if set(theirs) != set(ours_state) or extra:
        gate.failures.append(("checkpoint", float("inf")))
        print(f"    FAIL checkpoint keys differ: {sorted(set(theirs) ^ set(ours_state) | extra)[:5]}")
    for key in sorted(theirs.keys() & ours_state.keys()):
        gate.check("checkpoint", ours_state[key].cpu(), theirs[key], key)
    del ours_state, theirs
    ref.state = None

    if falsifier and falsifier not in GLOBAL:
        perturb(falsifier, network)

    # --- which parameters require grad: on CUDA it picks the matmul path (VENDORING §5) ---
    ours_names, ours_grad = requires_grad(network)
    theirs_names, theirs_grad = requires_grad(ref.lit.network)
    if ours_names != theirs_names:
        gate.failures.append(("requires_grad", float("inf")))
        print("    FAIL requires_grad parameter names differ")
    else:
        gate.check("requires_grad", ours_grad, theirs_grad,
                   f"{int(ours_grad[:, 0].sum())} of {len(ours_grad)} require grad")

    ours_seen, ours_hooks = record_stages(network, spec, "")
    hf = _witness(variant) if witness and spec.task == "panoptic" and device == "cpu" else None
    for precision in precisions:
        print(f"  [{precision}]")
        vendor = Segmenter(None, spec, device=device, network=network, half=precision == "fp16")
        if falsifier:
            # The reference first, all of it: a global falsifier must not reach it.
            wanted = [ref.digests(label, rgb, precision) for label, rgb in frames]
            if falsifier in GLOBAL:
                perturb(falsifier, network)
        for index, (label, rgb) in enumerate(frames):
            if falsifier:
                theirs = wanted[index]
            else:
                theirs = ref.run(rgb, precision)
                for _ in range(repeat - 1):  # upstream run again: byte-identical?
                    again = ref.run(rgb, precision)
                    for stage in stage_order(spec):
                        if stage in theirs:
                            gate.check("determinism", again[stage], theirs[stage],
                                       f"{label} upstream {stage}")
                    del again
            ours = None
            for run in range(max(1, repeat if not falsifier else 1)):
                out, found = vendor_answer(vendor, spec, rgb)
                current = {**{k: ours_seen[k] for k in
                              ("layout", "patch_embed", *stage_names(spec, ""))}, **out}
                if ours is None:
                    ours, ours_found = current, found
                else:  # the vendor run again: byte-identical?
                    for stage in stage_order(spec):
                        if stage in ours:
                            gate.check("determinism", current[stage], ours[stage],
                                       f"{label} mozo {stage}")
            for stage in stage_order(spec):
                if stage in theirs:
                    gate.check(stage, ours[stage], theirs[stage], label)
            if hf is not None and index < 3:
                gate.check("score", torch.tensor([round(s, 6) for _, _, s in ours_found]),
                           torch.tensor(hf(rgb)), label)
            del theirs, ours, current
            free(device)
        if falsifier in GLOBAL:
            break  # one precision per falsifier process
        if use_mozo and not falsifier:
            against_mozo(variant, vendor, frames, photographs(), gate, device, precision)

    for handle in ours_hooks:
        handle.remove()
    ref.close()
    del network, segmenter
    free(device)


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
    """Apply one falsifier to the vendor (and only the vendor), or flip one global setting."""
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
            return torch.stack([F.pad(resized, (side - new_w, 0, side - new_h, 0))])
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
        backbone.blocks[3].ls2.forward = lambda x: x  # the parameter stays, for requires_grad
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
            sums = torch.zeros((crop.shape[1], h, w), device=crop.device)
            counts = torch.zeros((1, h, w), device=crop.device)
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
    elif name == "inference_build":
        pass  # applied where the network is built (compare_variant)
    elif name == "requires_grad_off":
        for parameter in network.parameters():
            parameter.requires_grad_(False)
    elif name == "channels_last":
        forward = network.forward
        network.forward = lambda x: forward(x.contiguous(memory_format=torch.channels_last))
    elif name == "sdpa_math":
        from torch.nn.attention import SDPBackend, sdpa_kernel

        forward = network.forward

        def math_forward(x):
            with sdpa_kernel(SDPBackend.MATH):
                return forward(x)
        network.forward = math_forward
    elif name == "tf32_matmul_on":
        torch.backends.cuda.matmul.allow_tf32 = True
    elif name == "cudnn_tf32_off":
        torch.backends.cudnn.allow_tf32 = False
    elif name == "fp32_postprocess":
        def outside_autocast(function):
            def run(mask_logits, class_logits, *args, **kwargs):
                with torch.autocast("cuda", enabled=False):
                    return function(mask_logits.float(), class_logits.float(), *args, **kwargs)
            return run
        vpost.panoptic = outside_autocast(vpost.panoptic)
        vpost.semantic = outside_autocast(vpost.semantic)
    else:
        raise SystemExit(f"unknown falsifier {name!r}; known: {sorted(FALSIFIERS)}")


def against_mozo(variant: str, vendor: Segmenter, frames, paths: list[Path], gate: Gate,
                 device: str, precision: str) -> None:
    """mozo's adapter, through the registry and the weights tree, against the vendor's segments
    (which the stages above hold to upstream): the arrays it hands to ``from_arrays``, exactly,
    and the ``Detections`` it returns."""
    from mozo.adapters.eomt import EomtPredictor
    from mozo.image import load_image
    from pixelflow.validators import round_to_decimal

    runtime = {"fp32": "torch-fp32", "fp16": "torch-fp16"}[precision]
    if device == "cuda" and precision == "fp16":
        chosen = EomtPredictor(variant, device=device).runtime
        if chosen != runtime:
            gate.failures.append(("mozo", float("inf")))
            print(f"    FAIL mozo auto on CUDA chose {chosen}, expected {runtime}")
    model = EomtPredictor(variant, device=device, runtime=runtime)
    inputs = [(label, rgb, rgb) for label, rgb in frames]
    inputs += [(p.name + " (path)", str(p), load_image(str(p))) for p in paths]
    for label, given, rgb in inputs:
        want = vendor.segment(rgb)
        handed = model._segmenter.segment(rgb)  # what predict() hands to from_arrays
        if label == inputs[0][0]:  # built on that first call, as in production: built how?
            gate.check("mozo", requires_grad(model._segmenter.network())[1],
                       requires_grad(vendor.network())[1], "requires_grad / inference tensors")
        with torch.inference_mode():
            ours, theirs = model._segmenter.raw(rgb), vendor.raw(rgb)
        gate.check("mozo", ours[1], theirs[1], label + " mask logits")
        gate.check("mozo", ours[2], theirs[2], label + " class logits")
        del ours, theirs
        gate.check("mozo", torch.from_numpy(handed.masks), torch.from_numpy(want.masks),
                   label + " handed masks")
        gate.check("mozo", torch.from_numpy(handed.class_ids), torch.from_numpy(want.class_ids),
                   label + " handed classes")
        if want.scores is not None or handed.scores is not None:
            gate.check("mozo", torch.from_numpy(handed.scores), torch.from_numpy(want.scores),
                       label + " handed scores")
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
        free(device)
    del model
    free(device)


def versions() -> str:
    import timm
    import torchvision

    return (f"torch {torch.__version__}, torchvision {torchvision.__version__}, "
            f"timm {timm.__version__} (pinned {PINNED})")


def environment(device: str) -> str:
    """torch, CUDA, cuDNN, GPU, driver and the global settings both sides run under."""
    lines = [f"torch {torch.__version__}  CUDA {torch.version.cuda}  "
             f"cuDNN {torch.backends.cudnn.version()}"]
    if device == "cuda":
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


def applies(name: str, device: str, precision: str) -> bool:
    _, _, _, precisions, devices = FALSIFIERS[name]
    return device in devices and precision in precisions


def falsify_all(argv: list[str], device: str, precisions: list[str]) -> int:
    """Run every falsifier that applies, one precision at a time, each in a process of its own,
    and report which were caught where."""
    base, skip = [], False
    for a in argv:
        if skip:
            skip = False
        elif a in ("--falsify", "--dtype"):
            skip = True
        elif not (a.startswith("--falsify=") or a.startswith("--dtype=")):
            base.append(a)
    runs = [(name, p) for p in precisions for name in FALSIFIERS if applies(name, device, p)]
    missed = []
    for name, precision in runs:
        run = subprocess.run([sys.executable, "-I", __file__, *base, "--falsify", name,
                              "--dtype", precision], capture_output=True, text=True)
        verdict = [line for line in run.stdout.splitlines() if line.startswith("falsifier")]
        print(verdict[-1] if verdict else f"falsifier {name!r} [{precision}]: no verdict\n"
              f"{run.stdout[-1500:]}\n{run.stderr[-2000:]}")
        if not verdict or "CAUGHT at the expected stage" not in verdict[-1]:
            missed.append(f"{name} [{precision}]")
    print(f"\n{len(runs) - len(missed)} of {len(runs)} falsifiers caught where expected "
          f"({device}, {'/'.join(precisions)})" + (f"; missed: {missed}" if missed else ""))
    return 1 if missed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variants", nargs="*", help=f"default: all of {list(SPECS)}")
    parser.add_argument("--upstream", type=Path, required=True,
                        help=f"a tue-mps/eomt checkout at {COMMIT[:7]}")
    parser.add_argument("--authors", type=Path, required=True,
                        help="directory holding the authors' pytorch_model.bin files (filled if empty)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=PRECISIONS, action="append", default=None,
                        help="precision(s) to gate; default fp32 on cpu, fp32 and fp16 on cuda")
    parser.add_argument("--repeat", type=int, default=None,
                        help="runs per side per picture, all byte-identical; default 2 on cuda, "
                             "1 on cpu")
    parser.add_argument("--reference-cache", type=Path, default=None,
                        help="directory for the falsifiers' reference digests, reused across runs")
    parser.add_argument("--swap-models", action="store_true",
                        help="keep only the running side's network on the GPU (always for "
                             f"{sorted(SWAP)} on cuda)")
    parser.add_argument("--mozo", action="store_true", help="also compare mozo's adapter")
    parser.add_argument("--witness", action="store_true",
                        help="also check panoptic scores against transformers' slow pipeline (CPU)")
    parser.add_argument("--coco", type=Path, default=None, help="a directory of .jpg to sample")
    parser.add_argument("--count", type=int, default=10, help="how many --coco images")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--falsify", default=None, help="a falsifier name, 'list' or 'all'")
    args = parser.parse_args()
    device = args.device
    precisions = sorted(dict.fromkeys(args.dtype or (["fp32"] if device == "cpu"
                                                     else ["fp32", "fp16"])),
                        key=PRECISIONS.index)
    if device == "cpu" and "fp16" in precisions:
        raise SystemExit("fp16 is CUDA autocast; there is no CPU half path to gate")
    repeat = args.repeat or (2 if device == "cuda" else 1)

    if args.falsify == "list":
        for name, (what, stage, variant, ps, devices) in FALSIFIERS.items():
            print(f"{name:26s} {what}  [expected at: {stage}; on {variant}; "
                  f"{'/'.join(ps)}; {'/'.join(devices)}]")
        return 0
    if args.falsify == "all":
        return falsify_all(sys.argv[1:], device, precisions)
    if args.falsify and not all(applies(args.falsify, device, p) for p in precisions):
        raise SystemExit(f"falsifier {args.falsify!r} does not apply to {device} {precisions}")

    if device == "cpu" and os.environ.get("MKL_CBWR") != "AVX2,STRICT":
        print("WARNING: MKL_CBWR is not AVX2,STRICT; CPU results may differ in the last bit")
    torch.set_num_threads(args.threads)
    variants = args.variants or ([FALSIFIERS[args.falsify][2]] if args.falsify else list(SPECS))
    print(f"EoMT gate: {versions()}, upstream {COMMIT[:7]}, {device} {'/'.join(precisions)}, "
          f"{args.threads} threads, {repeat} run(s) per side")
    print(environment(device))
    modules = load_upstream(args.upstream)
    frames = fixtures(args.coco, args.count)
    print(f"{len(frames)} images: {', '.join(label for label, _ in frames[:12])}"
          + (", ..." if len(frames) > 12 else ""))

    gate = Gate()
    for variant in variants:
        before = (gate.count, len(gate.failures))
        print(f"\n[{variant}]")
        compare_variant(variant, modules, args.authors, frames, gate, device, precisions,
                        falsifier=args.falsify, witness=args.witness, repeat=repeat,
                        cache=args.reference_cache, use_mozo=args.mozo, swap=args.swap_models)
        count, failed = gate.count - before[0], len(gate.failures) - before[1]
        print(f"  {variant}: {count - failed} of {count} comparisons exact")

    if args.falsify:
        what = FALSIFIERS[args.falsify][0]
        expected = FALSIFIERS[args.falsify][1]
        first = gate.failures[0][0] if gate.failures else None
        where = ("CAUGHT at the expected stage" if first == expected
                 else f"CAUGHT, but first at {first!r} (expected {expected!r})" if first
                 else "NOT CAUGHT")
        stages = sorted({stage for stage, _ in gate.failures})
        print(f"falsifier {args.falsify!r} [{device} {'/'.join(precisions)}] ({what}): {where}; "
              f"{len(gate.failures)} of {gate.count} comparisons failed, stages {stages}")
        return 0 if first == expected else 1

    print("comparisons per stage: " + ", ".join(f"{k} {v}" for k, v in gate.stages.items()))
    print(f"\n{gate.count - len(gate.failures)} of {gate.count} comparisons exact"
          + ("" if not gate.failed else f"; {len(gate.failures)} FAILED")
          + f" ({device}, {'/'.join(precisions)})")
    return 1 if gate.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

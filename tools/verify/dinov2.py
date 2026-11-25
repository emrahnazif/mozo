#!/usr/bin/env python3
"""Verify the vendored DINOv2 against the authors' torch-hub code, stage by stage, exactly.

    MKL_CBWR=AVX2,STRICT python -I tools/verify/dinov2.py --upstream /path/to/dinov2 \\
        --authors /path/to/pth-dir [--mozo] [--coco /path/to/val2017 --count 20] [variants...]

The reference is ``facebookresearch/dinov2`` at commit ``7764ea0`` (Apache-2.0), the code
``torch.hub.load('facebookresearch/dinov2', ...)`` serves: its hub constructors with a strict load
of the **authors' own** ``*_pretrain.pth`` (``--authors``, downloaded there if absent), and its
evaluation transform ``make_classification_eval_transform``. Three paths are compared on the CPU,
with ``torch.equal`` and no tolerance:

1. **upstream** -- the above, on a PIL image, with ``F.normalize(model(x), dim=1, p=2)`` as its own
   k-NN evaluation applies (``ModelWithNormalize``), one image per forward.
2. **vendor** -- :class:`mozo.vendors.dinov2_deploy.Encoder` on mozo's published artifact.
3. **mozo** -- ``Dinov2Predictor`` through the registry and the weights tree (``--mozo``), on file
   paths decoded by ``mozo.image.load_image`` and on arrays, against upstream on the same RGB.

Stages: the published checkpoint against the authors' file (every key, every tensor); the pixels;
the tokens after the position embedding (and registers); every block's output; the final norm;
the class token; the normalised vector. The backbone alone is also compared at 518x518 (no
interpolation) and 224x336 (non-square) on random input, so both interpolation branches and the
early return are covered for every variant.

The reference runs with ``XFORMERS_DISABLED=1``, set before it is imported: upstream switches to
xFormers' kernels whenever ``xformers`` is importable, and the vendored package never does.

**Set ``MKL_CBWR=AVX2,STRICT``.** MKL picks kernels by buffer alignment, so two CPU torch
processes can differ in the last bit without it (VENDORING §5).

Batching is not among the falsifiers: under ``MKL_CBWR=AVX2,STRICT`` one forward over nine images
is bit-identical to nine single forwards on the CPU, so the gate cannot see it. Without that
setting -- as users run -- it is not (1.5e-05 on base, 1.8e-05 on large), which is why the encoder
runs one image at a time anyway.

``--falsify NAME`` perturbs one thing in the vendor and expects the gate to fail at the stage
named; ``--falsify list`` names them, ``--falsify all`` runs every one (each in a fresh process)
and fails unless each was caught where expected.
"""

from __future__ import annotations

import argparse
import functools
import glob
import os
import random
import subprocess
import sys
import urllib.request
from pathlib import Path

os.environ["XFORMERS_DISABLED"] = "1"  # read by upstream at import time

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.stdout.reconfigure(line_buffering=True)
# This checks a tree you just built; reaching for the published bucket would verify other bytes.
os.environ.setdefault("MOZO_BASE_URL", f"file://{ROOT / 'weights'}")

import warnings  # noqa: E402

# Upstream announces at import whether it found xFormers; XFORMERS_DISABLED makes that "disabled".
warnings.filterwarnings("ignore", message="xFormers is")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from common import fixtures as photographs  # noqa: E402
from fetch.dinov2 import COMMIT, authors_url  # noqa: E402

from mozo.vendors.dinov2_deploy import SPECS, Encoder  # noqa: E402
from mozo.vendors.dinov2_deploy import image as vendor_image  # noqa: E402

#: The versions parity was established against. Printed, not enforced.
PINNED = {"torch": "2.11.0", "torchvision": "0.26.0"}

#: Perturbations of the vendor: name -> (what, the stage expected to fail first).
FALSIFIERS = {
    "crop_floor": ("centre crop at (w-224)//2, as transformers does", "pixels"),
    "squash": ("Resize((224, 224)) instead of the short side", "pixels"),
    "bgr": ("channels reversed before the transform", "pixels"),
    "offset": ("position-embedding offset 0.1 -> 0 (plain models)", "tokens"),
    "antialias": ("position-embedding antialiasing flipped", "tokens"),
    "ln_eps": ("block 0 norm1 eps 1e-6 -> 1e-5", "blocks.0"),
    "layerscale": ("block 3 ls2 skipped", "blocks.3"),
    "gelu_tanh": ("MLP GELU -> tanh approximation", "blocks.0"),
    "attn_scale": ("attention scale x (1 + 1e-6)", "blocks.0"),
    "no_final_norm": ("final LayerNorm skipped", "norm"),
    "unnormalised": ("vector returned without L2 normalisation", "vector"),
}


class Gate:
    """Exact comparisons, with the first failing stage remembered."""

    def __init__(self) -> None:
        self.count = 0
        self.failures: list[tuple[str, float]] = []

    @property
    def failed(self) -> bool:
        return bool(self.failures)

    def check(self, stage: str, ours: torch.Tensor, theirs: torch.Tensor, label: str = "") -> bool:
        self.count += 1
        if ours.shape == theirs.shape and torch.equal(ours, theirs):
            return True
        delta = (float((ours.float() - theirs.float()).abs().max())
                 if ours.shape == theirs.shape else float("inf"))
        self.failures.append((stage, delta))
        print(f"    FAIL {stage:16s} {label:24s} max|delta| = {delta:.3e}")
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
    from dinov2.data.transforms import make_classification_eval_transform
    from dinov2.hub import backbones
    from dinov2.models import vision_transformer  # noqa: F401  (the hub imports it lazily)
    sys.path.remove(str(checkout))
    if "xformers" in sys.modules:
        raise SystemExit("xformers was imported; the reference must take its plain path")
    return backbones, make_classification_eval_transform()


def authors_checkpoint(variant: str, directory: Path) -> Path:
    """The authors' own ``.pth`` for *variant*, downloaded into *directory* if absent."""
    url = authors_url(variant)
    target = directory / url.rsplit("/", 1)[1]
    if not target.is_file():
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {url}")
        part = target.with_name(target.name + ".part")
        urllib.request.urlretrieve(url, part)
        part.replace(target)
    return target


def fixtures(coco: Path | None, count: int, seed: int = 0) -> list[tuple[str, np.ndarray]]:
    """RGB images: the repo's photographs, awkward synthetic cases, and random COCO val."""
    frames = [(p.name, np.asarray(Image.open(p).convert("RGB"))) for p in photographs()]
    rng = np.random.default_rng(seed)
    example = Image.fromarray(frames[0][1])
    frames += [
        ("noise-343x256", rng.integers(0, 256, (256, 343, 3), dtype=np.uint8)),  # odd margin
        ("odd-383x257", np.asarray(example.resize((383, 257)))),
        ("black-224", np.zeros((224, 224, 3), np.uint8)),
        ("tiny-9x13", np.asarray(example.resize((9, 13)))),
        ("one-pixel", np.asarray(example)[:1, :1].copy()),
        ("wide-2000x120", np.asarray(example.resize((2000, 120)))),
        ("tall-90x1500", np.asarray(example.resize((90, 1500)))),
        ("grey", np.asarray(example.convert("L").convert("RGB"))),
    ]
    if coco is not None and count:
        names = sorted(glob.glob(str(coco / "*.jpg")))
        for path in random.Random(seed).sample(names, count):
            frames.append((Path(path).name, np.asarray(Image.open(path).convert("RGB"))))
    return frames


def capture(model: torch.nn.Module, names: list[str]) -> tuple[dict, list]:
    """Record each named module's output (the last call), via forward hooks."""
    seen: dict[str, torch.Tensor] = {}
    handles = []
    modules = dict(model.named_modules())
    for name in names:
        def hook(_module, _inputs, output, name=name):
            seen[name] = output.detach().clone()
        handles.append(modules[name].register_forward_hook(hook))
    return seen, handles


def tokens_of(model, x: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        return model.prepare_tokens_with_masks(x)


def compare_variant(variant: str, backbones, transform, authors: Path,
                    frames: list[tuple[str, np.ndarray]], gate: Gate,
                    falsifier: str | None = None) -> Encoder:
    """Every stage for one variant. Returns the vendor encoder, for the mozo path."""
    from mozo.weights import resolve

    spec = SPECS[variant]
    original = authors_checkpoint(variant, authors)
    published = resolve("dinov2", variant, "torch-fp32")

    # --- the checkpoint ---
    theirs_state = torch.load(original, map_location="cpu", weights_only=True)
    ours_state = torch.load(published, map_location="cpu", weights_only=True)
    if set(theirs_state) != set(ours_state):
        gate.failures.append(("checkpoint", float("inf")))
        print(f"    FAIL checkpoint keys differ: {sorted(set(theirs_state) ^ set(ours_state))[:5]}")
    for key in sorted(theirs_state.keys() & ours_state.keys()):
        gate.check("checkpoint", ours_state[key], theirs_state[key], key)
    del ours_state

    # --- the two models ---
    reference = getattr(backbones, spec.upstream)(pretrained=False)
    reference.load_state_dict(theirs_state, strict=True)
    reference.eval()
    del theirs_state

    encoder = Encoder(published, spec, device="cpu")
    network = encoder.network()
    if falsifier:
        perturb(falsifier, encoder, network)

    names = [f"blocks.{i}" for i in range(spec.depth)] + ["norm"]
    ours_seen, ours_hooks = capture(network, names)
    theirs_seen, theirs_hooks = capture(reference, names)

    # --- per image, every stage ---
    theirs_vectors, pixels = [], []
    for label, rgb in frames:
        mine = vendor_image.preprocess(rgb)
        x = transform(Image.fromarray(rgb))
        gate.check("pixels", mine, x, label)
        pixels.append(mine)
        gate.check("tokens", tokens_of(network, mine[None]), tokens_of(reference, x[None]), label)
        with torch.inference_mode():
            ours_cls = network(mine[None])
            theirs_cls = reference(x[None])
        for name in names:
            gate.check(name, ours_seen[name], theirs_seen[name], label)
        gate.check("cls", ours_cls, theirs_cls, label)
        theirs_vectors.append(F.normalize(theirs_cls, dim=1, p=2))

    # --- the vector, through the encoder's own path, all images in one call ---
    ours_vectors = encoder.encode_image([rgb for _, rgb in frames])
    for i, (label, _) in enumerate(frames):
        gate.check("vector", ours_vectors[i:i + 1], theirs_vectors[i], label)

    for handle in ours_hooks + theirs_hooks:
        handle.remove()

    # --- the backbone off the 224 path: no interpolation, and a non-square grid ---
    generator = torch.Generator().manual_seed(0)
    for shape in ((518, 518), (224, 336)):
        x = torch.randn(1, 3, *shape, generator=generator)
        with torch.inference_mode():
            gate.check("backbone", network(x), reference(x), f"{shape[0]}x{shape[1]}")

    encoder.reference_vectors = torch.cat(theirs_vectors)  # for the mozo path
    return encoder


def perturb(name: str, encoder: Encoder, network) -> None:
    """Apply one falsifier to the vendor (and only the vendor)."""
    import torchvision.transforms.functional as TF

    from mozo.vendors.dinov2_deploy.config import CROP, MEAN, RESIZE, STD

    original = vendor_image.preprocess
    if name == "crop_floor":
        def preprocess(image):
            resized = TF.resize(Image.fromarray(image), RESIZE,
                                interpolation=TF.InterpolationMode.BICUBIC)
            w, h = resized.size
            top, left = (h - CROP) // 2, (w - CROP) // 2
            cropped = resized.crop((left, top, left + CROP, top + CROP))
            return TF.normalize(TF.to_tensor(cropped), mean=list(MEAN), std=list(STD))
        vendor_image.preprocess = preprocess
    elif name == "squash":
        def preprocess(image):
            resized = TF.resize(Image.fromarray(image), [CROP, CROP],
                                interpolation=TF.InterpolationMode.BICUBIC)
            return TF.normalize(TF.to_tensor(resized), mean=list(MEAN), std=list(STD))
        vendor_image.preprocess = preprocess
    elif name == "bgr":
        vendor_image.preprocess = lambda image: original(np.ascontiguousarray(image[..., ::-1]))
    elif name == "offset":
        network.interpolate_offset = 0.0
    elif name == "antialias":
        network.interpolate_antialias = not network.interpolate_antialias
    elif name == "ln_eps":
        network.blocks[0].norm1.eps = 1e-5
    elif name == "layerscale":
        network.blocks[3].ls2 = torch.nn.Identity()
    elif name == "gelu_tanh":
        for block in network.blocks:
            if hasattr(block.mlp, "act"):
                block.mlp.act = torch.nn.GELU(approximate="tanh")
    elif name == "attn_scale":
        # Per instance, so the reference -- which calls the same torch function -- is untouched.
        def scaled(self, x, is_causal=False):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
            q, k, v = [t.transpose(1, 2) for t in torch.unbind(qkv, 2)]
            x = F.scaled_dot_product_attention(q, k, v, scale=self.scale * (1 + 1e-6))
            return self.proj(x.transpose(1, 2).contiguous().view(B, N, C))
        for block in network.blocks:
            block.attn.forward = scaled.__get__(block.attn)
    elif name == "no_final_norm":
        network.norm = torch.nn.Identity()
    elif name == "unnormalised":
        encoder.encode_image = lambda images: torch.cat(
            [network(original(rgb)[None]) for rgb in images])
    else:
        raise SystemExit(f"unknown falsifier {name!r}; known: {sorted(FALSIFIERS)}")
    # The stage functions above call ``vendor_image.preprocess`` by attribute, so the vector path
    # is perturbed with them.
    if name in ("crop_floor", "squash", "bgr"):
        import mozo.vendors.dinov2_deploy.predictor as predictor
        predictor.preprocess = vendor_image.preprocess


def against_mozo(variant: str, vendor: Encoder, frames, paths: list[Path], gate: Gate) -> None:
    """mozo's adapter, through the registry and the weights tree, against upstream."""
    import mozo
    from mozo.image import load_image

    model = mozo.get_model(f"dinov2/{variant}", device="cpu")
    if model.runtime != "torch-fp32":
        print(f"    note: mozo chose {model.runtime}; comparing that")
    got = torch.from_numpy(model.encode_image([rgb for _, rgb in frames]))
    for i, (label, _) in enumerate(frames):
        gate.check("mozo", got[i:i + 1], vendor.reference_vectors[i:i + 1], label)
    # File paths go through mozo.image.load_image on mozo's side; upstream gets the same RGB.
    decoded = [load_image(str(p)) for p in paths]
    want = torch.cat([vendor.encode_image(rgb) for rgb in decoded])
    got = torch.from_numpy(model.encode_image([str(p) for p in paths]))
    gate.check("mozo-paths", got, want, f"{len(paths)} files")


def versions() -> str:
    import torchvision

    return f"torch {torch.__version__}, torchvision {torchvision.__version__} (pinned {PINNED})"


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
                        help=f"a facebookresearch/dinov2 checkout at {COMMIT[:7]}")
    parser.add_argument("--authors", type=Path, required=True,
                        help="directory holding the authors' *_pretrain.pth (filled if empty)")
    parser.add_argument("--mozo", action="store_true", help="also compare mozo's adapter")
    parser.add_argument("--coco", type=Path, default=None, help="a directory of .jpg to sample")
    parser.add_argument("--count", type=int, default=10, help="how many --coco images")
    parser.add_argument("--falsify", default=None,
                        help="a falsifier name, 'list' or 'all' (default variant: base)")
    args = parser.parse_args()

    if args.falsify == "list":
        for name, (what, stage) in FALSIFIERS.items():
            print(f"{name:14s} {what}  [expected at: {stage}]")
        return 0
    if args.falsify == "all":
        return falsify_all(sys.argv[1:])

    if os.environ.get("MKL_CBWR") != "AVX2,STRICT":
        print("WARNING: MKL_CBWR is not AVX2,STRICT; CPU results may differ in the last bit")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    variants = args.variants or (["base"] if args.falsify else list(SPECS))
    print(f"DINOv2 gate: {versions()}, upstream {COMMIT[:7]}, CPU")
    backbones, transform = load_upstream(args.upstream)
    frames = fixtures(args.coco, args.count)
    print(f"{len(frames)} images: {', '.join(label for label, _ in frames[:12])}"
          + (", ..." if len(frames) > 12 else ""))

    gate = Gate()
    for variant in variants:
        before = (gate.count, len(gate.failures))
        print(f"\n[{variant}]")
        vendor = compare_variant(variant, backbones, transform, args.authors, frames, gate,
                                 falsifier=args.falsify)
        if args.mozo and not args.falsify:
            against_mozo(variant, vendor, frames, photographs(), gate)
        count, failed = gate.count - before[0], len(gate.failures) - before[1]
        print(f"  {variant}: {count - failed} of {count} comparisons exact")

    if args.falsify:
        what, expected = FALSIFIERS[args.falsify]
        first = gate.failures[0][0] if gate.failures else None
        where = ("CAUGHT at the expected stage" if first == expected
                 else f"CAUGHT, but first at {first!r} (expected {expected!r})" if first
                 else "NOT CAUGHT")
        stages = sorted({stage for stage, _ in gate.failures}, key=lambda s: s)
        print(f"falsifier {args.falsify!r} ({what}): {where}; "
              f"{len(gate.failures)} of {gate.count} comparisons failed, stages {stages}")
        return 0 if first == expected else 1

    print(f"\n{gate.count - len(gate.failures)} of {gate.count} comparisons exact"
          + ("" if not gate.failed else f"; {len(gate.failures)} FAILED"))
    return 1 if gate.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

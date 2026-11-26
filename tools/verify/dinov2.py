#!/usr/bin/env python3
"""Verify the vendored DINOv2 against the authors' torch-hub code, stage by stage, exactly.

    python -I tools/verify/dinov2.py --device cuda --upstream /path/to/dinov2 \\
        --authors /path/to/pth-dir [--mozo] [--coco /path/to/val2017 --count 20] [variants...]
    python -I tools/verify/dinov2.py --device cuda ... --falsify all
    MKL_CBWR=AVX2,STRICT python -I tools/verify/dinov2.py ...          # the CPU gate (the default)

The reference is ``facebookresearch/dinov2`` at commit ``7764ea0`` (Apache-2.0), the code
``torch.hub.load('facebookresearch/dinov2', ...)`` serves: its hub constructors with a strict load
of the **authors' own** ``*_pretrain.pth`` (``--authors``, downloaded there if absent), and its
evaluation transform ``make_classification_eval_transform``. Three paths are compared in one
process, on one device, with ``torch.equal`` and no tolerance:

1. **upstream** -- the above, on a PIL image, with ``F.normalize(model(x), dim=1, p=2)`` as its own
   k-NN evaluation applies (``ModelWithNormalize``), one image per forward. On CUDA it runs as that
   evaluation does: the transform on the host, the batch moved to the GPU, the network and the
   normalisation there.
2. **vendor** -- :class:`mozo.vendors.dinov2_deploy.Encoder` on mozo's published artifact.
3. **mozo** -- ``Dinov2Predictor`` through the registry and the weights tree (``--mozo``), on the
   ``torch-fp32`` runtime, on file paths decoded by ``mozo.image.load_image`` and on arrays, against
   upstream on the same RGB.

Stages: the published checkpoint against the authors' file (every key, every tensor, by bytes);
the pixels; the tokens after the position embedding (and registers); every block's output; the
final norm; the class token; the normalised vector. The backbone alone is also compared at
518x518 (no interpolation) and 224x336 (non-square) on random input, so both interpolation
branches and the early return are covered for every variant. Last, the attention ops and kernels
each side ran (from ``torch.profiler``) must be the same.

Upstream runs first, alone on the device, and its stages are kept on the host; then it is freed and
the vendor runs, then mozo. So the giant (4.5 GB in fp32) never needs two copies on a 12 GB card,
and the device holds one model at a time for every variant alike.

**The attention path.** Upstream's ``MemEffAttention`` takes xFormers' kernel whenever ``xformers``
is importable; the vendor has only the plain path. A plain ``pip install`` of mozo and torch brings
no xFormers, so the gate runs upstream the way such a user would: ``XFORMERS_DISABLED=1`` is set
before upstream is imported, and the run stops if ``xformers`` was imported anyway. Both sides then
call ``F.scaled_dot_product_attention``; the kernels it picked are printed and compared.

**On CUDA** (``--device cuda``) the global settings are torch's defaults, which mozo's runtime
leaves alone: cuDNN may use TF32 (the patch-embedding convolution), matmuls may not, every SDPA
backend is enabled, ``cudnn.benchmark`` and deterministic algorithms are off. They are printed and
checked unchanged after both models are loaded. Each side runs every image twice, and the two
off-224 backbone shapes five times, and must repeat itself bit for bit, or a match would mean
nothing (VENDORING §2.7).

``--dtype`` is fp32 only. mozo's CUDA runtimes for DINOv2 through torch are ``torch-fp32`` alone:
``auto`` on CUDA takes a TensorRT fp32 engine where a graph is published (every variant but giant,
which runs ``torch-fp32``), tensorrt-fp16 is on request, and those are measured by parity and
accuracy instead. No torch half runtime is published (an index must not change with the runtime
``auto`` picked), and upstream has no half-precision inference path of its own.

**On CPU, set ``MKL_CBWR=AVX2,STRICT``.** MKL picks kernels by buffer alignment, so two CPU torch
processes can differ in the last bit without it (VENDORING §5).

Batching is not among the CPU falsifiers: under ``MKL_CBWR=AVX2,STRICT`` one forward over nine
images is bit-identical to nine single forwards on the CPU, so the gate cannot see it. Without that
setting -- as users run -- it is not (1.5e-05 on base, 1.8e-05 on large), which is why the encoder
runs one image at a time anyway. On CUDA a batch differs, so there it is a falsifier.

``--falsify NAME`` perturbs one thing in the vendor and expects the gate to fail at the stage
named; ``--falsify list`` names them, ``--falsify all`` runs every one (each in a fresh process)
and fails unless each was caught where expected. On CUDA the falsifiers only a GPU can show are
added (:data:`FALSIFIERS_CUDA`), and ``all`` first records upstream once, unperturbed, into a
reference cache (``--reference-cache DIR``, a temporary one if not given) that every falsifier
reuses: upstream never changes between them. The cache keeps large stages as their shape, dtype
and sha256.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import random
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import NamedTuple

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
warnings.filterwarnings("ignore", message="Warning: Profiler clears events")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from common import fixtures as photographs  # noqa: E402
from fetch.dinov2 import COMMIT, authors_url  # noqa: E402

from mozo.vendors.dinov2_deploy import SPECS, Encoder  # noqa: E402
from mozo.vendors.dinov2_deploy import predictor  # noqa: E402

#: The versions CPU parity was first established against. Printed, not enforced.
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

#: Perturbations only a GPU run can show, added on ``--device cuda``. cuDNN TF32 off around the
#: patch embedding is not among them: on an RTX 3060 (torch 2.13, cuDNN 9.10) the 14x14 stride-14
#: convolution's output is identical with it on or off, so it cannot be caught -- the
#: preprocessing moved to the GPU, as an adapter might "optimise" it, stands in its place.
FALSIFIERS_CUDA = {
    "tf32-on": ("matmul TF32 on around the vendor's forward", "blocks.0"),
    "pixels-on-device": ("the uint8 crop moved to the GPU before /255 and the normalisation",
                         "pixels"),
    "sdpa-math": ("SDPA pinned to the math backend in the vendor", "blocks.0"),
    "channels-last": ("pixels handed to the vendor's patch embedding channels-last", "tokens"),
    "host-normalise": ("class token moved to the CPU before F.normalize", "vector"),
    "batched": ("every image in one forward, as a batch", "vector"),
}

#: How often each side runs the off-224 backbone shapes on CUDA, the inputs with the longest
#: sequences (1,369 tokens at 518x518) and so the likeliest to show a non-deterministic kernel.
BACKBONE_REPEATS = 5

#: Stages small enough to keep whole in the reference cache; larger ones keep their digest.
WHOLE = 4096


class Digest(NamedTuple):
    """A tensor's shape, dtype and the sha256 of its bytes: what the cache keeps of a large one."""

    shape: tuple
    dtype: str
    sha256: str


def digest(tensor: torch.Tensor) -> Digest:
    flat = tensor.detach().cpu().contiguous().reshape(-1)
    return Digest(tuple(tensor.shape), str(tensor.dtype),
                  hashlib.sha256(flat.view(torch.uint8).numpy().tobytes()).hexdigest())


def compact(value):
    """*value* with every large tensor replaced by its digest, for the cache file."""
    if isinstance(value, torch.Tensor):
        return value if value.numel() <= WHOLE else digest(value)
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [compact(v) for v in value]
    return value


class Gate:
    """Exact comparisons, with the first failing stage remembered."""

    def __init__(self) -> None:
        self.count = 0
        self.repeats = 0  # of ``count``, a side against itself (the determinism check)
        self.failures: list[tuple[str, float]] = []

    @property
    def failed(self) -> bool:
        return bool(self.failures)

    def check(self, stage: str, ours, theirs, label: str = "") -> bool:
        """``torch.equal`` on the host; *theirs* may be a :class:`Digest` from the cache."""
        self.count += 1
        self.repeats += stage.startswith("repeat:")
        ours = ours.detach().cpu() if isinstance(ours, torch.Tensor) else ours
        if isinstance(theirs, Digest):
            if digest(ours) == theirs:
                return True
            self.failures.append((stage, float("nan")))
            print(f"    FAIL {stage:16s} {label:24s} bytes differ from the cached reference")
            return False
        if isinstance(ours, torch.Tensor):
            theirs = theirs.detach().cpu()
            if ours.shape == theirs.shape and ours.dtype == theirs.dtype and torch.equal(ours, theirs):
                return True
            delta = (float((ours.float() - theirs.float()).abs().max())
                     if ours.shape == theirs.shape else float("inf"))
            self.failures.append((stage, delta))
            print(f"    FAIL {stage:16s} {label:24s} max|delta| = {delta:.3e}")
            return False
        if ours == theirs:
            return True
        self.failures.append((stage, float("inf")))
        print(f"    FAIL {stage:16s} {label:24s} {ours!r} != {theirs!r}")
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
    from dinov2.layers import attention
    from dinov2.models import vision_transformer  # noqa: F401  (the hub imports it lazily)
    sys.path.remove(str(checkout))
    if "xformers" in sys.modules or attention.XFORMERS_AVAILABLE:
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


def stage_names(variant: str) -> list[str]:
    return [f"blocks.{i}" for i in range(SPECS[variant].depth)] + ["norm"]


def forward_stages(model, x: torch.Tensor, seen: dict, names: list[str]) -> dict:
    """Tokens, every block, the norm and the class token for one ``(1, 3, H, W)`` input, on the
    host."""
    with torch.inference_mode():
        stages = {"tokens": model.prepare_tokens_with_masks(x).cpu()}
        cls = model(x)
    stages.update({name: seen[name].cpu() for name in names})
    stages["cls"] = cls.cpu()
    return stages, cls


def attention_kernels(model, x: torch.Tensor) -> tuple[str, ...]:
    """The attention ops and kernels one forward of *model* ran, by name, from the profiler."""
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if x.is_cuda else [])
    with torch.inference_mode(), profile(activities=activities) as prof:
        model(x)
        if x.is_cuda:
            torch.cuda.synchronize()
    pattern = re.compile(r"attention|fmha|flash|sdpa|xformers|cutlass", re.IGNORECASE)
    return tuple(sorted({e.name for e in prof.events() if pattern.search(e.name)}))


def repeat_check(gate: Gate, side: str, first: dict, second: dict, label: str) -> None:
    """A side must repeat itself bit for bit before a match against the other side means much."""
    for stage, value in first.items():
        gate.check(f"repeat:{side}", second[stage], value, f"{label} {stage}")


def settings() -> dict:
    """The global settings a CUDA result depends on."""
    return {
        "cudnn.allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn.benchmark": torch.backends.cudnn.benchmark,
        "cudnn.deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "sdp flash/mem_efficient/math/cudnn": (
            torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.mem_efficient_sdp_enabled(),
            torch.backends.cuda.math_sdp_enabled(), torch.backends.cuda.cudnn_sdp_enabled()),
    }


def environment(device: str) -> str:
    """torch, CUDA, cuDNN, the GPU, its driver and the settings in force: what a result holds for."""
    import torchvision

    head = (f"torch {torch.__version__}, torchvision {torchvision.__version__} "
            f"(CPU parity first pinned at {PINNED})")
    if device == "cpu":
        return f"{head}, CPU, MKL_CBWR={os.environ.get('MKL_CBWR', '')}"
    driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader",
                             "-i", "0"], capture_output=True, text=True).stdout.strip()
    return (f"{head}, CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}, "
            f"{torch.cuda.get_device_name(0)}, driver {driver or 'unknown'}; {settings()}")


def free(device: str) -> None:
    import gc

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


# -- upstream -------------------------------------------------------------------------------------


def reference_records(variant: str, backbones, transform, authors: Path,
                      frames: list[tuple[str, np.ndarray]], decoded: list[np.ndarray],
                      device: str, gate: Gate) -> dict:
    """Upstream's every stage for one variant, on the host. Each image runs twice on CUDA and
    must repeat itself."""
    spec = SPECS[variant]
    cuda = device == "cuda"
    theirs_state = torch.load(authors_checkpoint(variant, authors), map_location="cpu",
                              weights_only=True)
    records: dict = {"checkpoint": {key: digest(t) for key, t in theirs_state.items()}}
    reference = getattr(backbones, spec.upstream)(pretrained=False)
    reference.load_state_dict(theirs_state, strict=True)
    del theirs_state
    reference.eval().to(device)  # upstream's evaluations: ``model.cuda()``

    names = stage_names(variant)
    seen, hooks = capture(reference, names)
    records["frames"] = []
    for label, rgb in frames:
        x = transform(Image.fromarray(rgb))
        stages, cls = forward_stages(reference, x[None].to(device), seen, names)
        stages["vector"] = F.normalize(cls, dim=1, p=2).cpu()  # ``ModelWithNormalize``, on device
        if cuda:
            again, cls = forward_stages(reference, x[None].to(device), seen, names)
            again["vector"] = F.normalize(cls, dim=1, p=2).cpu()
            repeat_check(gate, "upstream", stages, again, label)
        stages["pixels"] = x
        records["frames"].append(stages)
    for handle in hooks:
        handle.remove()

    with torch.inference_mode():
        records["paths"] = [
            F.normalize(reference(transform(Image.fromarray(rgb))[None].to(device)),
                        dim=1, p=2).cpu() for rgb in decoded]
        records["backbone"] = {}
        for label, x in backbone_inputs():
            outs = [reference(x.to(device)).cpu() for _ in range(BACKBONE_REPEATS if cuda else 1)]
            for again in outs[1:]:
                gate.check("repeat:upstream", again, outs[0], f"backbone {label}")
            records["backbone"][label] = outs[0]
    first = transform(Image.fromarray(frames[0][1]))[None].to(device)
    records["kernels"] = attention_kernels(reference, first)
    del reference
    free(device)
    return records


def backbone_inputs() -> list[tuple[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(0)
    return [(f"{h}x{w}", torch.randn(1, 3, h, w, generator=generator))
            for h, w in ((518, 518), (224, 336))]


# -- the vendor and mozo -------------------------------------------------------------------------


def compare_vendor(variant: str, records: dict, frames: list[tuple[str, np.ndarray]],
                   device: str, gate: Gate, falsifier: str | None = None) -> None:
    """Every stage of the vendor against upstream's records."""
    from mozo.weights import resolve

    spec = SPECS[variant]
    cuda = device == "cuda"
    published = resolve("dinov2", variant, "torch-fp32")

    # --- the checkpoint: the published file, as the vendor loaded it (``assign=True`` keeps the
    # file's own tensors), against the authors', by bytes ---
    encoder = Encoder(published, spec, device=device)
    network = encoder.network()
    ours_state = network.state_dict()
    theirs_state = records["checkpoint"]
    if set(theirs_state) != set(ours_state):
        gate.failures.append(("checkpoint", float("inf")))
        print(f"    FAIL checkpoint keys differ: {sorted(set(theirs_state) ^ set(ours_state))[:5]}")
    for key in sorted(theirs_state.keys() & ours_state.keys()):
        gate.check("checkpoint", ours_state[key], theirs_state[key], key)
    del ours_state

    if falsifier:
        perturb(falsifier, encoder, network)

    names = stage_names(variant)
    seen, hooks = capture(network, names)
    for (label, rgb), theirs in zip(frames, records["frames"]):
        mine = predictor.preprocess(rgb)  # the binding the encoder calls, so one patch covers both
        gate.check("pixels", mine, theirs["pixels"], label)
        x = mine[None].to(device)  # as the encoder places it
        stages, _ = forward_stages(network, x, seen, names)
        if cuda:
            again, _ = forward_stages(network, x, seen, names)
            repeat_check(gate, "vendor", stages, again, label)
        for stage in ["tokens", *names, "cls"]:
            gate.check(stage, stages[stage], theirs[stage], label)
    for handle in hooks:
        handle.remove()

    # --- the vector, through the encoder's own path, all images in one call ---
    images = [rgb for _, rgb in frames]
    ours_vectors = encoder.encode_image(images)
    if cuda:
        gate.check("repeat:vendor", encoder.encode_image(images), ours_vectors, "vectors")
    for i, ((label, _), theirs) in enumerate(zip(frames, records["frames"])):
        gate.check("vector", ours_vectors[i:i + 1], theirs["vector"], label)

    # --- the backbone off the 224 path: no interpolation, and a non-square grid ---
    with torch.inference_mode():
        for label, x in backbone_inputs():
            outs = [network(x.to(device)).cpu() for _ in range(BACKBONE_REPEATS if cuda else 1)]
            for again in outs[1:]:
                gate.check("repeat:vendor", again, outs[0], f"backbone {label}")
            gate.check("backbone", outs[0], records["backbone"][label], label)

    # --- the attention path each side took ---
    kernels = attention_kernels(network, predictor.preprocess(frames[0][1])[None].to(device))
    print(f"    attention, upstream: {', '.join(records['kernels'])}")
    print(f"    attention, vendor:   {', '.join(kernels)}")
    gate.check("kernels", kernels, records["kernels"], "attention ops and kernels")
    del encoder, network
    free(device)


def perturb(name: str, encoder: Encoder, network) -> None:
    """Apply one falsifier to the vendor (and only the vendor)."""
    import contextlib

    import torchvision.transforms.functional as TF

    from mozo.vendors.dinov2_deploy.config import CROP, MEAN, RESIZE, STD

    original = predictor.preprocess
    device = encoder.device

    def around(method: str, context) -> None:
        """Run *network.method* inside *context*, on this instance only."""
        inner = getattr(network, method)

        def wrapped(*args, **kwargs):
            with context():
                return inner(*args, **kwargs)
        setattr(network, method, wrapped)

    @contextlib.contextmanager
    def flag(owner, attribute: str, value: bool):
        before = getattr(owner, attribute)
        setattr(owner, attribute, value)
        try:
            yield
        finally:
            setattr(owner, attribute, before)

    if name == "crop_floor":
        def preprocess(image):
            resized = TF.resize(Image.fromarray(image), RESIZE,
                                interpolation=TF.InterpolationMode.BICUBIC)
            w, h = resized.size
            top, left = (h - CROP) // 2, (w - CROP) // 2
            cropped = resized.crop((left, top, left + CROP, top + CROP))
            return TF.normalize(TF.to_tensor(cropped), mean=list(MEAN), std=list(STD))
        predictor.preprocess = preprocess
    elif name == "squash":
        def preprocess(image):
            resized = TF.resize(Image.fromarray(image), [CROP, CROP],
                                interpolation=TF.InterpolationMode.BICUBIC)
            return TF.normalize(TF.to_tensor(resized), mean=list(MEAN), std=list(STD))
        predictor.preprocess = preprocess
    elif name == "bgr":
        predictor.preprocess = lambda image: original(np.ascontiguousarray(image[..., ::-1]))
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
        def unnormalised(images):
            with torch.inference_mode():
                return torch.cat([network(original(rgb)[None].to(device)) for rgb in images]).cpu()
        encoder.encode_image = unnormalised
    elif name == "tf32-on":
        around("forward", lambda: flag(torch.backends.cuda.matmul, "allow_tf32", True))
    elif name == "pixels-on-device":
        def preprocess(image):
            resized = TF.resize(Image.fromarray(image), RESIZE,
                                interpolation=TF.InterpolationMode.BICUBIC)
            crop = torch.from_numpy(np.asarray(TF.center_crop(resized, CROP)).copy())
            pixels = crop.permute(2, 0, 1).to(device).float().div(255)
            return TF.normalize(pixels, mean=list(MEAN), std=list(STD)).cpu()
        predictor.preprocess = preprocess
    elif name == "sdpa-math":
        from torch.nn.attention import SDPBackend, sdpa_kernel
        around("forward", lambda: sdpa_kernel(SDPBackend.MATH))
    elif name == "channels-last":
        inner = network.prepare_tokens_with_masks
        network.prepare_tokens_with_masks = lambda x, masks=None: inner(
            x.contiguous(memory_format=torch.channels_last), masks)
    elif name == "host-normalise":
        def on_host(images):
            with torch.inference_mode():
                return torch.cat([F.normalize(network(original(rgb)[None].to(device)).cpu(),
                                              dim=1, p=2) for rgb in images])
        encoder.encode_image = on_host
    elif name == "batched":
        def batched(images):
            with torch.inference_mode():
                pixels = torch.stack([original(rgb) for rgb in images]).to(device)
                return F.normalize(network(pixels), dim=1, p=2).cpu()
        encoder.encode_image = batched
    else:
        raise SystemExit(f"unknown falsifier {name!r}")


def against_mozo(variant: str, records: dict, frames, paths: list[Path], device: str,
                 gate: Gate) -> None:
    """mozo's adapter, through the registry and the weights tree, on ``torch-fp32``, against
    upstream."""
    from mozo import registry
    from mozo.manager import ModelManager
    from mozo.runtimes import select_runtime
    from mozo.weights import artifacts

    auto = select_runtime(device, artifacts("dinov2", variant), "auto",
                          executes=registry.EXECUTES["dinov2"],
                          derives=registry.DERIVES.get("dinov2"))
    print(f"    mozo: comparing runtime torch-fp32 ('auto' on {device} picks {auto})")
    model = ModelManager().get_model("dinov2", variant, device, runtime="torch-fp32")
    images = [rgb for _, rgb in frames]
    got = torch.from_numpy(model.encode_image(images))
    if device == "cuda":
        gate.check("repeat:mozo", torch.from_numpy(model.encode_image(images)), got, "vectors")
    for i, ((label, _), theirs) in enumerate(zip(frames, records["frames"])):
        gate.check("mozo", got[i:i + 1], theirs["vector"], label)
    # File paths go through mozo.image.load_image on mozo's side; upstream got the same RGB.
    got = torch.from_numpy(model.encode_image([str(p) for p in paths]))
    gate.check("mozo-paths", got, torch.cat(records["paths"]), f"{len(paths)} files")
    del model
    free(device)


# -- driving -------------------------------------------------------------------------------------


def falsify_all(argv: list[str], names: list[str], device: str) -> int:
    """Run every falsifier in a process of its own and report which were caught where.

    On CUDA upstream is first recorded once, by an unperturbed run that must itself pass, into a
    reference cache every falsifier then reads; a second unperturbed run must pass against the
    cache, so a falsifier's failure is the perturbation and not the cache.
    """
    base, skip = [], False
    for a in argv:
        if skip:
            skip = False
        elif a == "--falsify":
            skip = True
        elif not a.startswith("--falsify="):
            base.append(a)
    if not any(a for a in base if not a.startswith("-") and a in SPECS):
        base.append("base")
    with tempfile.TemporaryDirectory() as scratch:
        if device == "cuda":
            if not any(a.startswith("--reference-cache") for a in base):
                base += ["--reference-cache", scratch]
            for what in ("reference", "against the cache"):
                run = subprocess.run([sys.executable, "-I", __file__, *base], capture_output=True,
                                     text=True)
                lines = run.stdout.strip().splitlines() or [run.stderr[-2000:]]
                print(f"unperturbed, {what}: {lines[-1].strip()}")
                if run.returncode != 0:
                    print(run.stdout[-3000:], run.stderr[-3000:])
                    return 1
        missed = []
        for name in names:
            run = subprocess.run([sys.executable, "-I", __file__, *base, "--falsify", name],
                                 capture_output=True, text=True)
            verdict = [line for line in run.stdout.splitlines() if line.startswith("falsifier")]
            print(verdict[-1] if verdict else f"falsifier {name!r}: no verdict\n{run.stderr[-2000:]}")
            if not verdict or "CAUGHT at the expected stage" not in verdict[-1]:
                missed.append(name)
    print(f"\n{len(names) - len(missed)} of {len(names)} falsifiers caught where expected"
          + (f"; missed: {missed}" if missed else ""))
    return 1 if missed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variants", nargs="*", help=f"default: all of {list(SPECS)}")
    parser.add_argument("--upstream", type=Path, default=None,
                        help=f"a facebookresearch/dinov2 checkout at {COMMIT[:7]} "
                             "(not needed when every variant is in --reference-cache)")
    parser.add_argument("--authors", type=Path, default=None,
                        help="directory holding the authors' *_pretrain.pth (filled if empty)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--mozo", action="store_true", help="also compare mozo's adapter")
    parser.add_argument("--coco", type=Path, default=None, help="a directory of .jpg to sample")
    parser.add_argument("--count", type=int, default=10, help="how many --coco images")
    parser.add_argument("--reference-cache", type=Path, default=None,
                        help="a directory: upstream's records per variant are read from it, or "
                             "computed and saved there")
    parser.add_argument("--falsify", default=None,
                        help="a falsifier name, 'list' or 'all' (default variant: base)")
    args = parser.parse_args()

    if args.dtype != "fp32":
        parser.error("DINOv2 runs through torch in fp32 only: mozo publishes no torch half runtime "
                     "for it (its fp16 is TensorRT, on request) and upstream has no "
                     "half-precision inference path")
    falsifiers = {**FALSIFIERS, **(FALSIFIERS_CUDA if args.device == "cuda" else {})}
    if args.falsify == "list":
        for name, (what, stage) in falsifiers.items():
            print(f"{name:15s} {what}  [expected at: {stage}]")
        return 0
    if args.falsify == "all":
        return falsify_all(sys.argv[1:], list(falsifiers), args.device)
    if args.falsify is not None and args.falsify not in falsifiers:
        parser.error(f"--falsify {args.falsify} does not apply on {args.device}; "
                     f"known: {', '.join(falsifiers)}")
    if args.device == "cpu" and os.environ.get("MKL_CBWR") != "AVX2,STRICT":
        print("WARNING: MKL_CBWR is not AVX2,STRICT; CPU results may differ in the last bit")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda, but torch sees no CUDA device")

    if args.device == "cpu":
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS") or min(8, os.cpu_count() or 1)))
    variants = args.variants or (["base"] if args.falsify else list(SPECS))
    before = settings()
    print(f"DINOv2 gate: {environment(args.device)}; upstream {COMMIT[:7]}, {args.dtype}")
    frames = fixtures(args.coco, args.count)
    print(f"{len(frames)} images: {', '.join(label for label, _ in frames[:12])}"
          + (", ..." if len(frames) > 12 else ""))
    paths = photographs()
    key = (args.device, args.dtype, torch.__version__, tuple(label for label, _ in frames))

    from mozo.image import load_image

    upstream = None
    gate = Gate()
    for variant in variants:
        start = (gate.count, len(gate.failures))
        print(f"\n[{variant}]")
        cache = (args.reference_cache / f"{variant}-{args.device}-{args.dtype}.pt"
                 if args.reference_cache else None)
        records = None
        if cache is not None and cache.is_file():
            records = torch.load(cache, weights_only=False)
            if records.pop("for") != key:
                raise SystemExit(f"{cache} was made for another device, precision, torch or "
                                 "fixture set")
            print(f"  upstream: from {cache}")
        else:
            if args.upstream is None or args.authors is None:
                parser.error(f"--upstream and --authors are needed to record {variant}")
            if upstream is None:
                upstream = load_upstream(args.upstream)
            records = reference_records(variant, *upstream, args.authors, frames,
                                        [load_image(str(p)) for p in paths], args.device, gate)
        compare_vendor(variant, records, frames, args.device, gate, falsifier=args.falsify)
        if args.mozo and not args.falsify:
            against_mozo(variant, records, frames, paths, args.device, gate)
        count, failed = gate.count - start[0], len(gate.failures) - start[1]
        print(f"  {variant}: {count - failed} of {count} comparisons exact")
        if cache is not None and not cache.is_file() and not args.falsify and not failed:
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save({**compact(records), "for": key}, cache)
    if settings() != before:
        gate.failures.append(("settings", float("inf")))
        print(f"    FAIL settings changed during the run: {before} -> {settings()}")

    if args.falsify:
        what, expected = falsifiers[args.falsify]
        first = gate.failures[0][0] if gate.failures else None
        where = ("CAUGHT at the expected stage" if first == expected
                 else f"CAUGHT, but first at {first!r} (expected {expected!r})" if first
                 else "NOT CAUGHT")
        stages = sorted({stage for stage, _ in gate.failures})
        print(f"falsifier {args.falsify!r} ({what}): {where}; "
              f"{len(gate.failures)} of {gate.count} comparisons failed, stages {stages}")
        return 0 if first == expected else 1

    print(f"\n{gate.count - len(gate.failures)} of {gate.count} comparisons exact"
          + (f", {gate.repeats} of them a side against its own repeat" if gate.repeats else "")
          + ("" if not gate.failed else f"; {len(gate.failures)} FAILED"))
    return 1 if gate.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

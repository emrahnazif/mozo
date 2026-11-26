#!/usr/bin/env python3
"""Check that mozo's MM-Grounding-DINO returns exactly what the published model does.

Three paths reach the same weights and this compares all three. ``mozo/vendors/
mm_grounding_dino_deploy`` is one. ``transformers``' ``MMGroundingDinoForObjectDetection`` with its
``AutoProcessor`` is the second -- the reference, because the checkpoints are the HF-format files
the LLMDet authors publish and tell users to run with it. mozo itself is the third: registry,
weights, adapter, PixelFlow.

**The comparison is exact**, ``torch.equal`` at every stage: pixels, token ids, BERT, projected
text, every Swin level handed to the neck, encoder image and text memory, each decoder layer's
output and reference boxes, the final logits and boxes -- and the detections after
post-processing, against the processor's own ``post_process_grounded_object_detection``. mozo's
side is recorded from the vendor ``Predictor`` call itself, the code path the adapter runs.

**The reference must be transformers 4.x.** 5.x (measured on 5.9.0 and 5.12.1) loads these
checkpoints with the six decoder box heads tied to the first and every score bias set to the
first's, so it runs a different model; this script refuses to compare against it. 4.55.0 -- the
version the LLMDet README names -- and 4.57.3 both match this package exactly::

    python -m venv --system-site-packages refenv && ./refenv/bin/pip install 'transformers==4.57.3'
    MKL_CBWR=AVX2,STRICT ./refenv/bin/python tools/verify/mm_grounding_dino.py --reference

``MKL_CBWR`` pins MKL's kernel choice, which otherwise depends on buffer alignment and can move
the last bit between two processes. Without ``--reference`` only the mozo-vs-vendor half runs.

**On CUDA** (``--device cuda``), both sides run in this process on the same GPU, with the global
settings mozo's runtime leaves (TF32 for cuDNN on, for matmul off; every SDPA backend enabled;
cuDNN benchmark and deterministic off), checked unchanged after ``transformers`` is imported and
the model loaded. ``--dtype`` picks the precision; omitted, CUDA checks ``fp32`` and ``fp16``,
the ``torch-fp16`` runtime that ``auto`` picks there. fp16 is CUDA autocast around the model's
forward, mozo's mechanism; upstream has no official half-precision path for these checkpoints
(the authors evaluate in fp32), so the reference runs under the same autocast, and
post-processing runs outside it on both sides, on the device the outputs are on.
Each side runs every case twice and must repeat itself to the byte (determinism).

**Deformable attention runs the pure-torch ``grid_sample`` path on both sides.** transformers can
swap in a compiled kernel (``kernels-community/deformable-detr``) only when the ``kernels``
package is installed *and* the model is loaded with ``use_kernels=True``; the script refuses a
reference where that swap took place, and prints which path ran.

``--reference-cache DIR`` keeps upstream's records per (variant, device, precision) as each
stage's shape, dtype and sha256, and the detections whole, so falsifiers reuse one reference
run. ``--falsify NAME`` perturbs one thing on mozo's side only and expects the gate to fail;
``--kernels`` prints the CUDA kernels one case runs on each side.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import os
import random
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))

os.environ.setdefault("MOZO_BASE_URL", f"file://{ROOT / 'weights'}")

from common import fixtures  # noqa: E402
from fetch.mm_grounding_dino import CHECKPOINTS  # noqa: E402
from conftest import as_pixelflow_reports  # noqa: E402
from mozo.image import load_image  # noqa: E402
from mozo.vendors.mm_grounding_dino_deploy import SPECS, Predictor  # noqa: E402
from mozo.vendors.mm_grounding_dino_deploy.predictor import caption_for  # noqa: E402
from mozo.vendors.mm_grounding_dino_deploy.text.tokenizer import Tokenizer  # noqa: E402
from mozo.weights import WeightsError, resolve  # noqa: E402

FAMILY = "mm_grounding_dino"

#: The CUDA precision each ``--dtype`` names, as the ``torch-fp16`` runtime applies it.
PRECISIONS = {"fp32": None, "fp16": torch.float16}

#: Prompt sets: one phrase, several, multi-word phrases whose tokens must stay grouped, mixed case
#: and inner punctuation (which the caption lowercases and BERT splits), and one matching nothing.
PROMPTS = [
    ["person"],
    ["person", "laptop", "cup"],
    ["a yellow school bus", "a person holding a mug"],
    ["Coffee-Mug", "USB cable", "the man's hand"],
    ["dinosaur"],
]

#: Phrases for the tokenizer fuzz, beyond the random ones it generates.
_TOKENIZER_CASES = [
    "Café", "naïve façade", "東京タワー", "Ελληνικά", "русский", "عربى", "emoji 🙂 face",
    "tab\there", "ﬁ ligature", "ǅ titlecase", "İstanbul", "ß straße", "a/b-c_d", "x" * 120,
    "́accent", "[MASK] token", "[unused1]", "  spaced   out  ", "Ⅻ roman", "½ half",
]

#: Perturbations for ``--falsify``: what each breaks, and so the first stage that must fail.
#: Each touches mozo's side only -- the reference shares torch, so patching a torch function or a
#: global flag for both would move both sides and prove nothing.
FALSIFIERS = {
    "swin-norm-eps": "Swin LayerNorm eps 1e-5 -> 1e-6 (swin onwards; bert and text pass)",
    "temperature": "image position temperature 20 -> 21 (memory onwards; swin, bert, text pass)",
    "no-score-scale": "drop the contrastive 1/sqrt(256) (logits; in fp32 query selection ranks "
                      "by the same score up to a monotone map, so the decoder passes; in fp16 "
                      "the rounding reorders it and the decoder fails too)",
    "shared-box-head": "every decoder layer uses layer 0's box head, as transformers 5 loads it "
                       "(references onwards)",
    "bert-eager": "BERT softmax by hand instead of SDPA (bert onwards)",
    "resize-rounding": "Grounding DINO's 167x1336 resize rounding (pixels, on the panorama only)",
    # Hardware arithmetic only, CUDA only: the same code, another kernel.
    "tf32-on": "matmul TF32 on around mozo's forward (CUDA fp32: bert onwards)",
    "cudnn-tf32-off": "cuDNN TF32 off around mozo's forward (CUDA fp32: memory onwards -- the "
                      "convolutions that pick a TF32 kernel are the neck's, not Swin's)",
    "sdpa-math": "SDPA forced to the math backend around mozo's forward (CUDA: bert onwards)",
    "channels-last": "every convolution's input channels-last (CUDA: swin onwards)",
    "fp16-full-reduction": "fp16 GEMMs without reduced-precision reduction (CUDA fp16)",
    # The autocast scope: what the fp16 gate proves is that both sides cast the same ops.
    "bert-outside-autocast": "BERT tower run in fp32 inside the fp16 forward (CUDA fp16: bert "
                             "onwards)",
}

#: Falsifiers that only exist on CUDA, and the precisions where what they change runs.
_CUDA_ONLY = {
    "tf32-on": {"fp32"}, "cudnn-tf32-off": {"fp32"}, "sdpa-math": {"fp32", "fp16"},
    "channels-last": {"fp32", "fp16"}, "fp16-full-reduction": {"fp16"},
    "bert-outside-autocast": {"fp16"},
}


@contextlib.contextmanager
def _flag(owner, attribute: str, value):
    saved = getattr(owner, attribute)
    setattr(owner, attribute, value)
    try:
        yield
    finally:
        setattr(owner, attribute, saved)


@contextlib.contextmanager
def falsified(name: str | None):
    """Patch mozo's side for one perturbation.

    Yields ``(mutate, around)``: a function to apply to the vendor's model, and a context manager
    factory entered around mozo's forward only (never the reference's).
    """
    import math

    from mozo.vendors.mm_grounding_dino_deploy import image, network, transformer
    from mozo.vendors.mm_grounding_dino_deploy.text import bert

    undo = []

    def patch(owner, attribute, value):
        undo.append((owner, attribute, getattr(owner, attribute)))
        setattr(owner, attribute, value)

    def mutate(model):
        return model

    around = contextlib.nullcontext

    if name is None:
        pass
    elif name == "swin-norm-eps":
        def mutate(model):  # noqa: F811
            for module in model.backbone.modules():
                if isinstance(module, torch.nn.LayerNorm):
                    module.eps = 1e-6
            return model
    elif name == "temperature":
        position = network.image_position
        patch(network, "image_position", lambda valid, feats, temperature:
              position(valid, feats, temperature + 1))
    elif name == "no-score-scale":
        def forward(self, vision, text, text_valid):
            scores = vision @ text.transpose(-1, -2) + self.bias
            scores = scores.masked_fill(~text_valid[:, None, :], float("-inf"))
            padded = torch.full((*scores.shape[:-1], self.max_text_len), float("-inf"),
                                device=scores.device)
            padded[..., : scores.shape[-1]] = scores
            return padded
        patch(transformer.ContrastiveEmbedding, "forward", forward)
    elif name == "shared-box-head":
        def mutate(model):  # noqa: F811
            for head in model.bbox_embed[1:]:
                head.load_state_dict(model.bbox_embed[0].state_dict())
            return model
    elif name == "bert-eager":
        def forward(self, hidden, mask):
            batch, tokens, _ = hidden.shape
            q, k, v = (self._heads(f(hidden)) for f in (self.query, self.key, self.value))
            scores = q @ k.transpose(-1, -2) / math.sqrt(self.attention_head_size) + mask
            context = (scores.softmax(-1) @ v).transpose(1, 2)
            return context.reshape(batch, tokens, -1)
        patch(bert._SelfAttention, "forward", forward)
    elif name == "resize-rounding":
        def rounded_first(height, width, short_side, max_side):
            small, large = float(min(height, width)), float(max(height, width))
            if large / small * short_side > max_side:
                short_side = int(round(max_side * small / large))
            if (width <= height and width == short_side) or (
                    height <= width and height == short_side):
                return height, width
            if width < height:
                return int(short_side * height / width), short_side
            return short_side, int(short_side * width / height)
        patch(image, "resized_size", rounded_first)
    elif name == "tf32-on":
        def around():  # noqa: F811
            return _flag(torch.backends.cuda.matmul, "allow_tf32", True)
    elif name == "cudnn-tf32-off":
        def around():  # noqa: F811
            return _flag(torch.backends.cudnn, "allow_tf32", False)
    elif name == "sdpa-math":
        def around():  # noqa: F811
            from torch.nn.attention import SDPBackend, sdpa_kernel
            return sdpa_kernel(SDPBackend.MATH)
    elif name == "fp16-full-reduction":
        def around():  # noqa: F811
            return _flag(torch.backends.cuda.matmul,
                         "allow_fp16_reduced_precision_reduction", False)
    elif name == "bert-outside-autocast":
        def mutate(model):  # noqa: F811
            forward = model.text_backbone.forward

            def fp32(*args):
                with torch.autocast("cuda", enabled=False):
                    return forward(*args)
            model.text_backbone.forward = fp32
            return model
    elif name == "channels-last":
        def mutate(model):  # noqa: F811
            def last(_module, args):
                return (args[0].contiguous(memory_format=torch.channels_last),) + args[1:]
            for module in model.modules():
                if isinstance(module, torch.nn.Conv2d):
                    module.register_forward_pre_hook(last)
            return model
    else:
        raise SystemExit(f"unknown falsifier {name!r}. Known: {sorted(FALSIFIERS)}")
    try:
        yield mutate, around
    finally:
        for owner, attribute, value in reversed(undo):
            setattr(owner, attribute, value)


# --- environment --------------------------------------------------------------------------------

def settings() -> dict[str, object]:
    """The process-wide switches that change GPU arithmetic, as they stand."""
    out: dict[str, object] = {
        "cudnn.benchmark": torch.backends.cudnn.benchmark,
        "cudnn.deterministic": torch.backends.cudnn.deterministic,
        "cudnn.allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "matmul.fp16_reduced_precision_reduction":
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    for name in ("flash", "mem_efficient", "math", "cudnn"):
        out[f"sdp.{name}"] = getattr(torch.backends.cuda, f"{name}_sdp_enabled")()
    return out


def environment(device: str) -> str:
    line = f"torch {torch.__version__}"
    if device == "cuda":
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                                 "--format=csv,noheader", "--id=0"],
                                capture_output=True, text=True).stdout.strip()
        line += (f", CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}, "
                 f"{torch.cuda.get_device_name(0)}, driver {driver or 'unknown'}")
    else:
        line += f", MKL_CBWR={os.environ.get('MKL_CBWR', 'unset')}"
    return line


# --- inputs -------------------------------------------------------------------------------------

def images() -> dict:
    """Every fixture photograph, plus crops that reach the awkward geometry.

    A crop wider than 1333/800 makes the long-side cap bind (where this processor's rounding
    differs from Grounding DINO's), a narrow one makes the image portrait, and a tiny one is
    upscaled and padded inside Swin at every stage.
    """
    out = {}
    for path in fixtures():
        image = load_image(str(path))
        height, width = image.shape[:2]
        out[path.name] = image
        out[f"{path.name}[panorama]"] = image[: max(1, width // 8), :]
        out[f"{path.name}[portrait]"] = image[:, : max(1, width // 9)]
        out[f"{path.name}[tiny]"] = image[:37, :45]
    return out


def cases() -> list[tuple[str, object, list[str]]]:
    """Every (label, image, prompts) the reference comparison runs."""
    return [(f"{name} {prompts}", image, prompts)
            for name, image in images().items() for prompts in PROMPTS]


# --- reference ----------------------------------------------------------------------------------

def require_reference():
    """Import ``transformers`` and refuse a version that runs a different model."""
    import transformers

    major = int(transformers.__version__.split(".")[0])
    if major >= 5:
        raise SystemExit(
            f"transformers {transformers.__version__} ties these checkpoints' six decoder heads to "
            "the first on load, so it is not the reference. Use 4.55-4.57 (see the docstring).")
    return transformers


def deformable_path(model) -> str:
    """Which deformable-attention implementation the reference runs; refuses a compiled kernel."""
    from transformers.integrations import hub_kernels
    from transformers.models.mm_grounding_dino import modeling_mm_grounding_dino as upstream

    swapped = [name for name, module in model.named_modules()
               if isinstance(module, upstream.MultiScaleDeformableAttention)
               and type(module).forward is not upstream.MultiScaleDeformableAttention.forward]
    if getattr(model, "use_kernels", False) or swapped:
        raise SystemExit("the reference runs transformers' hub deformable-attention kernel, not "
                         "the grid_sample path mozo carries; load it without use_kernels")
    return (f"grid_sample (pure torch); hub kernel not used: kernels package "
            f"{'installed' if hub_kernels._kernels_available else 'not installed'}, "
            f"use_kernels={getattr(model, 'use_kernels', False)}")


def reference(variant: str, device: str):
    """The upstream model and processor at the pinned commit, parameters re-allocated."""
    transformers = require_reference()
    # The commit tools/fetch publishes, so the two cannot drift apart.
    repo, commit = CHECKPOINTS[variant].repo, CHECKPOINTS[variant].commit
    model = transformers.MMGroundingDinoForObjectDetection.from_pretrained(
        repo, revision=commit).eval()
    # ``from_pretrained`` leaves tensors at the checkpoint's offsets; BLAS picks vector paths by
    # alignment, so re-allocate before comparing arithmetic (VENDORING section 4).
    for parameter in model.parameters():
        parameter.data = parameter.data.clone()
    model = model.to(device)
    processor = transformers.AutoProcessor.from_pretrained(repo, revision=commit)
    if type(processor.image_processor).__name__ != "GroundingDinoImageProcessor":
        raise SystemExit(f"expected the slow image processor, got {type(processor.image_processor)}")
    return model, processor


def tap(module, store: dict, name: str, read=lambda output: output):
    """Record one module's output under *name* every time it runs."""
    return module.register_forward_hook(
        lambda _m, _i, output: store.__setitem__(name, read(output)))


def taps(ours, theirs, mine: dict, upstream: dict) -> None:
    """Hook the same stages on both sides, from one function so they cannot drift apart.

    *theirs* is None when the reference's records come from the cache.
    """
    first, second = (lambda o: o[0]), (lambda o: o[1])
    stages = [
        # (stage, mozo's module, how to read it, upstream's module, how to read it)
        ("bert", ours.text_backbone, None, "model.text_backbone", lambda o: o.last_hidden_state),
        ("text", ours.text_projection, None, "model.text_projection", None),
        ("swin", ours.backbone, None, "model.backbone.conv_encoder.model",
         lambda o: list(o.feature_maps)),
        ("memory", ours.encoder, first, "model.encoder", first),
        ("memory_text", ours.encoder, second, "model.encoder", second),
        ("decoder", ours.decoder, first, "model.decoder",
         lambda o: list(o.intermediate_hidden_states.unbind(1))),
        ("references", ours.decoder, lambda o: o[1][1:], "model.decoder",
         lambda o: list(o.intermediate_reference_points.unbind(1))),
    ]
    for stage, module, read, path, upstream_read in stages:
        tap(module, mine, stage, read or (lambda o: o))
        if theirs is not None:
            tap(theirs.get_submodule(path), upstream, stage, upstream_read or (lambda o: o))
    # The inputs mozo's Predictor hands its model, and what the model returns.
    ours.register_forward_pre_hook(lambda _m, args: mine.update(pixels=args[0][0], ids=args[1][0]))
    ours.register_forward_hook(lambda _m, _i, out: mine.update(logits=out[0], boxes=out[1]))


def flatten(store: dict) -> dict[str, torch.Tensor]:
    """One tensor per stage name, lists expanded to ``stage[level]``."""
    out = {}
    for stage, value in store.items():
        if isinstance(value, (list, tuple)):
            out.update({f"{stage}[{level}]": t for level, t in enumerate(value)})
        else:
            out[stage] = value
    return {name: t.detach() for name, t in out.items()}


def digest(t: torch.Tensor) -> tuple:
    """A tensor as its shape, dtype and the sha256 of its bytes."""
    flat = t.detach().cpu().contiguous().reshape(-1)
    return (tuple(t.shape), str(t.dtype), hashlib.sha256(flat.view(torch.uint8).numpy()).hexdigest())


def _differs(got, want) -> str | None:
    """None when equal to the bit; otherwise what differs. *want* may be a cached digest."""
    if isinstance(want, tuple):
        mine = digest(got)
        return None if mine == want else (
            f"shape/dtype {mine[:2]} vs {want[:2]}" if mine[:2] != want[:2] else "bytes differ")
    got, want = got.cpu(), want.cpu()
    if got.shape != want.shape:
        return f"shape {tuple(got.shape)} vs {tuple(want.shape)}"
    if got.dtype != want.dtype:
        return f"dtype {got.dtype} vs {want.dtype}"
    if torch.equal(got, want):
        return None
    finite = want.isfinite() & got.isfinite()
    if not torch.equal(got.isfinite(), want.isfinite()):
        return "different non-finite entries"
    return f"{(got[finite].double() - want[finite].double()).abs().max().item():.3e}"


def against_tokenizer(processor, seed: int = 0, count: int = 20000) -> tuple[int, list[str]]:
    """The vendor's caption and WordPiece against the processor's, on hand-picked and random text.

    The processor lowercases with Python's ``str.lower`` and then its Rust normaliser lowercases,
    strips accents and drops control characters by its own Unicode tables; the vendor reads those
    tables from an asset derived from it. Fuzzed because a table built one codepoint at a time
    could still miss what happens between neighbours.
    """
    tokenizer = Tokenizer()
    rng = random.Random(seed)
    # Half the draws from the scripts people write prompts in, half from anywhere at all. Never
    # ``.`` or ``?``: a prompt carrying one is refused before it is tokenized.
    common = [chr(c) for c in range(32, 0x3000)]
    anywhere = [chr(c) for c in range(32, 0x110000) if not 0xD800 <= c <= 0xDFFF]

    def draw(alphabet):
        return "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 12))).replace(
            ".", "").replace("?", "") or "x"

    cases = [[case] for case in _TOKENIZER_CASES]
    for index in range(count):
        alphabet = common if index % 2 else anywhere
        cases.append([draw(alphabet) for _ in range(rng.randint(1, 3))])
    failures = []
    for prompts in cases:
        if any(not p.strip() for p in prompts):
            continue
        want = processor.tokenizer(processor._preprocess_input_text([prompts])[0])["input_ids"]
        got, _, _ = tokenizer.encode(caption_for(prompts))
        if got != want:
            failures.append(f"tokenizer {prompts!r}: {got[:12]} vs {want[:12]}")
    return len(cases), failures


def _precision(device: str, dtype: str):
    half = PRECISIONS[dtype]
    return torch.autocast("cuda", half) if half is not None else contextlib.nullcontext()


def run_upstream(theirs, processor, upstream: dict, image, prompts, device: str, dtype: str,
                 threshold: float) -> tuple[dict, list]:
    """One reference forward and post-processing, as a transformers user on *device* runs it."""
    from PIL import Image

    upstream.clear()
    inputs = processor(images=Image.fromarray(image), text=[prompts], return_tensors="pt")
    on_device = inputs.to(device)
    with torch.no_grad(), _precision(device, dtype):
        want = theirs(**on_device)
    height, width = image.shape[:2]
    post = processor.post_process_grounded_object_detection(
        want, on_device["input_ids"], threshold=threshold, target_sizes=[(height, width)])[0]
    records = {"pixels": inputs["pixel_values"][0], "ids": inputs["input_ids"][0],
               **flatten(upstream), "logits": want.logits, "boxes": want.pred_boxes}
    rows = list(zip(post["scores"].tolist(), post["boxes"].tolist()))
    return records, rows


def run_mozo(vendor, mine: dict, image, prompts, around) -> tuple[dict, list]:
    """One call of the vendor Predictor -- mozo's production path -- with its stages recorded."""
    mine.clear()
    with around():
        kept = vendor(image, prompts)
    return flatten(mine), [(d.score, list(d.box)) for d in kept]


def compare(label: str, mine: dict, rows: list, want: dict, want_rows: list
            ) -> tuple[int, list[str]]:
    """Every recorded stage to the bit, then the detections against upstream's post-processing."""
    checked, failures = 0, []
    for stage, expected in want.items():
        checked += 1
        if stage not in mine:
            failures.append(f"{label}: {stage} not recorded on mozo's side")
        elif (problem := _differs(mine[stage], expected)) is not None:
            failures.append(f"{label}: {stage} {problem}")
    # The vendor drops a query whose peak token is a separator, so a count mismatch alone is
    # not a failure: every vendor detection must be one of upstream's rows, to the bit.
    checked += 1
    upstream_rows = [(score, list(box)) for score, box in want_rows]
    if not all(row in upstream_rows for row in rows):
        failures.append(f"{label}: detections differ from post-processing")
    return checked, failures


class References:
    """Upstream's records for one (variant, device, precision): from ``--reference-cache`` or live.

    A cached record keeps each stage as :func:`digest` and the detections whole.
    """

    def __init__(self, directory: Path | None, variant: str, device: str, dtype: str, meta: dict):
        self.path = directory / f"{variant}-{device}-{dtype}.pt" if directory else None
        self.meta = meta
        self.records: dict[str, tuple[dict, list]] = {}
        self.cached = bool(self.path and self.path.exists())
        if self.cached:
            saved = torch.load(self.path, weights_only=False)
            if saved["meta"] != meta:
                raise SystemExit(f"{self.path} was made by {saved['meta']}, not {meta}. Delete it, "
                                 "or use another --reference-cache.")
            self.records = saved["records"]

    def put(self, label: str, records: dict, rows: list) -> None:
        self.records[label] = ({k: digest(v) for k, v in records.items()}, rows)

    def save(self) -> None:
        if self.path and not self.cached:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"meta": self.meta, "records": self.records}, self.path)


def against_reference(variant: str, checkpoint: Path, falsify: str | None, device: str,
                      dtype: str, cache: Path | None) -> tuple[int, list[str]]:
    """Compare every stage of the vendored model, and its detections, against the reference."""
    import transformers

    spec = SPECS[variant]
    before = settings()
    meta = {"transformers": transformers.__version__, "torch": torch.__version__,
            "environment": environment(device), "settings": before,
            "cases": [label for label, _, _ in cases()],
            "fixtures": [hashlib.sha256(image.tobytes()).hexdigest()
                         for _, image, _ in cases()]}
    refs = References(cache, variant, device, dtype, meta)
    theirs = processor = None
    checked, failures = 0, []
    if not refs.cached:
        theirs, processor = reference(variant, device)
        print(f"  deformable attention: {deformable_path(theirs)}")
        if device == "cpu":
            checked, failures = against_tokenizer(processor)
    else:
        print(f"  reference from {refs.path}")
    if settings() != before:
        raise SystemExit(f"loading the reference changed global settings: {before} -> {settings()}")

    with falsified(falsify) as (mutate, around):
        vendor = Predictor(checkpoint, spec, device=device, autocast=PRECISIONS[dtype] is not None)
        vendor.model = mutate(vendor.model)
        mine: dict = {}
        upstream: dict = {}
        taps(vendor.model, theirs, mine, upstream)

        repeat = falsify is None and device == "cuda"
        for label, image, prompts in cases():
            if refs.cached:
                want, want_rows = refs.records[label]
            else:
                want, want_rows = run_upstream(theirs, processor, upstream, image, prompts,
                                               device, dtype, spec.box_threshold)
                refs.put(label, want, want_rows)
                if repeat:
                    again, again_rows = run_upstream(theirs, processor, upstream, image, prompts,
                                                     device, dtype, spec.box_threshold)
                    checked += 1
                    if refs.records[label] != ({k: digest(v) for k, v in again.items()},
                                               again_rows):
                        failures.append(f"{label}: upstream not deterministic")
            got, rows = run_mozo(vendor, mine, image, prompts, around)
            if repeat:
                again, again_rows = run_mozo(vendor, mine, image, prompts, around)
                checked += 1
                if ({k: digest(v) for k, v in got.items()}, rows) != (
                        {k: digest(v) for k, v in again.items()}, again_rows):
                    failures.append(f"{label}: mozo not deterministic")
            n, problems = compare(label, got, rows, want, want_rows)
            checked += n
            failures.extend(problems)
    if settings() != before:
        raise SystemExit(f"the run changed global settings: {before} -> {settings()}")
    refs.save()
    return checked, failures


def kernels(variant: str, checkpoint: Path, dtype: str) -> None:
    """Print the CUDA kernels one case runs on each side: which GEMM, convolution, attention and
    sampling kernels production actually uses, so a falsifier can be checked against them."""
    from torch.profiler import ProfilerActivity, profile

    spec = SPECS[variant]
    theirs, processor = reference(variant, "cuda")
    vendor = Predictor(checkpoint, spec, device="cuda", autocast=PRECISIONS[dtype] is not None)
    label, image, prompts = cases()[1]

    def names(fn) -> collections.Counter:
        fn()  # warm up: first calls pick algorithms
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        return collections.Counter(e.name for e in prof.events()
                                   if e.device_type == torch.autograd.DeviceType.CUDA)

    print(f"kernels, {variant} {dtype}, {label}")
    for side, fn in (("upstream", lambda: run_upstream(theirs, processor, {}, image, prompts,
                                                       "cuda", dtype, spec.box_threshold)),
                     ("mozo", lambda: vendor(image, prompts))):
        counted = names(fn)
        print(f"  {side}: {sum(counted.values())} launches, {len(counted)} kernels")
        for name, count in sorted(counted.items()):
            if any(k in name.lower() for k in ("gemm", "conv", "fmha", "attention", "sdp",
                                               "grid_sampler", "xmma", "cutlass", "softmax")):
                print(f"    {count:5d}  {name[:150]}")


def kernel_change(variant: str, checkpoint: Path, dtype: str, falsify: str) -> str:
    """Which CUDA kernels a hardware falsifier changes on mozo's side, on one case."""
    from torch.profiler import ProfilerActivity, profile

    _, image, prompts = cases()[1]

    found = []
    for name in (None, falsify):
        with falsified(name) as (mutate, around):
            vendor = Predictor(checkpoint, SPECS[variant], device="cuda",
                               autocast=PRECISIONS[dtype] is not None)
            vendor.model = mutate(vendor.model)
            with around():
                vendor(image, prompts)
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    vendor(image, prompts)
                    torch.cuda.synchronize()
            found.append({e.name for e in prof.events()
                          if e.device_type == torch.autograd.DeviceType.CUDA})
            del vendor
    gone, new = found[0] - found[1], found[1] - found[0]
    show = lambda s: "; ".join(sorted(n[:90] for n in s)[:4]) or "none"  # noqa: E731
    return f"{len(gone)} kernels replaced by {len(new)}: [{show(gone)}] -> [{show(new)}]"


def against_mozo(variant: str, checkpoint: Path, device: str, dtype: str
                 ) -> tuple[int, list[str]]:
    """The adapter sorts, converts to PixelFlow and rounds. It must change nothing else."""
    from mozo.adapters.mm_grounding_dino import MMGroundingDinoPredictor

    half = PRECISIONS[dtype] is not None
    vendor = Predictor(checkpoint, SPECS[variant], device=device, autocast=half)
    if device == "cpu":
        adapter = MMGroundingDinoPredictor(variant, device="cpu", checkpoint_path=checkpoint)
    else:
        # Through the registry and manifest. fp16 is what ``auto`` picks on CUDA; checked, so a
        # registry change that moves ``auto`` off it shows up here.
        adapter = MMGroundingDinoPredictor(variant, device=device,
                                           runtime="auto" if half else "torch-fp32")
        if adapter.runtime != ("torch-fp16" if half else "torch-fp32"):
            return 1, [f"the adapter runs {adapter.runtime} on {device}, not the {dtype} gated"]

    checked, failures = 0, []
    for name, image in images().items():
        for prompts in PROMPTS:
            raw = sorted(vendor(image, prompts), key=lambda d: -d.score)
            got = adapter.predict(image, prompts).to_dict()
            checked += 1
            if len(raw) != len(got):
                failures.append(f"{name} {prompts}: {len(raw)} from vendor, {len(got)} from mozo")
                continue
            want_boxes, want_scores = as_pixelflow_reports(
                [d.box for d in raw], [d.score for d in raw], [d.prompt_index for d in raw])
            if [list(b) for b in want_boxes] != [row["bbox"] for row in got]:
                failures.append(f"{name} {prompts}: boxes differ after rounding")
            elif list(want_scores) != [row["confidence"] for row in got]:
                failures.append(f"{name} {prompts}: scores differ after rounding")
            elif [prompts[d.prompt_index] for d in raw] != [row["class_name"] for row in got]:
                failures.append(f"{name} {prompts}: names differ")
    return checked, failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variants", nargs="*", default=["llmdet-tiny"],
                        help=f"variants to check (default: llmdet-tiny; known: {list(SPECS)})")
    parser.add_argument("--reference", action="store_true",
                        help="also compare against transformers 4.x (see the docstring)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dtype", choices=sorted(PRECISIONS),
                        help="precision (default: fp32 on CPU; fp32 and fp16 on CUDA)")
    parser.add_argument("--reference-cache", type=Path,
                        help="keep upstream's records here and reuse them (see the docstring)")
    parser.add_argument("--falsify", choices=sorted(FALSIFIERS),
                        help="perturb mozo's side and expect the reference comparison to fail")
    parser.add_argument("--kernels", action="store_true",
                        help="print the CUDA kernels one case runs on each side, then stop")
    args = parser.parse_args()
    if args.falsify and not args.reference:
        parser.error("--falsify needs --reference")
    if args.device == "cpu" and args.dtype not in (None, "fp32"):
        parser.error("half precision is a CUDA runtime; on the CPU only fp32 is gated")
    if args.device == "cpu" and (args.kernels or args.falsify in _CUDA_ONLY):
        parser.error("that option changes or reads CUDA kernels; it needs --device cuda")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda, but torch sees no CUDA device")
    dtypes = [args.dtype] if args.dtype else (["fp32", "fp16"] if args.device == "cuda"
                                              else ["fp32"])

    print(environment(args.device))
    if args.device == "cuda":
        print(f"settings: {settings()}")
    if args.reference or args.kernels:
        print(f"transformers {require_reference().__version__}")
    total, problems = 0, []
    for variant in args.variants:
        try:
            checkpoint = resolve(FAMILY, variant, "torch-fp32")
        except WeightsError as error:
            print(f"{variant}: {error}")
            return 1
        for dtype in dtypes:
            if args.kernels:
                kernels(variant, checkpoint, dtype)
                continue
            if args.falsify in _CUDA_ONLY:
                if dtype not in _CUDA_ONLY[args.falsify]:
                    print(f"{variant} {dtype}: {args.falsify} does not apply (what it changes "
                          "does not run in this precision)")
                    continue
                print(f"  {args.falsify}: "
                      f"{kernel_change(variant, checkpoint, dtype, args.falsify)}")
            where = f"{variant} {args.device} {dtype}"
            if not args.falsify:
                checked, failures = against_mozo(variant, checkpoint, args.device, dtype)
                total += checked
                problems.extend(f"[mozo/{where}] {f}" for f in failures)
                print(f"{where}: {checked} mozo-vs-vendor comparisons, {len(failures)} failed")

            if args.reference:
                checked, failures = against_reference(variant, checkpoint, args.falsify,
                                                      args.device, dtype, args.reference_cache)
                total += checked
                problems.extend(f"[reference/{where}] {f}" for f in failures)
                print(f"{where}: {checked} reference comparisons, {len(failures)} failed")
            if args.device == "cuda":
                torch.cuda.empty_cache()
    if args.kernels:
        return 0

    if args.falsify:
        print(f"\nfalsified with {args.falsify}: {FALSIFIERS[args.falsify]}")
        first = problems[0] if problems else "nothing"
        print(f"{len(problems)} failures; first: {first}")
        return 0 if problems else 1

    if problems:
        print(f"\n{len(problems)} FAILURES:")
        for problem in problems[:40]:
            print(f"  {problem}")
        return 1

    print(f"\n{total} comparisons, all identical.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

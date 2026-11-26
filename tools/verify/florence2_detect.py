#!/usr/bin/env python3
"""Check that mozo's Florence-2 decodes exactly what Microsoft's code does, step by step.

Three paths reach the same weights and this compares all three: the vendored package
(``mozo/vendors/florence2_deploy``), the reference (Microsoft's remote code in the Hub repository,
``AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`` + its processor + ``generate``
with no decoding arguments), and mozo itself (registry, weights, ``florence2_detect`` adapter).

**The reference is Microsoft's remote code, under transformers 4.53.3.** Microsoft publishes
Florence-2 only as Hub repositories carrying their own ``modeling_florence2.py``; there is no GitHub
implementation. That code runs on transformers up to 4.53.3 (4.54 onwards fail inside it), and needs
``attn_implementation="eager"``. ``transformers``' native ``Florence2ForConditionalGeneration`` is a
port onto converted weights, not the reference. The post-processor (with ``<OD>``'s scores) is read
from the ``Florence-2-base`` repository for every variant: the ``-ft`` repositories carry an older
copy that predates the scores.

**Every stage, then every step.** Pixels, prompt ids, each parameter's ``requires_grad``, each
DaViT stage, the projected image tokens, the encoder's input and each encoder layer -- then every
decoding step's processed log-probabilities for all three beams, the number of steps, the sequence,
the beam indices, the transition scores, and the parsed boxes, labels and scores; and for mozo's
served path, the exact arrays the adapter hands ``pf.detections.from_arrays``, then the
Detections through PixelFlow's door. ``torch.equal`` and ``==``, no tolerance.

**On the CPU** (the default): float32 (the published fp16 weights upcast at load on both sides),
eager attention, ``MKL_CBWR=AVX2,STRICT`` expected in the environment.

**On CUDA** (``--device cuda``): both sides in this process, on the same GPU, with the global
settings mozo's runtime leaves (it sets none: TF32 on for cuDNN, off for matmul; every SDPA
backend enabled; cuDNN benchmark and deterministic off), checked unchanged after the reference is
loaded. ``--dtype`` picks the precision; omitted, CUDA checks ``fp32`` and ``fp16``, the
``torch-fp16`` runtime ``auto`` picks there. **fp16 is the whole model in half precision, upstream's
own mechanism**: the model card's CUDA path is ``from_pretrained(..., torch_dtype=torch.float16)``
with the processor's pixels cast ``.to(device, torch.float16)``, and mozo's ``torch-fp16`` loads the
published fp16 tensors as they are and casts the pixels the same way. No autocast on either side.
The reference runs ``generate`` under ``no_grad`` (see :meth:`Reference.run`), mozo under
``inference_mode``, as each does in production.

In both cases the reference's parameters are re-allocated before comparing, and every case runs
``--repeat`` times (default 2) on each side, each run byte-identical to the first (determinism).

**The gate is shown to fail.** ``--falsify NAME`` (or ``all``) perturbs one thing on mozo's side
and requires the comparison to fail at the stage that thing reaches first. ``--reference-cache DIR``
keeps the reference's records per (variant, device, precision, case) as each tensor's shape, dtype
and sha256, so falsifiers and reruns reuse one reference run (delete the directory for a fresh one).

    python tools/verify/florence2_detect.py --reference base=DIR          # base, every fixture
    python tools/verify/florence2_detect.py --reference base=DIR large=DIR2 --quick
    python tools/verify/florence2_detect.py --reference base=DIR --falsify all
    python tools/verify/florence2_detect.py --device cuda --reference base=DIR --reference-cache C

``--reference VARIANT=DIR`` names a local snapshot of the variant's Hub repository at the pinned
commit (``hf download <repo> --revision <commit> --local-dir DIR``); ``--processor DIR`` the
``Florence-2-base`` snapshot whose post-processor is used (default: the ``base`` reference).
The vendor and mozo read the published ``torch-fp16.pth`` from ``weights/``.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import io
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault("MOZO_BASE_URL", f"file://{ROOT / 'weights'}")

from fetch.florence2_detect import published  # noqa: E402

from mozo.image import load_image  # noqa: E402
from mozo.vendors.florence2_deploy import Florence2Predictor  # noqa: E402
from mozo.vendors.florence2_deploy import decoding as vdecoding  # noqa: E402
from mozo.vendors.florence2_deploy import image as vimage  # noqa: E402
from mozo.vendors.florence2_deploy import language as vlanguage  # noqa: E402
from mozo.vendors.florence2_deploy import network as vnetwork  # noqa: E402
from mozo.vendors.florence2_deploy import parse as vparse  # noqa: E402
from mozo.vendors.florence2_deploy import vision as vvision  # noqa: E402
from mozo.vendors.florence2_deploy.tokenizer import PROMPTS  # noqa: E402

PINNED = {"torch": "2.13.0", "transformers": "4.53.3", "tokenizers": "0.21.4"}
FIXTURES = ROOT / "tests" / "fixtures"

#: Each ``--dtype``, as both sides load the model: mozo's ``torch-fp32`` / ``torch-fp16``.
PRECISIONS = {"fp32": torch.float32, "fp16": torch.float16}


def versions() -> str:
    import tokenizers
    import transformers
    got = {"torch": torch.__version__, "transformers": transformers.__version__,
           "tokenizers": tokenizers.__version__}
    return ", ".join(f"{k} {v}" + ("" if v.split("+")[0] == PINNED[k] else f" (pinned {PINNED[k]})")
                     for k, v in got.items())


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
    line = f"versions: {versions()}"
    if device == "cuda":
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                                 "--format=csv,noheader", "--id=0"],
                                capture_output=True, text=True).stdout.strip()
        line += (f"; CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}, "
                 f"{torch.cuda.get_device_name(0)}, driver {driver or 'unknown'}")
    else:
        line += f"; MKL_CBWR={os.environ.get('MKL_CBWR', 'unset')}; threads {torch.get_num_threads()}"
    return line


@dataclass(frozen=True, eq=False)
class Case:
    """One fixture: an image, a task, a length limit. ``also_large`` marks the larger variants'."""

    name: str
    rgb: np.ndarray
    task: str = "<OD>"
    max_new_tokens: int = 1024
    also_large: bool = False

    def key(self) -> str:
        """What identifies this case's reference records in the cache."""
        digest = hashlib.sha256(np.ascontiguousarray(self.rgb).tobytes()).hexdigest()
        return hashlib.sha256(f"{self.name}|{self.task}|{self.max_new_tokens}|{digest}"
                              .encode()).hexdigest()[:16]


def cases(quick: bool) -> list[Case]:
    """The fixtures, awkward ones included (VENDORING §2.6)."""
    from PIL import Image

    photo = load_image(str(FIXTURES / "images" / "example.jpg"))
    small = np.array(Image.fromarray(photo).resize((480, 320), Image.BICUBIC))
    if quick:
        return [Case("photo", photo, also_large=True)]
    rgba, grey = io.BytesIO(), io.BytesIO()
    Image.fromarray(small).convert("RGBA").save(rgba, "PNG")
    Image.fromarray(small).convert("L").save(grey, "JPEG")
    text = {name: load_image(str(FIXTURES / "text" / f"{name}.png"))
            for name in ("printed", "rotated", "cjk", "vertical", "blank")}
    return [
        Case("photo", photo, also_large=True),
        Case("photo, cut at 24 tokens (forced </s>)", photo, max_new_tokens=24, also_large=True),
        Case("small photo", small),
        Case("tall strip 120x1281 (aspect not kept)", np.ascontiguousarray(photo[:, 600:720])),
        Case("blank white", np.full((300, 400, 3), 255, np.uint8)),
        Case("RGBA PNG via load_image", load_image(rgba.getvalue())),
        Case("greyscale JPEG via load_image", load_image(grey.getvalue())),
        Case("text: printed, <OD>", text["printed"]),
        Case("OCR regions: printed", text["printed"], "<OCR_WITH_REGION>", also_large=True),
        Case("OCR regions: rotated", text["rotated"], "<OCR_WITH_REGION>"),
        Case("OCR regions: CJK", text["cjk"], "<OCR_WITH_REGION>"),
        Case("OCR regions: vertical", text["vertical"], "<OCR_WITH_REGION>"),
        Case("OCR regions: blank", text["blank"], "<OCR_WITH_REGION>"),
    ]


def _parsed(task: str, parsed: dict) -> dict:
    """The parsed answer in one shape for both sides."""
    if task == "<OD>":
        return {"boxes": parsed["bboxes"], "labels": parsed["labels"],
                "scores": parsed.get("scores", [])}
    return {"quads": parsed["quad_boxes"], "text": parsed["labels"]}


class Reference:
    """Microsoft's code, with hooks on every stage the vendor reports."""

    def __init__(self, model_dir: str, processor_dir: str, device: str = "cpu",
                 dtype: torch.dtype = torch.float32) -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor

        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, torch_dtype=dtype,
            attn_implementation="eager").eval()
        # Re-allocated, so what is compared is arithmetic, not the checkpoint's memory offsets.
        for parameter in model.parameters():
            parameter.data = parameter.data.clone()
        for module in model.modules():
            for name, buffer in list(module._buffers.items()):
                if buffer is not None:
                    module._buffers[name] = buffer.clone()
        self.model = model.to(device)
        self.device, self.dtype = device, dtype
        self.processor = AutoProcessor.from_pretrained(processor_dir, trust_remote_code=True)
        self.caught: dict[str, torch.Tensor] = {}
        for index, block in enumerate(model.vision_tower.blocks):
            block.register_forward_hook(self._post(f"vision_stage_{index}"))
        encoder = model.language_model.model.encoder
        encoder.register_forward_pre_hook(self._embeds, with_kwargs=True)
        for index, layer in enumerate(encoder.layers):
            layer.register_forward_hook(self._post(f"encoder_layer_{index}"))
        original = model._encode_image

        def encode_image(pixels):
            out = original(pixels)
            self.caught.setdefault("image_features", out.detach().clone())
            return out
        model._encode_image = encode_image

    def _post(self, name):
        def hook(module, args, out):
            self.caught.setdefault(name, (out[0] if isinstance(out, tuple) else out).detach().clone())
        return hook

    def _embeds(self, module, args, kwargs):
        self.caught.setdefault("encoder_input", kwargs["inputs_embeds"].detach().clone())

    @torch.no_grad()
    def run(self, case: Case) -> dict:
        """One ``generate`` as the model card runs it on this device and precision.

        Under ``no_grad``: the remote ``generate`` is not decorated with it (only the language
        model's, transformers', is), so the card's code records autograd through DaViT and the
        projection -- which on CUDA does not fit ``large`` twice in 12 GB. Grad mode picks no
        kernel: base passed every fixture in fp32, and the photograph in fp16, with it on.
        """
        from PIL import Image

        self.caught = {}
        inputs = self.processor(text=case.task, images=Image.fromarray(case.rgb),
                                return_tensors="pt")
        out = self.model.generate(
            input_ids=inputs["input_ids"].to(self.device),
            pixel_values=inputs["pixel_values"].to(self.device, self.dtype),
            max_new_tokens=case.max_new_tokens, return_dict_in_generate=True, output_scores=True)
        sequence = out.sequences[0]
        transition = self.model.compute_transition_scores(
            out.sequences, out.scores, out.beam_indices, normalize_logits=False)[0]
        size = (case.rgb.shape[1], case.rgb.shape[0])
        parsed = self.processor.post_process_generation(
            sequence=sequence, transition_beam_score=transition, task=case.task,
            image_size=size)[case.task]
        return {**self.caught, "pixel_values": inputs["pixel_values"],
                "input_ids": inputs["input_ids"][0].tolist(), "scores": list(out.scores),
                "sequence": sequence, "beam_indices": out.beam_indices[0].to(torch.int64),
                "transition": transition, "parsed": _parsed(case.task, parsed),
                "requires_grad": {name: p.requires_grad
                                  for name, p in self.model.named_parameters()}}


def ours(case: Case, vendor: Florence2Predictor) -> dict:
    """The vendored package's records for *case*, in the reference's shape."""
    hooks: dict[str, torch.Tensor] = {}
    output = vendor.run(case.rgb, case.task, hooks, max_new_tokens=case.max_new_tokens)
    beams = output.beams
    if case.task == "<OD>":
        boxes = vendor.parse_od(output)
        parsed = {"boxes": [b.box for b in boxes], "labels": [b.label for b in boxes],
                  "scores": [b.score for b in boxes]}
    else:
        lines = vendor.parse_ocr(output)
        parsed = {"quads": [line.quad for line in lines], "text": [line.text for line in lines]}
    return {**hooks, "input_ids": list(PROMPTS[case.task]), "scores": list(beams.scores),
            "sequence": beams.sequence, "beam_indices": beams.beam_indices[0].to(torch.int64),
            "transition": vdecoding.transition_scores(beams, vendor.spec.language.vocab),
            "parsed": parsed,
            "requires_grad": {name: p.requires_grad
                              for name, p in vendor.model.named_parameters()}}


def digest(t: torch.Tensor) -> tuple:
    """A tensor as its shape, dtype and the sha256 of its bytes."""
    flat = t.detach().cpu().contiguous().reshape(-1)
    return (tuple(t.shape), str(t.dtype), hashlib.sha256(flat.view(torch.uint8).numpy()).hexdigest())


def digested(record: dict) -> dict:
    """*record* with every tensor replaced by :func:`digest` -- what the cache keeps."""
    out = {}
    for name, value in record.items():
        if isinstance(value, torch.Tensor):
            out[name] = digest(value)
        elif name == "scores":
            out[name] = [digest(t) for t in value]
        else:
            out[name] = value
    return out


def _same(a, b) -> str | None:
    """None when equal to the bit; otherwise what differs. Either side may be a digest."""
    if isinstance(a, tuple) or isinstance(b, tuple):
        a = a if isinstance(a, tuple) else digest(a)
        b = b if isinstance(b, tuple) else digest(b)
        if a == b:
            return None
        return f"shape/dtype {a[:2]} vs {b[:2]}" if a[:2] != b[:2] else "bytes differ"
    a, b = a.detach().cpu(), b.detach().cpu()
    if a.shape != b.shape:
        return f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
    if a.dtype != b.dtype:
        return f"dtype {a.dtype} vs {b.dtype}"
    if torch.equal(a, b):
        return None
    if not a.is_floating_point():
        return f"{int((a != b).sum())} of {a.numel()} differ"
    if not torch.equal(torch.isfinite(a), torch.isfinite(b)):
        return "different -inf positions"
    finite = torch.isfinite(a) & torch.isfinite(b)
    return f"max |d| {(a[finite].double() - b[finite].double()).abs().max().item():.2g}"


def compare(mine: dict, theirs: dict, depth: int) -> tuple[int, list[str]]:
    """Every comparison between two records: how many were made, and the failures in stage order."""
    checks = 0
    failures = []

    def check(name, problem):
        nonlocal checks
        checks += 1
        if problem:
            failures.append(f"{name}: {problem}")

    check("input_ids", None if mine["input_ids"] == theirs["input_ids"] else "prompt ids differ")
    flags, upstream = mine["requires_grad"], theirs["requires_grad"]
    absent = sorted(set(flags) - set(upstream))
    moved = sorted(name for name in set(flags) & set(upstream) if flags[name] != upstream[name])
    check("requires_grad", None if not absent and not moved else
          f"{len(moved)} flags differ ({moved[:2]}), {len(absent)} not upstream ({absent[:2]})")
    stages = (["pixel_values"] + [f"vision_stage_{i}" for i in range(4)]
              + ["image_features", "encoder_input"] + [f"encoder_layer_{i}" for i in range(depth)])
    for name in stages:
        check(name, _same(mine[name], theirs[name]))
    for step, (a, b) in enumerate(zip(mine["scores"], theirs["scores"])):
        check(f"step_{step}", _same(a, b))
    check("steps", None if len(mine["scores"]) == len(theirs["scores"])
          else f"{len(mine['scores'])} vs {len(theirs['scores'])}")
    for name in ("sequence", "beam_indices", "transition"):
        check(name, _same(mine[name], theirs[name]))
    for name, a in mine["parsed"].items():
        b = theirs["parsed"][name]
        check(f"parsed_{name}", None if a == b else f"{a[:2]!r}... vs {b[:2]!r}...")
    return checks, failures


def against_mozo(model, case: Case, want: dict) -> tuple[int, list[str]]:
    """The served path: registry, weights from the mirror, the adapter, PixelFlow.

    First the exact arrays the adapter hands ``pf.detections.from_arrays`` against the reference's
    parsed boxes and scores. Then the Detections: PixelFlow rounds coordinates and scores to its
    own precision, so the reference's parsed boxes go through the same door before comparing.
    """
    import pixelflow as pf

    handed: dict = {}
    original = pf.detections.from_arrays

    def spy(**kwargs):
        handed.update(kwargs)
        return original(**kwargs)
    with mock.patch.object(pf.detections, "from_arrays", spy):
        found = model.predict(case.rgb)
    parsed = want["parsed"]
    boxes = np.array(parsed["boxes"], dtype=float).reshape(-1, 4)
    scores = np.array(parsed["scores"], dtype=float)
    door = pf.detections.from_arrays(boxes=boxes, scores=scores)
    failures = []
    for name, a, b in (
            ("mozo_arrays_boxes", handed["boxes"], boxes),
            ("mozo_arrays_scores", handed["scores"], scores)):
        if a.dtype != b.dtype or not np.array_equal(a, b):
            failures.append(f"{name}: {a[:2]!r}... vs {b[:2]!r}...")
    for name, a, b in (
            ("mozo_boxes", [d.bbox for d in found], [d.bbox for d in door]),
            ("mozo_labels", [d.class_name for d in found], parsed["labels"]),
            ("mozo_scores", [d.confidence for d in found], [d.confidence for d in door])):
        if a != b:
            failures.append(f"{name}: {a[:2]!r}... vs {b[:2]!r}...")
    return 5, failures


class References:
    """The reference's records per case: from ``--reference-cache`` when there, else run.

    The model is loaded only on a cache miss. A run repeats ``repeat`` times and must repeat
    itself to the byte; the cache keeps the first run's digests.
    """

    def __init__(self, variant: str, model_dir: str, processor_dir: str, device: str,
                 precision: str, cache: Path | None, repeat: int, depth: int) -> None:
        self.args = (model_dir, processor_dir, device, PRECISIONS[precision])
        self.tag = f"{variant}-{device}-{precision}"
        self.cache, self.repeat, self.depth = cache, repeat, depth
        self.model: Reference | None = None
        self.determinism = 0

    def get(self, case: Case) -> dict:
        path = self.cache / f"{self.tag}-{case.key()}.pt" if self.cache else None
        if path is not None and path.exists():
            return torch.load(path, weights_only=False)
        if self.model is None:
            before = settings()
            self.model = Reference(*self.args)
            if settings() != before:
                raise SystemExit(f"loading the reference changed global settings: {before} -> "
                                 f"{settings()}")
        first = self.model.run(case)
        for _ in range(self.repeat - 1):
            checks, failures = compare(self.model.run(case), first, self.depth)
            self.determinism += checks
            if failures:
                raise SystemExit(f"the reference is not deterministic on {case.name}: {failures[0]}")
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(digested(first), path)
        return first

    def close(self) -> None:
        self.model = None


def run_ours(case: Case, vendor: Florence2Predictor, repeat: int, depth: int) -> tuple[dict, int]:
    """The vendor's records, run *repeat* times; each must equal the first to the byte."""
    first = ours(case, vendor)
    checks = 0
    for _ in range(repeat - 1):
        more, failures = compare(ours(case, vendor), first, depth)
        checks += more
        if failures:
            raise SystemExit(f"mozo's vendor is not deterministic on {case.name}: {failures[0]}")
    return first, checks


# --- Falsification: each perturbation must be caught at the stage it reaches first. ---
# Each touches mozo's side only: the reference shares torch, so patching a torch function or a
# global flag for both would move both sides and prove nothing. The reference's records are
# produced (or read from the cache) before any perturbation is entered.

def _bilinear_preprocess(image):
    from PIL import Image

    resized = Image.fromarray(image).resize((768, 768), resample=Image.BILINEAR, reducing_gap=None)
    x = (np.array(resized).astype(np.float64) * (1 / 255)).astype(np.float32)
    x = (x - np.array(vimage.MEAN, dtype=np.float32)) / np.array(vimage.STD, dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]


def _reducing_gap(image):
    """PIL's two-step downscale (``reducing_gap=1.0``), not the exact one-step bicubic.

    Only a downscale by 2x or more is affected, so it runs on the full 1920 x 1281 photograph."""
    from PIL import Image

    resized = Image.fromarray(image).resize((768, 768), resample=Image.BICUBIC, reducing_gap=1.0)
    x = (np.array(resized).astype(np.float64) * (1 / 255)).astype(np.float32)
    x = (x - np.array(vimage.MEAN, dtype=np.float32)) / np.array(vimage.STD, dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]


def _channel_scale_by_group():
    """Channel attention scaled by the group width instead of the token count."""
    def forward(self, x, size):
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.groups, channels // self.groups)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        q = q * (float(channels // self.groups) ** -0.5)
        attention = (q.transpose(-1, -2) @ k).softmax(dim=-1)
        x = (attention @ v.transpose(-1, -2)).transpose(-1, -2)
        return self.proj(x.transpose(1, 2).reshape(batch, tokens, channels)), size
    return mock.patch.object(vvision._ChannelAttention, "forward", forward)


def _rows_columns_swapped():
    def forward(self, grid):
        height, width = grid.shape[1:3]
        x = self.row_embeddings(torch.arange(width, device=grid.device))
        y = self.column_embeddings(torch.arange(height, device=grid.device))
        pos = torch.cat([x.unsqueeze(0).repeat(height, 1, 1), y.unsqueeze(1).repeat(1, width, 1)],
                        dim=-1)
        pos = pos.permute(2, 0, 1).unsqueeze(0).repeat(grid.shape[0], 1, 1, 1)
        return pos.permute(0, 2, 3, 1)
    return mock.patch.object(vnetwork._PositionEmbedding2D, "forward", forward)


def _no_temporal():
    return mock.patch.object(vnetwork._TemporalEmbedding, "forward",
                             lambda self, frames: torch.zeros_like(self.pos_idx_to_embed[:1])[None])


def _encoder_offset_zero():
    return mock.patch.object(vlanguage, "_OFFSET", 0)


def _decoder_norm_eps():
    """Every LayerNorm in the decoder at eps 1e-6 (BART's is torch's 1e-5)."""
    original = vlanguage._DecoderLayer.forward

    def forward(self, hidden, cross, past):
        norms = (self.self_attn_layer_norm, self.encoder_attn_layer_norm, self.final_layer_norm)
        saved = [norm.eps for norm in norms]
        for norm in norms:
            norm.eps = 1e-6
        try:
            return original(self, hidden, cross, past)
        finally:
            for norm, eps in zip(norms, saved):
                norm.eps = eps
    return mock.patch.object(vlanguage._DecoderLayer, "forward", forward)


def _no_ngram():
    return mock.patch.object(vdecoding, "_no_repeat_ngram",
                             lambda ids, scores, size: scores.clone())


def _rules(**changes):
    rules = vdecoding.Decoding(**{**vdecoding.DECODING.__dict__, **changes})
    original = vdecoding.beam_search

    def beam_search(language, encoded, max_new_tokens, _rules=None):
        return original(language, encoded, max_new_tokens, rules)
    from mozo.vendors.florence2_deploy import predictor as vpredictor
    return mock.patch.object(vpredictor, "beam_search", beam_search)


def _no_forced_eos():
    """The length limit without ``ForcedEOSTokenLogitsProcessor``."""
    def process(ids, scores, max_length, rules):
        scores = vdecoding._no_repeat_ngram(ids, scores, rules.no_repeat_ngram)
        if ids.shape[-1] == 1:
            scores = torch.full_like(scores, -math.inf)
            scores[:, rules.bos] = 0
        return scores
    return mock.patch.object(vdecoding, "_process", process)


def _label_scores():
    """``<OD>``'s score averaged over tokens one position off its box's four location tokens."""
    original = vparse._overlapping

    def overlapping(span, spans):
        found = original(span, spans)
        return [max(0, found[0] - 1)] + found[:-1]
    return mock.patch.object(vparse, "_overlapping", overlapping)


def _no_half_bin():
    def boxes(bins, size):
        xmin, ymin, xmax, ymax = torch.tensor(bins).split(1, dim=-1)
        per_x, per_y = size[0] / 1000, size[1] / 1000
        return torch.cat((xmin * per_x, ymin * per_y, xmax * per_x, ymax * per_y), dim=-1).tolist()
    return mock.patch.object(vparse, "_boxes", boxes)


# CUDA only: the same code, another kernel or another place.

@contextlib.contextmanager
def _flag(owner, attribute: str, value):
    saved = getattr(owner, attribute)
    setattr(owner, attribute, value)
    try:
        yield
    finally:
        setattr(owner, attribute, saved)


def _sdpa_math():
    from torch.nn.attention import SDPBackend, sdpa_kernel
    return sdpa_kernel(SDPBackend.MATH)


@contextlib.contextmanager
def _channels_last(vendor):
    """Every convolution's input channels-last: ``torch.equal`` values, other strides."""
    def last(_module, args):
        return (args[0].contiguous(memory_format=torch.channels_last),) + args[1:]
    handles = [m.register_forward_pre_hook(last) for m in vendor.model.modules()
               if isinstance(m, torch.nn.Conv2d)]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextlib.contextmanager
def _frozen(vendor):
    """Every parameter ``requires_grad=False`` -- upstream leaves them trainable."""
    for parameter in vendor.model.parameters():
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter in vendor.model.parameters():
            parameter.requires_grad_(True)


def _log_softmax_on_host():
    """Each step's log-softmax on the CPU, then back: upstream takes it where the logits are."""
    def log_softmax(x, dim):
        return F.log_softmax(x.cpu(), dim=dim).to(x.device)
    return mock.patch.object(vdecoding, "F", SimpleNamespace(log_softmax=log_softmax))


@contextlib.contextmanager
def _autocast_not_half(vendor):
    """fp32 weights under fp16 autocast, instead of the whole model in fp16 as upstream runs it."""
    vendor.model.float()
    vendor.dtype = torch.float32
    run = vendor.run

    def autocast_run(*args, **kwargs):
        with torch.autocast("cuda", torch.float16):
            return run(*args, **kwargs)
    vendor.run = autocast_run
    try:
        yield
    finally:
        del vendor.run
        vendor.model.half()  # the values came from fp16, so the round trip is exact
        vendor.dtype = torch.float16


_ALL = {"cpu-fp32", "cuda-fp32", "cuda-fp16"}
_CUDA = {"cuda-fp32", "cuda-fp16"}

#: name: (perturbation(vendor) -> context manager, fixture, first stage that must fail, where).
FALSIFICATIONS = {
    "resize-bilinear": (lambda _: mock.patch("mozo.vendors.florence2_deploy.predictor.preprocess",
                                             _bilinear_preprocess), "small", "pixel_values", _ALL),
    "pil-reducing-gap": (lambda _: mock.patch("mozo.vendors.florence2_deploy.predictor.preprocess",
                                              _reducing_gap), "cut", "pixel_values", _ALL),
    "channel-attn-scale": (lambda _: _channel_scale_by_group(), "small", "vision_stage_0", _ALL),
    "pos-embed-rows-columns": (lambda _: _rows_columns_swapped(), "small", "image_features", _ALL),
    "no-temporal-embed": (lambda _: _no_temporal(), "small", "image_features", _ALL),
    "position-offset-0": (lambda _: _encoder_offset_zero(), "small", "encoder_layer_0", _ALL),
    "decoder-norm-eps": (lambda _: _decoder_norm_eps(), "small", "step_1", _ALL),
    # Only where the beams would repeat a 3-token sequence; the blank image's do.
    "no-repeat-ngram-off": (lambda _: _no_ngram(), "blank", "step_", _ALL),
    "forced-eos-off": (lambda _: _no_forced_eos(), "cut", "step_23", _ALL),
    # The length penalty only decides between finished hypotheses of different lengths, so it
    # never shows in a step's scores: the first thing it moves is which sequence wins.
    "length-penalty-2": (lambda _: _rules(length_penalty=2.0), "ocr-blank", "sequence", _ALL),
    "od-score-shifted-tokens": (lambda _: _label_scores(), "small", "parsed_scores", _ALL),
    "box-without-half-bin": (lambda _: _no_half_bin(), "small", "parsed_boxes", _ALL),
    # The parameters' flags are a recorded stage of their own. (On CUDA they change no
    # arithmetic here: frozen, every later stage still matched.)
    "requires-grad-off": (_frozen, "small", "requires_grad", _ALL),
    # CUDA only. The first Linear is in DaViT stage 0.
    "tf32-on": (lambda _: _flag(torch.backends.cuda.matmul, "allow_tf32", True), "small",
                "vision_stage_0", {"cuda-fp32"}),
    # Measured on an RTX 3060: stage 0's convolutions (3 -> 128, 7 x 7, and the depthwise ones)
    # get no TF32 kernel, so the first convolution TF32 reaches is stage 1's embedding.
    "cudnn-tf32-off": (lambda _: _flag(torch.backends.cudnn, "allow_tf32", False), "small",
                       "vision_stage_1", {"cuda-fp32"}),
    # Measured on an RTX 3060: only from stage 2's shapes does cuBLAS pick a kernel whose
    # reduction the flag changes.
    "fp16-full-reduction": (lambda _: _flag(torch.backends.cuda.matmul,
                                            "allow_fp16_reduced_precision_reduction", False),
                            "small", "vision_stage_2", {"cuda-fp16"}),
    "channels-last": (_channels_last, "small", "vision_stage_0", _CUDA),
    # SDPA is only in the BART half (DaViT's attention is written out).
    "sdpa-math": (lambda _: _sdpa_math(), "small", "encoder_layer_0", _CUDA),
    # Step 0 is forced (every score -inf but <s>), so the first step the log-softmax shows in is 1.
    "log-softmax-on-host": (lambda _: _log_softmax_on_host(), "small", "step_1", _CUDA),
    "autocast-not-half": (_autocast_not_half, "small", "vision_stage_0", {"cuda-fp16"}),
}


def falsify(names: list[str], references: References, vendor: Florence2Predictor, depth: int,
            where: str) -> int:
    """Each perturbation must make the comparison fail, at the stage it reaches first."""
    from PIL import Image

    photo = load_image(str(FIXTURES / "images" / "example.jpg"))
    fixtures = {
        "small": Case("small photo", np.array(Image.fromarray(photo).resize((480, 320), Image.BICUBIC))),
        "blank": Case("blank white", np.full((300, 400, 3), 255, np.uint8)),
        "ocr-blank": Case("OCR blank", load_image(str(FIXTURES / "text" / "blank.png")),
                          "<OCR_WITH_REGION>"),
        "cut": Case("photo cut at 24", photo, max_new_tokens=24),
    }
    names = [name for name in names if where in FALSIFICATIONS[name][3]]
    needed = {FALSIFICATIONS[name][1] for name in names}
    wanted = {name: references.get(case) for name, case in fixtures.items() if name in needed}
    references.close()
    unperturbed = {name: compare(ours(fixtures[name], vendor), wanted[name], depth)[1]
                   for name in needed}
    if any(unperturbed.values()):
        raise SystemExit(f"the gate fails before any perturbation: {unperturbed}")
    missed = 0
    for name in names:
        perturbation, fixture, stage, _ = FALSIFICATIONS[name]
        with perturbation(vendor):
            _, failures = compare(ours(fixtures[fixture], vendor), wanted[fixture], depth)
        first = failures[0] if failures else "nothing"
        caught = bool(failures) and first.startswith(stage)
        missed += not caught
        then = f"; then {failures[1][:60]}" if len(failures) > 1 else ""
        print(f"  {'caught' if caught else 'MISSED'} {name:26s} first failure: {first[:80]}{then}"
              + ("" if caught else f"   (expected {stage})"), flush=True)
    # A perturbation must leave nothing behind: the unperturbed comparison passes again.
    for name in needed:
        if compare(ours(fixtures[name], vendor), wanted[name], depth)[1]:
            raise SystemExit(f"a perturbation leaked: {fixtures[name].name} no longer passes")
    print(f"  {len(names) - missed} of {len(names)} caught ({where})", flush=True)
    return missed


def weights(variant: str) -> Path:
    return published(variant)


def served(variant: str, device: str, precision: str):
    """mozo's adapter as a user gets it; on CUDA fp16 through ``auto``, which must pick fp16."""
    from mozo.adapters.florence2_detect import Florence2DetectPredictor

    wanted = {"fp32": "torch-fp32", "fp16": "torch-fp16"}[precision]
    runtime = "auto" if (device, precision) in (("cpu", "fp32"), ("cuda", "fp16")) else wanted
    model = Florence2DetectPredictor(variant, device=device, runtime=runtime)
    if model.runtime != wanted:
        raise SystemExit(f"{variant} on {device} via {runtime} ran {model.runtime}, not {wanted}")
    return model


def free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", nargs="+", required=True, metavar="VARIANT=DIR")
    parser.add_argument("--processor", help="Florence-2-base snapshot (post-processor)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=tuple(PRECISIONS),
                        help="default: fp32 on the CPU; fp32 and fp16 on CUDA")
    parser.add_argument("--quick", action="store_true", help="the photograph only")
    parser.add_argument("--no-mozo", action="store_true", help="skip the served path")
    parser.add_argument("--repeat", type=int, default=2,
                        help="runs per case on each side, each byte-identical to the first")
    parser.add_argument("--reference-cache", type=Path, metavar="DIR",
                        help="keep and reuse the reference's records (digests)")
    parser.add_argument("--falsify", nargs="+", metavar="NAME",
                        choices=[*FALSIFICATIONS, "all"], help="perturbations that must be caught")
    args = parser.parse_args()
    references = dict(item.split("=", 1) for item in args.reference)
    processor = args.processor or references.get("base")
    if processor is None:
        raise SystemExit("--processor is needed when base is not among the references")
    if args.device == "cpu" and args.dtype not in (None, "fp32"):
        raise SystemExit("mozo runs Florence-2 in fp16 on CUDA only")
    precisions = [args.dtype] if args.dtype else (["fp32", "fp16"] if args.device == "cuda"
                                                  else ["fp32"])
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda, but torch sees no GPU")
    print(environment(args.device))
    before = settings()
    if args.device == "cuda":
        print("settings:", ", ".join(f"{k}={v}" for k, v in before.items()))

    failed = 0
    grand = 0
    for precision in precisions:
        for variant, model_dir in references.items():
            vendor = Florence2Predictor(weights(variant), variant, args.device,
                                        PRECISIONS[precision])
            depth = vendor.spec.language.layers
            upstream = References(variant, model_dir, processor, args.device, precision,
                                  args.reference_cache, args.repeat, depth)
            where = f"{args.device}-{precision}"
            if args.falsify:
                names = list(FALSIFICATIONS) if "all" in args.falsify else args.falsify
                print(f"{variant} {where}: falsifying", flush=True)
                failed += falsify(names, upstream, vendor, depth, where)
                del vendor, upstream
                free()
                continue
            mozo_model = None
            total = determinism = 0
            for case in cases(args.quick):
                if variant not in ("base", "base-ft") and not case.also_large:
                    continue
                want = upstream.get(case)
                mine, repeats = run_ours(case, vendor, args.repeat, depth)
                determinism += repeats
                checks, failures = compare(mine, want, depth)
                if not args.no_mozo and case.task == "<OD>" and case.max_new_tokens == Case.max_new_tokens:
                    if mozo_model is None:
                        mozo_model = served(variant, args.device, precision)
                    more, extra = against_mozo(mozo_model, case, want)
                    checks += more
                    failures += extra
                total += checks
                failed += bool(failures)
                print(f"  {variant:8s} {precision} {case.name:44s} {checks:4d} comparisons, "
                      f"{len(want['scores']):4d} steps: "
                      + ("all equal" if not failures else f"FAILED {failures[0][:100]}"), flush=True)
            determinism += upstream.determinism
            grand += total + determinism
            print(f"{variant} {where}: {total} comparisons against the reference, {determinism} "
                  f"determinism (mozo repeats {determinism - upstream.determinism}, reference "
                  f"{upstream.determinism})", flush=True)
            del vendor, upstream, mozo_model
            free()
    if settings() != before:
        print(f"global settings changed during the run: {before} -> {settings()}")
        failed += 1
    if not args.falsify:
        print(f"{grand} comparisons in all")
    print("PASS" if not failed else f"FAIL ({failed})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

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

**Every stage, then every step.** Pixels, prompt ids, each DaViT stage, the projected image tokens,
the encoder's input and each encoder layer -- then every decoding step's processed log-probabilities
for all three beams, the number of steps, the sequence, the beam indices, the transition scores, and
the parsed boxes, labels and scores. ``torch.equal`` and ``==``, no tolerance.

**What is pinned.** CPU, float32 (the published fp16 weights upcast at load on both sides), eager
attention, the reference's parameters re-allocated before comparing, ``MKL_CBWR=AVX2,STRICT``
expected in the environment. Versions in :data:`PINNED` are printed, not enforced.

**The gate is shown to fail.** ``--falsify NAME`` (or ``all``) perturbs one thing in the vendored
package and requires the comparison to fail at the stage that thing reaches first.

    python tools/verify/florence2_detect.py --reference base=DIR          # base, every fixture
    python tools/verify/florence2_detect.py --reference base=DIR large=DIR2 --quick
    python tools/verify/florence2_detect.py --reference base=DIR --falsify all

``--reference VARIANT=DIR`` names a local snapshot of the variant's Hub repository at the pinned
commit (``hf download <repo> --revision <commit> --local-dir DIR``); ``--processor DIR`` the
``Florence-2-base`` snapshot whose post-processor is used (default: the ``base`` reference).
The vendor and mozo read the published ``torch-fp16.pth`` from ``weights/``.
"""

from __future__ import annotations

import argparse
import io
import math
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import numpy as np
import torch

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


def versions() -> str:
    import tokenizers
    import transformers
    got = {"torch": torch.__version__, "transformers": transformers.__version__,
           "tokenizers": tokenizers.__version__}
    return ", ".join(f"{k} {v}" + ("" if v.split("+")[0] == PINNED[k] else f" (pinned {PINNED[k]})")
                     for k, v in got.items())


@dataclass(frozen=True, eq=False)
class Case:
    """One fixture: an image, a task, a length limit. ``also_large`` marks the larger variants'."""

    name: str
    rgb: np.ndarray
    task: str = "<OD>"
    max_new_tokens: int = 1024
    also_large: bool = False


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


class Reference:
    """Microsoft's code, with hooks on every stage the vendor reports."""

    def __init__(self, model_dir: str, processor_dir: str) -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor

        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, torch_dtype=torch.float32,
            attn_implementation="eager").eval()
        # Re-allocated, so what is compared is arithmetic, not the checkpoint's memory offsets.
        for parameter in model.parameters():
            parameter.data = parameter.data.clone()
        for module in model.modules():
            for name, buffer in list(module._buffers.items()):
                if buffer is not None:
                    module._buffers[name] = buffer.clone()
        self.model = model
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

    @torch.inference_mode()
    def run(self, case: Case) -> dict:
        from PIL import Image

        self.caught = {}
        inputs = self.processor(text=case.task, images=Image.fromarray(case.rgb),
                                return_tensors="pt")
        out = self.model.generate(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
            max_new_tokens=case.max_new_tokens, return_dict_in_generate=True, output_scores=True)
        sequence = out.sequences[0]
        transition = self.model.compute_transition_scores(
            out.sequences, out.scores, out.beam_indices, normalize_logits=False)[0]
        size = (case.rgb.shape[1], case.rgb.shape[0])
        parsed = self.processor.post_process_generation(
            sequence=sequence, transition_beam_score=transition, task=case.task,
            image_size=size)[case.task]
        return {**self.caught, "pixel_values": inputs["pixel_values"],
                "input_ids": inputs["input_ids"][0], "scores": list(out.scores),
                "sequence": sequence, "beam_indices": out.beam_indices[0],
                "transition": transition, "parsed": parsed}


def _same(a, b) -> str | None:
    if a.shape != b.shape:
        return f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
    if torch.equal(a, b):
        return None
    finite = torch.isfinite(a) & torch.isfinite(b)
    if not torch.equal(torch.isfinite(a), torch.isfinite(b)):
        return "different -inf positions"
    return f"max |d| {(a[finite] - b[finite]).abs().max().item():.2g}"


def compare(case: Case, want: dict, vendor: Florence2Predictor, depth: int) -> tuple[int, list[str]]:
    """Every comparison for *case*: how many were made, and the failures in stage order."""
    hooks: dict[str, torch.Tensor] = {}
    output = vendor.run(case.rgb, case.task, hooks, max_new_tokens=case.max_new_tokens)
    beams = output.beams
    checks = 0
    failures = []

    def check(name, problem):
        nonlocal checks
        checks += 1
        if problem:
            failures.append(f"{name}: {problem}")

    check("input_ids", None if list(PROMPTS[case.task]) == want["input_ids"].tolist()
          else "prompt ids differ")
    stages = (["pixel_values"] + [f"vision_stage_{i}" for i in range(4)]
              + ["image_features", "encoder_input"] + [f"encoder_layer_{i}" for i in range(depth)])
    for name in stages:
        check(name, _same(hooks[name], want[name]))
    for step, (ours, theirs) in enumerate(zip(beams.scores, want["scores"])):
        check(f"step_{step}", _same(ours, theirs))
    check("steps", None if len(beams.scores) == len(want["scores"])
          else f"{len(beams.scores)} vs {len(want['scores'])}")
    check("sequence", _same(beams.sequence, want["sequence"]))
    check("beam_indices", _same(beams.beam_indices[0].to(torch.int64),
                                want["beam_indices"].to(torch.int64)))
    check("transition", _same(vdecoding.transition_scores(beams, vendor.spec.language.vocab),
                              want["transition"]))
    parsed = want["parsed"]
    if case.task == "<OD>":
        boxes = vendor.parse_od(output)
        ours = ([b.box for b in boxes], [b.label for b in boxes], [b.score for b in boxes])
        theirs = (parsed["bboxes"], parsed["labels"], parsed.get("scores", []))
        names = ("parsed_boxes", "parsed_labels", "parsed_scores")
    else:
        lines = vendor.parse_ocr(output)
        ours = ([line.quad for line in lines], [line.text for line in lines])
        theirs = (parsed["quad_boxes"], parsed["labels"])
        names = ("parsed_quads", "parsed_text")
    for name, a, b in zip(names, ours, theirs):
        check(name, None if a == b else f"{a[:2]!r}... vs {b[:2]!r}...")
    return checks, failures


def against_mozo(variant: str, case: Case, want: dict) -> tuple[int, list[str]]:
    """The served path: registry, weights from the mirror, the adapter, PixelFlow.

    PixelFlow rounds coordinates and scores to its own precision, so the reference's parsed boxes
    go through the same door (``pf.detections.from_arrays``) before they are compared.
    """
    import pixelflow as pf

    import mozo

    found = mozo.get_model(f"florence2_detect/{variant}", device="cpu").predict(case.rgb)
    parsed = want["parsed"]
    door = pf.detections.from_arrays(
        boxes=np.array(parsed["bboxes"], dtype=float).reshape(-1, 4),
        scores=np.array(parsed.get("scores", []), dtype=float))
    failures = []
    for name, ours, theirs in (
            ("mozo_boxes", [d.bbox for d in found], [d.bbox for d in door]),
            ("mozo_labels", [d.class_name for d in found], parsed["labels"]),
            ("mozo_scores", [d.confidence for d in found], [d.confidence for d in door])):
        if ours != theirs:
            failures.append(f"{name}: {ours[:2]!r}... vs {theirs[:2]!r}...")
    return 3, failures


# --- Falsification: each perturbation must be caught at the stage it reaches first. ---

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


FALSIFICATIONS = {
    "resize-bilinear": (lambda: mock.patch("mozo.vendors.florence2_deploy.predictor.preprocess",
                                           _bilinear_preprocess), "small", "pixel_values"),
    "pil-reducing-gap": (lambda: mock.patch("mozo.vendors.florence2_deploy.predictor.preprocess",
                                            _reducing_gap), "cut", "pixel_values"),
    "channel-attn-scale": (_channel_scale_by_group, "small", "vision_stage_0"),
    "pos-embed-rows-columns": (_rows_columns_swapped, "small", "image_features"),
    "no-temporal-embed": (_no_temporal, "small", "image_features"),
    "position-offset-0": (_encoder_offset_zero, "small", "encoder_layer_0"),
    "decoder-norm-eps": (_decoder_norm_eps, "small", "step_1"),
    # Only where the beams would repeat a 3-token sequence; the blank image's do.
    "no-repeat-ngram-off": (_no_ngram, "blank", "step_"),
    "forced-eos-off": (_no_forced_eos, "cut", "step_23"),
    # The length penalty only decides between finished hypotheses of different lengths, so it
    # never shows in a step's scores: the first thing it moves is which sequence wins.
    "length-penalty-2": (lambda: _rules(length_penalty=2.0), "ocr-blank", "sequence"),
    "od-score-shifted-tokens": (_label_scores, "small", "parsed_scores"),
    "box-without-half-bin": (_no_half_bin, "small", "parsed_boxes"),
}


def falsify(names: list[str], reference: Reference, vendor: Florence2Predictor, depth: int) -> int:
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
    needed = {FALSIFICATIONS[name][1] for name in names}
    wanted = {name: reference.run(case) for name, case in fixtures.items() if name in needed}
    missed = 0
    for name in names:
        perturbation, fixture, stage = FALSIFICATIONS[name]
        with perturbation():
            _, failures = compare(fixtures[fixture], wanted[fixture], vendor, depth)
        first = failures[0] if failures else "nothing"
        caught = bool(failures) and first.startswith(stage)
        missed += not caught
        print(f"  {'caught' if caught else 'MISSED'} {name:26s} first failure: {first[:90]}"
              + ("" if caught else f"   (expected {stage})"), flush=True)
    return missed


def weights(variant: str) -> Path:
    return published(variant)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", nargs="+", required=True, metavar="VARIANT=DIR")
    parser.add_argument("--processor", help="Florence-2-base snapshot (post-processor)")
    parser.add_argument("--quick", action="store_true", help="the photograph only")
    parser.add_argument("--no-mozo", action="store_true", help="skip the served path")
    parser.add_argument("--falsify", nargs="+", metavar="NAME",
                        choices=[*FALSIFICATIONS, "all"], help="perturbations that must be caught")
    args = parser.parse_args()
    references = dict(item.split("=", 1) for item in args.reference)
    processor = args.processor or references.get("base")
    if processor is None:
        raise SystemExit("--processor is needed when base is not among the references")
    print(f"versions: {versions()}; MKL_CBWR={os.environ.get('MKL_CBWR', 'unset')}; "
          f"threads {torch.get_num_threads()}")

    failed = 0
    for variant, model_dir in references.items():
        reference = Reference(model_dir, processor)
        vendor = Florence2Predictor(weights(variant), variant)
        depth = vendor.spec.language.layers
        if args.falsify:
            names = list(FALSIFICATIONS) if "all" in args.falsify else args.falsify
            print(f"{variant}: falsifying {len(names)}", flush=True)
            failed += falsify(names, reference, vendor, depth)
            continue
        total = 0
        for case in cases(args.quick):
            if variant not in ("base", "base-ft") and not case.also_large:
                continue
            want = reference.run(case)
            checks, failures = compare(case, want, vendor, depth)
            if not args.no_mozo and case.task == "<OD>" and case.max_new_tokens == Case.max_new_tokens:
                more, extra = against_mozo(variant, case, want)
                checks += more
                failures += extra
            total += checks
            failed += bool(failures)
            print(f"  {variant:8s} {case.name:44s} {checks:4d} comparisons, "
                  f"{len(want['scores']):4d} steps: "
                  + ("all equal" if not failures else f"FAILED {failures[0][:100]}"), flush=True)
        print(f"{variant}: {total} comparisons", flush=True)
    print("PASS" if not failed else f"FAIL ({failed})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Check that mozo's Qwen3-VL writes exactly what the published model does, token by token.

Three paths reach the same weights and this compares all three: the vendored package
(``mozo/vendors/qwen3_vl_deploy``), the reference (``transformers``'
``Qwen3VLForConditionalGeneration`` + ``AutoProcessor`` + ``generate``), and mozo itself (registry,
weights, adapter).

**The reference is ``transformers``, and that is not a choice.** Qwen publishes no inference code of
its own; the model card's only code path is ``from_pretrained`` and ``generate``.

**Every stage, then every step.** Pixels, prompt ids, the patch embedding with its interpolated
positions, each of the 24 vision blocks, the merger and the three DeepStack features, the input
embeddings after the image is scattered in, the 3-D rotary positions, each decoder layer at
prefill, the final norm -- and then every decoding step's float32 logits and chosen id, and the
decoded string. ``torch.equal``, no tolerance. Greedy and sampled: for sampling the reference runs
after ``torch.manual_seed(seed)`` and mozo draws from ``torch.Generator().manual_seed(seed)``, and
the ids must agree exactly.

**What is pinned.** The CPU, in float32 (the reference loads in float32 explicitly -- transformers 5
would otherwise run the stored bfloat16); SDPA attention; the parameters re-allocated before
comparing; ``MKL_CBWR=AVX2,STRICT`` is expected in the environment so MKL cannot pick a code path
by alignment. The reference's version matters: 4.57.3 and 5.24.1 give the same words and
different logits at every step. Run it under the versions in :data:`PINNED`.

**The gate is shown to fail.** ``--falsify NAME`` (or ``all``) perturbs one thing in the vendored
package and requires the comparison to fail, at the stage that thing reaches first.

    python tools/verify/qwen3_vl.py                      # 2b everything, 4b on three fixtures
    python tools/verify/qwen3_vl.py --quick              # one small fixture per variant
    python tools/verify/qwen3_vl.py --falsify all        # every perturbation must be caught
    python tools/verify/qwen3_vl.py --fuzz 20000         # the tokenizer against the reference's

``--reference DIR`` points at a local snapshot of the Hub repository (the pinned commit); otherwise
it is read from the Hub at that commit.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import random
import sys
import time
from pathlib import Path
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault("MOZO_BASE_URL", f"file://{ROOT / 'weights'}")

from fetch.qwen3_vl import COMMITS, repository  # noqa: E402

from mozo.vendors.qwen3_vl_deploy import (  # noqa: E402
    MAX_PIXELS, SPECS, Generator, Sampling)
from mozo.vendors.qwen3_vl_deploy import image as vimage  # noqa: E402
from mozo.vendors.qwen3_vl_deploy import language as vlanguage  # noqa: E402
from mozo.vendors.qwen3_vl_deploy import model as vmodel  # noqa: E402
from mozo.vendors.qwen3_vl_deploy import tokenizer as vtokenizer  # noqa: E402
from mozo.vendors.qwen3_vl_deploy import vision as vvision  # noqa: E402
from mozo.weights import WeightsError, resolve  # noqa: E402

#: The versions this family's parity was established against. Printed, not enforced.
PINNED = {"torch": "2.13.0", "torchvision": "0.28.0", "transformers": "5.24.1",
          "tokenizers": "0.23.3"}

FIXTURE = ROOT / "tests" / "fixtures" / "images" / "example.jpg"


def versions() -> str:
    import tokenizers
    import torchvision
    import transformers
    got = {"torch": torch.__version__, "torchvision": torchvision.__version__,
           "transformers": transformers.__version__, "tokenizers": tokenizers.__version__}
    return ", ".join(f"{k} {v}" + ("" if v.split("+")[0] == PINNED[k] else f" (pinned {PINNED[k]})")
                     for k, v in got.items())


class Case:
    """One fixture: an image, a prompt, a budget, how to choose, how many pixels."""

    def __init__(self, name: str, rgb: np.ndarray, prompt: str, steps: int,
                 sampling: Sampling = Sampling(), max_pixels: int = MAX_PIXELS) -> None:
        self.name, self.rgb, self.prompt, self.steps = name, rgb, prompt, steps
        self.sampling, self.max_pixels = sampling, max_pixels


def _load(path: Path, mode: str = "RGB") -> np.ndarray:
    from PIL import Image
    return np.array(Image.open(path).convert(mode))


def cases(quick: bool) -> list[Case]:
    """The fixtures, awkward ones included (VENDORING §2.6)."""
    import io

    from PIL import Image

    from mozo.image import load_image

    photo = _load(FIXTURE)
    small = np.array(Image.fromarray(photo).resize((640, 427), Image.BICUBIC))
    if quick:
        return [Case("small photo, greedy", small, "Describe this image.", 12)]
    # RGBA and greyscale go through mozo's own loader, so both sides are handed the same RGB array.
    rgba, grey = io.BytesIO(), io.BytesIO()
    Image.fromarray(small).convert("RGBA").save(rgba, "PNG")
    Image.fromarray(small).convert("L").save(grey, "JPEG")
    return [
        Case("photo, default 2.6 MP budget", photo, "Describe this image.", 32),
        Case("photo, sampled (t=0.7, p=0.8, k=20, seed 3)", small, "Describe this image.", 24,
             Sampling(0.7, 0.8, 20, 3)),
        Case("photo, sampled (t=1.3, seed 11)", small, "What could happen next?", 16,
             Sampling(1.3, 1.0, 0, 11)),
        Case("stops on its own", small, "How many people are in this picture? Answer with a number.", 40),
        Case("tiny 9x13, scaled up to the pixel floor", photo[:13, :9], "What colour is this?", 12),
        Case("strip 2000x120", np.array(Image.fromarray(photo).resize((2000, 120))),
             "Describe this image.", 12),
        Case("512x512, no resize", photo[:512, :512], "What is here?", 12),
        Case("above the budget (262,144 px)", photo, "Describe this image.", 12, max_pixels=262_144),
        Case("RGBA PNG", load_image(rgba.getvalue()), "Describe this image.", 8),
        Case("greyscale JPEG", load_image(grey.getvalue()), "Describe this image.", 8),
        Case("CJK, emoji, combining mark, newline", small,
             "这张图片里有什么？😀 Café́ 'S\nanswer briefly", 16),
        Case("long prompt", small, " ".join(["Describe every object you can see, its colour, and "
                                             "where it is in the frame."] * 20), 8),
    ]


# --- the reference -------------------------------------------------------------------------------

class Reference:
    """transformers' model and processor, with hooks on every stage the vendor records."""

    def __init__(self, source: str, revision: str | None) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        kw = {} if revision is None else {"revision": revision}
        self.processor = AutoProcessor.from_pretrained(source, **kw)
        self.model = AutoModelForImageTextToText.from_pretrained(
            source, dtype=torch.float32, attn_implementation="sdpa", **kw).eval()
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.data = parameter.data.clone()
        self.caught: dict[str, torch.Tensor] = {}
        visual, text = self.model.model.visual, self.model.model.language_model
        visual.blocks[0].register_forward_pre_hook(self._pre("patch_embed"), with_kwargs=True)
        for i, block in enumerate(visual.blocks):
            block.register_forward_hook(self._post(f"vision_block_{i}"))
        visual.merger.register_forward_hook(self._post("merger"))
        for i, merger in enumerate(visual.deepstack_merger_list):
            merger.register_forward_hook(self._post(f"deepstack_{i}"))
        text.layers[0].register_forward_pre_hook(self._pre("embeds"), with_kwargs=True)
        # A layer's output as the vendor records it is after DeepStack, so it is read as the next
        # layer's input (and the last as the final norm's).
        for i in range(1, len(text.layers)):
            text.layers[i].register_forward_pre_hook(self._pre(f"text_layer_{i - 1}"), with_kwargs=True)
        text.norm.register_forward_pre_hook(self._pre(f"text_layer_{len(text.layers) - 1}"),
                                            with_kwargs=True)
        text.norm.register_forward_hook(self._post("final_norm"))

    def _pre(self, name):
        def hook(module, args, kwargs):
            x = args[0] if args else kwargs["hidden_states"]
            self.caught.setdefault(name, x.detach().clone())
        return hook

    def _post(self, name):
        def hook(module, args, out):
            self.caught.setdefault(name, (out[0] if isinstance(out, tuple) else out).detach().clone())
        return hook

    @torch.inference_mode()
    def run(self, case: Case) -> dict:
        from PIL import Image

        image = Image.fromarray(case.rgb)
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                 {"type": "text", "text": case.prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt",
                                size={"longest_edge": case.max_pixels, "shortest_edge": 65_536})
        s = case.sampling
        how = (dict(do_sample=False, temperature=None, top_p=None, top_k=None) if s.greedy else
               dict(do_sample=True, temperature=s.temperature, top_p=s.top_p, top_k=s.top_k))
        self.caught = {}
        if not s.greedy:
            torch.manual_seed(s.seed)
        out = self.model.generate(**inputs, max_new_tokens=case.steps, output_logits=True,
                                  return_dict_in_generate=True, **how)
        ids = out.sequences[0, inputs["input_ids"].shape[1]:].tolist()
        return {"pixel_values": inputs["pixel_values"], "input_ids": inputs["input_ids"],
                "grid": tuple(inputs["image_grid_thw"][0].tolist()), **self.caught,
                "logits": [step[0] for step in out.logits], "ids": ids,
                "text": self.processor.decode(ids, skip_special_tokens=True)}


# --- comparing -----------------------------------------------------------------------------------

def compare(case: Case, want: dict, vendor: Generator) -> list[str]:
    """Every difference between the vendor's run of *case* and the reference's, first stage first."""
    record: dict = {}
    try:
        answer = vendor.generate(case.rgb, case.prompt, case.steps, case.sampling, case.max_pixels,
                                 record=record)
    except Exception as error:  # a perturbation can also break a shape
        return [f"raised {type(error).__name__}: {error}"]
    failures = []
    if record["grid"] != want["grid"]:
        return [f"grid {record['grid']} != {want['grid']}"]
    order = (["pixel_values", "input_ids", "patch_embed"]
             + [f"vision_block_{i}" for i in range(vendor.spec.vision.depth)]
             + [f"deepstack_{i}" for i in range(len(vendor.spec.vision.deepstack))] + ["merger", "embeds"]
             + [f"text_layer_{i}" for i in range(vendor.spec.text.layers)] + ["final_norm"])
    for stage in order:
        a, b = record[stage], want[stage]
        if a.shape != b.shape:
            failures.append(f"{stage}: shape {tuple(a.shape)} != {tuple(b.shape)}")
        elif not torch.equal(a, b):
            failures.append(f"{stage}: max |d| {(a.float() - b.float()).abs().max().item():.3e}")
    steps = min(len(record["logits"]), len(want["logits"]))
    for step in range(steps):
        if not torch.equal(record["logits"][step], want["logits"][step]):
            d = (record["logits"][step] - want["logits"][step]).abs().max().item()
            failures.append(f"step {step} logits: max |d| {d:.3e}")
            break
    if answer.ids != want["ids"]:
        first = next((i for i, (x, y) in enumerate(zip(answer.ids, want["ids"])) if x != y),
                     min(len(answer.ids), len(want["ids"])))
        failures.append(f"generated ids differ from token {first}")
    if answer.text != want["text"]:
        failures.append("decoded text differs")
    return failures


def against_mozo(variant: str, case: Case, want: dict) -> list[str]:
    """mozo's adapter, through the registry and the weights tree, must write the same answer."""
    import mozo

    model = mozo.get_model(f"qwen3_vl/{variant}", device="cpu")
    s = case.sampling
    got = model.predict(case.rgb, case.prompt, max_new_tokens=case.steps, temperature=s.temperature,
                        top_p=s.top_p, top_k=s.top_k, seed=s.seed, max_pixels=case.max_pixels)
    failures = []
    if model.runtime != "torch-fp32":
        failures.append(f"auto chose {model.runtime} on the CPU")
    if got.text != want["text"]:
        failures.append("the adapter's text differs")
    finished = "stop" if want["ids"] and want["ids"][-1] in (151645, 151643) else "length"
    if (got.finish_reason, got.generated_tokens, got.prompt_tokens) != (
            finished, len(want["ids"]), want["input_ids"].shape[1]):
        failures.append(f"the adapter reports {got.finish_reason}/{got.generated_tokens}/"
                        f"{got.prompt_tokens}")
    if model.revision is None:
        failures.append("the adapter does not say which revision it runs")
    return failures


# --- falsification -------------------------------------------------------------------------------

def _no_antialias():
    import types
    real = vimage.tvF
    return types.SimpleNamespace(
        resize=lambda image, size, interpolation, antialias=True:
        real.resize(image, size, interpolation=interpolation, antialias=False),
        normalize=real.normalize, InterpolationMode=real.InterpolationMode)


def _axis_taps_half_pixel(index, size, side):
    index = index.to(torch.float32)
    src = (index + 0.5) * side / size - 0.5
    floor = torch.floor(src)
    offsets = torch.arange(0, 2, device=index.device)
    taps = (floor.long()[:, None] + offsets).clamp(0, side - 1)
    return taps, (1 - (src[:, None] - floor[:, None] - offsets).abs()).clamp(min=0)


def _smart_resize_floor(height, width, factor, min_pixels, max_pixels):
    h, w = vimage.math.floor(height / factor) * factor, vimage.math.floor(width / factor) * factor
    return max(h, factor), max(w, factor)


def _merger_norm_eps(self, x):
    """The mergers' LayerNorm at torch's default epsilon, 1e-5, instead of upstream's 1e-6."""
    x = x.view(-1, self.hidden) if self.postshuffle else x
    x = torch.nn.functional.layer_norm(x, self.norm.normalized_shape, self.norm.weight,
                                       self.norm.bias, 1e-5).view(-1, self.hidden)
    return self.linear_fc2(torch.nn.functional.gelu(self.linear_fc1(x)))


def _vision_mlp_exact(self, x):
    return self.linear_fc2(torch.nn.functional.gelu(self.linear_fc1(x)))


def _recompose_blocks(self, x, positions):
    """M-RoPE in three blocks (T then H then W) rather than interleaved."""
    positions = positions.expand(3, -1, -1)
    freqs = positions[..., None].float() * self.inv_freq.to(device=x.device, dtype=torch.float)
    sec = self.spec.mrope_section
    parts = torch.split(freqs, list(sec), dim=-1)
    f = torch.cat([parts[i][i] for i in range(3)], dim=-1)
    emb = torch.cat((f, f), dim=-1)
    return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


@contextlib.contextmanager
def _deepstack_shifted():
    original = vlanguage.TextDecoder.forward

    def forward(self, embeds, positions, cache, visual=None, deepstack=None, record=None):
        if deepstack is not None:
            deepstack = [torch.zeros_like(deepstack[0])] + list(deepstack)
        return original(self, embeds, positions, cache, visual, deepstack, record)

    with mock.patch.object(vlanguage.TextDecoder, "forward", forward):
        yield


@contextlib.contextmanager
def _patched(target, name, value):
    with mock.patch.object(target, name, value):
        yield


@contextlib.contextmanager
def _warp_order_swapped():
    original = vmodel._warp

    def warp(logits, sampling):
        p_first = vmodel.Sampling(1.0, sampling.top_p, 0, sampling.seed)
        k_then = vmodel.Sampling(sampling.temperature, 1.0, sampling.top_k, sampling.seed)
        return original(original(logits, p_first), k_then)

    with mock.patch.object(vmodel, "_warp", warp):
        yield


@contextlib.contextmanager
def _seed_off_by_one():
    original = vmodel.Qwen3VL.generate

    def generate(self, input_ids, pixels, grid, max_new_tokens, sampling=vmodel.Sampling(),
                 record=None):
        shifted = vmodel.Sampling(sampling.temperature, sampling.top_p, sampling.top_k,
                                  sampling.seed + 1)
        return original(self, input_ids, pixels, grid, max_new_tokens, shifted, record)

    with mock.patch.object(vmodel.Qwen3VL, "generate", generate):
        yield


@contextlib.contextmanager
def _no_nfc():
    with mock.patch.object(vtokenizer, "_nfc", lambda text, boundaries: text):
        vtokenizer._word.cache_clear()
        yield
    vtokenizer._word.cache_clear()


#: name -> (perturbation, the fixture that exposes it, the stage that must fail first).
FALSIFICATIONS = {
    "no-antialias": (lambda: _patched(vimage, "tvF", _no_antialias()), "small", "pixel_values"),
    "resize-floor": (lambda: _patched(vimage, "smart_resize", _smart_resize_floor), "small", "grid"),
    "pos-embed-half-pixel": (lambda: _patched(vvision, "_axis_taps", _axis_taps_half_pixel),
                             "small", "patch_embed"),
    "vision-gelu-exact": (lambda: _patched(vvision.MLP, "forward", _vision_mlp_exact),
                          "small", "vision_block_0"),
    "merger-norm-eps": (lambda: _patched(vvision.PatchMerger, "forward", _merger_norm_eps),
                        "small", "deepstack_0"),
    "vision-rope-theta": (lambda: _patched(vvision.VisionTower, "_rotary",
                                           _scaled_rotary(vvision.VisionTower._rotary)),
                          "small", "vision_block_0"),
    "mrope-blocks": (lambda: _patched(vlanguage.TextDecoder, "_rotary", _recompose_blocks),
                     "small", "text_layer_0"),
    "deepstack-one-layer-late": (_deepstack_shifted, "small", "text_layer_0"),
    "text-rope-theta": (lambda: _patched(vlanguage.TextDecoder, "_rotary", _text_theta(1_000_000.0)),
                        "small", "text_layer_0"),
    "rmsnorm-eps": (lambda: _patched(vlanguage.RMSNorm, "forward", _rmsnorm_eps(1e-5)),
                    "small", "text_layer_0"),
    # A sampled token that differs is fed back, so the first difference is a later step's logits.
    "top-p-before-top-k": (_warp_order_swapped, "sampled", "step"),
    "seed-off-by-one": (_seed_off_by_one, "open", "step"),
    "tokenizer-no-nfc": (_no_nfc, "nfc", "input_ids"),
}


def _text_theta(theta):
    """The text decoder's rotary base at *theta* instead of 5,000,000."""
    original = vlanguage.TextDecoder._rotary

    def rotary(self, x, positions):
        saved = self.inv_freq
        d = self.spec.head_dim
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float) / d))
        try:
            return original(self, x, positions)
        finally:
            self.inv_freq = saved
    return rotary


def _scaled_rotary(original):
    def rotary(self, x, positions):
        cos, sin = original(self, x, positions * 1.0000001)
        return cos, sin
    return rotary


def _rmsnorm_eps(eps):
    """The text decoder's RMSNorm (and its q/k norms) at another epsilon."""
    def forward(self, x):
        dtype = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        return self.weight * x.to(dtype)
    return forward


def falsify(names: list[str], reference: Reference, vendor_factory) -> int:
    """Each perturbation must make the comparison fail, at the stage it reaches first."""
    from PIL import Image

    photo = _load(FIXTURE)
    small = np.array(Image.fromarray(photo).resize((320, 214), Image.BICUBIC))
    fixtures = {
        "small": Case("small", small, "Describe this image.", 4),
        # top_k small and top_p below what three tokens cover, so the warpers' order matters.
        "sampled": Case("sampled", small, "Describe this image.", 12, Sampling(1.5, 0.6, 3, 5)),
        # The whole distribution at a high temperature, so two seeds part within a few tokens.
        "open": Case("open", small, "Describe this image.", 12, Sampling(1.5, 1.0, 0, 5)),
        "nfc": Case("nfc", small, "Café or café?", 3),
    }
    needed = {FALSIFICATIONS[name][1] for name in names}
    wanted = {name: reference.run(case) for name, case in fixtures.items() if name in needed}
    missed = 0
    for name in names:
        perturbation, fixture, stage = FALSIFICATIONS[name]
        with perturbation():
            vendor = vendor_factory()
            failures = compare(fixtures[fixture], wanted[fixture], vendor)
        first = failures[0] if failures else "nothing"
        caught = bool(failures) and first.startswith(stage)
        missed += not caught
        print(f"  {'caught' if caught else 'MISSED'} {name:26s} first failure: {first[:90]}"
              + ("" if caught else f"   (expected {stage})"))
    vtokenizer._word.cache_clear()
    return missed


def fuzz(count: int, reference: Reference) -> int:
    """The tokenizer on *count* random strings, and its decoder on random ids, against the reference."""
    tok = reference.processor.tokenizer
    ours = vtokenizer.Tokenizer()
    rng = random.Random(0)
    pools = ["abcxyz ABC'sStTmMdDlLreREve", "0123456789٠١٢३४५", " \t\n\r  　\u0085 ",
             "!?.,;:-_()[]{}\"'`~@#$%^&*+=/\\|<>", "éèêëçàñßſİıǅ", "éäõ̧\U00011d42",
             "中文字符漢字日本語ひらがなカタカナ한국어", "العربيةעברית", "ελληνικάкириллица", "😀👍🏽🇫🇷❤️‍🔥",
             "​‍﻿­", "Kſ'S'T'RE'VE'M'LL'D's't're've'm'll'd"]

    def text() -> str:
        if rng.random() < 0.15:
            raw = "".join(chr(rng.randrange(0, 0x110000)) for _ in range(rng.randrange(1, 30)))
            return raw.encode("utf-8", "ignore").decode("utf-8", "ignore") or "x"
        return "".join(rng.choice(rng.choice(pools)) for _ in range(rng.randrange(1, 60)))

    bad = 0
    for _ in range(count):
        t = text()
        if any(a in t for a in ours.added):
            continue
        bad += tok.encode(t, add_special_tokens=False) != ours.encode(t)
    for _ in range(count // 4):
        ids = [rng.randrange(0, 151669) for _ in range(rng.randrange(1, 40))]
        bad += tok.decode(ids, skip_special_tokens=True) != ours.decode(ids)
    print(f"  tokenizer: {count:,} strings and {count // 4:,} id sequences, {bad} differ")
    return bad


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variants", nargs="*", default=list(SPECS))
    parser.add_argument("--reference", help="local snapshot directory of the Hub repository")
    parser.add_argument("--quick", action="store_true", help="one small fixture per variant")
    parser.add_argument("--falsify", nargs="+", metavar="NAME",
                        choices=[*FALSIFICATIONS, "all"], help="perturbations that must be caught")
    parser.add_argument("--fuzz", type=int, default=0, metavar="N", help="tokenizer strings to fuzz")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if os.environ.get("MKL_CBWR") != "AVX2,STRICT":
        print("warning: MKL_CBWR is not AVX2,STRICT; MKL may pick paths by alignment\n")
    print(f"reference: {versions()}\n")

    problems: list[str] = []
    total = 0
    for variant in args.variants:
        try:
            checkpoint = resolve("qwen3_vl", variant, "torch-bf16")
        except WeightsError as error:
            print(f"{variant}: {error}")
            return 1
        source = args.reference if args.reference and variant == "2b" else None
        reference = Reference(source or repository(variant), None if source else COMMITS[variant])
        vendor = Generator(checkpoint, SPECS[variant])
        if args.fuzz:
            problems += [f"[{variant}] tokenizer"] * fuzz(args.fuzz, reference)
        if args.falsify:
            names = list(FALSIFICATIONS) if "all" in args.falsify else args.falsify
            print(f"{variant}: falsifying {len(names)}")
            missed = falsify(names, reference, lambda: vendor)
            if missed:
                problems.append(f"[{variant}] {missed} perturbation(s) not caught")
            continue
        chosen = cases(args.quick)
        if variant != "2b" and not args.quick:
            chosen = [chosen[0], chosen[1], chosen[3]]
        for number, case in enumerate(chosen):
            started = time.perf_counter()
            want = reference.run(case)
            failures = compare(case, want, vendor)
            if number == 0 or "sampled" in case.name:
                failures += [f"mozo: {f}" for f in against_mozo(variant, case, want)]
            stages = 3 + vendor.spec.vision.depth + len(vendor.spec.vision.deepstack) + 2 \
                + vendor.spec.text.layers + 1 + len(want["logits"]) + 2
            total += stages
            print(f"  {variant} {case.name}: {len(want['input_ids'][0])} prompt tokens, "
                  f"{len(want['ids'])} written, {stages} comparisons, "
                  f"{'identical' if not failures else f'{len(failures)} FAILED'} "
                  f"({time.perf_counter() - started:.0f}s)")
            problems += [f"[{variant}/{case.name}] {f}" for f in failures]
        del reference
    if problems:
        print(f"\n{len(problems)} FAILURES:")
        for problem in problems[:40]:
            print(f"  {problem}")
        return 1
    if not args.falsify:
        print(f"\n{total:,} comparisons, all identical.")
    else:
        print("\nevery perturbation was caught, at the stage it reaches first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

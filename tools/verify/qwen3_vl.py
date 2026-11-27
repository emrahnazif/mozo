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
after ``torch.manual_seed(seed)`` and mozo draws from a ``torch.Generator`` on the model's device
seeded the same, and the ids must agree exactly.

**What is pinned.** ``--device cpu`` (the default): float32 (the reference loads in float32
explicitly -- transformers 5 would otherwise run the stored bfloat16); SDPA attention; the parameters
re-allocated before comparing; ``MKL_CBWR=AVX2,STRICT`` is expected in the environment so MKL cannot
pick a code path by alignment. ``--device cuda``: float32 and bfloat16 (``torch-bf16``, what ``auto``
runs on CUDA), each side the whole model in that dtype -- the model card's own path,
``from_pretrained(dtype=...)`` with ``device_map`` on the GPU and the processor's output moved there.
Global settings are left as mozo leaves them (TF32 on for cuDNN, off for matmul; every SDPA
backend); the gate prints them and fails if loading or running either side changes one. The two
sides run one after the other in the same process (two 2B models in float32 do not fit a 12 GB
card together); a variant whose weights in a dtype do not fit the card at all is reported as not
gated. Each side runs every case ``--repeat`` times (2 on CUDA) and must repeat itself to the byte,
and the attention kernels each side launches (``torch.profiler``) must be the same. The reference's
version matters: 4.57.3 and 5.24.1 give the same words and different logits at every step. Run it
under the versions in :data:`PINNED`.

**The gate is shown to fail.** ``--falsify NAME`` (or ``all``) perturbs one thing in the vendored
package and requires the comparison to fail, at the stage that thing reaches first. Some exist only
on CUDA or only in bfloat16; ``all`` runs the ones that apply.

    python tools/verify/qwen3_vl.py                      # 2b everything, 4b on three fixtures
    python tools/verify/qwen3_vl.py --quick              # one small fixture per variant
    python tools/verify/qwen3_vl.py --falsify all        # every perturbation must be caught
    python tools/verify/qwen3_vl.py --fuzz 20000         # the tokenizer against the reference's
    python tools/verify/qwen3_vl.py --device cuda        # fp32 and bf16 on the GPU
    python tools/verify/qwen3_vl.py 2b --device cuda --falsify all --reference-cache DIR

``--reference DIR`` points at a local snapshot of the Hub repository (the pinned commit), or a
directory holding one per variant (``qwen3-vl-2b-instruct`` ...); otherwise it is read from the Hub at
that commit. ``--reference-cache DIR`` keeps the reference's outputs per (variant, device, dtype,
fixture set) and reuses them: upstream does not change between falsifiers.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import functools
import gc
import os
from dataclasses import dataclass
import random
import re
import subprocess
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

from common import FIXTURES  # noqa: E402
from fetch.qwen3_vl import COMMITS, repository  # noqa: E402

from mozo.vendors.qwen3_vl_deploy import (  # noqa: E402
    MAX_PIXELS, SPECS, Generator, Sampling)
from mozo.vendors.qwen3_vl_deploy import image as vimage  # noqa: E402
from mozo.vendors.qwen3_vl_deploy import language as vlanguage  # noqa: E402
from mozo.vendors.qwen3_vl_deploy import model as vmodel  # noqa: E402
from mozo.vendors.qwen3_vl_deploy import tokenizer as vtokenizer  # noqa: E402
from mozo.vendors.qwen3_vl_deploy import vision as vvision  # noqa: E402
from mozo.image import load_image  # noqa: E402
from mozo.vendors.qwen3_vl_deploy.config import Tokens  # noqa: E402
from mozo.weights import WeightsError, resolve  # noqa: E402

#: The versions this family's parity was established against. Printed, not enforced.
PINNED = {"torch": "2.13.0", "torchvision": "0.28.0", "transformers": "5.24.1",
          "tokenizers": "0.23.3"}

FIXTURE = FIXTURES / "example.jpg"


def versions() -> str:
    import tokenizers
    import torchvision
    import transformers
    got = {"torch": torch.__version__, "torchvision": torchvision.__version__,
           "transformers": transformers.__version__, "tokenizers": tokenizers.__version__}
    pinned = ", ".join(f"{k} {v}" + ("" if v.split("+")[0] == PINNED[k] else f" (pinned {PINNED[k]})")
                       for k, v in got.items())
    try:  # what from_pretrained's device_map needs on CUDA; it places weights, computes nothing
        import accelerate
        return f"{pinned}, accelerate {accelerate.__version__}"
    except ImportError:
        return pinned


@dataclass(frozen=True, eq=False)
class Case:
    """One fixture: an image, a prompt, a budget, how to choose, how many pixels.

    ``also_4b`` marks the fixtures the larger variant is checked on too.
    """

    name: str
    rgb: np.ndarray
    prompt: str
    steps: int
    sampling: Sampling = Sampling()
    max_pixels: int = MAX_PIXELS
    also_4b: bool = False


def stage_names(spec) -> list[str]:
    """Every stage compared before decoding, in the order the model computes them."""
    return (["pixel_values", "input_ids", "patch_embed"]
            + [f"vision_block_{i}" for i in range(spec.vision.depth)]
            + [f"deepstack_{i}" for i in range(len(spec.vision.deepstack))] + ["merger", "embeds"]
            + [f"text_layer_{i}" for i in range(spec.text.layers)] + ["final_norm"])


def cases(quick: bool) -> list[Case]:
    """The fixtures, awkward ones included (VENDORING §2.6)."""
    import io

    from PIL import Image

    photo = load_image(str(FIXTURE))
    small = np.array(Image.fromarray(photo).resize((640, 427), Image.BICUBIC))
    if quick:
        return [Case("small photo, greedy", small, "Describe this image.", 12)]
    # RGBA and greyscale go through mozo's own loader, so both sides are handed the same RGB array.
    rgba, grey = io.BytesIO(), io.BytesIO()
    Image.fromarray(small).convert("RGBA").save(rgba, "PNG")
    Image.fromarray(small).convert("L").save(grey, "JPEG")
    return [
        Case("photo, default 2.6 MP budget", photo, "Describe this image.", 32, also_4b=True),
        Case("photo, sampled (t=0.7, p=0.8, k=20, seed 3)", small, "Describe this image.", 24,
             Sampling(0.7, 0.8, 20, 3), also_4b=True),
        Case("photo, sampled (t=1.3, seed 11)", small, "What could happen next?", 16,
             Sampling(1.3, 1.0, 0, 11)),
        Case("stops on its own", small, "How many people are in this picture? Answer with a number.", 40,
             also_4b=True),
        Case("tiny 9x13, scaled up to the pixel floor", photo[:13, :9], "What colour is this?", 12),
        Case("strip 2000x120", np.array(Image.fromarray(photo).resize((2000, 120))),
             "Describe this image.", 12),
        Case("512x512, no resize", photo[:512, :512], "What is here?", 12),
        Case("above the budget (262,144 px)", photo, "Describe this image.", 12, max_pixels=262_144),
        Case("RGBA PNG", load_image(rgba.getvalue()), "Describe this image.", 8),
        Case("greyscale JPEG", load_image(grey.getvalue()), "Describe this image.", 8),
        Case("CJK, emoji, combining mark, newline", small,
             "这张图片里有什么？😀 Café\u0301 'S\nanswer briefly", 16),
        Case("long prompt", small, " ".join(["Describe every object you can see, its colour, and "
                                             "where it is in the frame."] * 20), 8),
    ]


# --- the environment ------------------------------------------------------------------------------

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def environment(device: str) -> str:
    """torch, CUDA, cuDNN, the GPU and its driver: printed on every run."""
    if device != "cuda":
        return f"torch {torch.__version__}, CPU, {torch.get_num_threads()} threads"
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, check=True).stdout.split()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        driver = "unknown"
    return (f"torch {torch.__version__}, CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}, "
            f"{torch.cuda.get_device_name(0)}, driver {driver}")


def global_settings() -> dict:
    """Every process-wide setting that can change a kernel's arithmetic. Compared, never set."""
    b = torch.backends
    return {
        "matmul TF32": b.cuda.matmul.allow_tf32, "cuDNN TF32": b.cudnn.allow_tf32,
        "float32 matmul precision": torch.get_float32_matmul_precision(),
        "fp16 reduced-precision reduction": b.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "bf16 reduced-precision reduction": b.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "cuDNN benchmark": b.cudnn.benchmark, "cuDNN deterministic": b.cudnn.deterministic,
        "deterministic algorithms": torch.are_deterministic_algorithms_enabled(),
        "SDPA flash/efficient/math/cuDNN": (b.cuda.flash_sdp_enabled(), b.cuda.mem_efficient_sdp_enabled(),
                                            b.cuda.math_sdp_enabled(), b.cuda.cudnn_sdp_enabled()),
        "default dtype": str(torch.get_default_dtype()), "grad enabled": torch.is_grad_enabled(),
    }


#: The attention kernels' names, as the profiler reports them.
ATTENTION_KERNEL = re.compile(r"flash|fmha|attention|sdp|efficient|softmax", re.IGNORECASE)


def cuda_kernels(run) -> collections.Counter:
    """The CUDA kernels *run* launches, by name, with how many times each."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        run()
        torch.cuda.synchronize()
    return collections.Counter(e.name for e in prof.events()
                               if e.device_type == torch.autograd.DeviceType.CUDA)


def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _host(x):
    return x.detach().to("cpu", copy=True) if isinstance(x, torch.Tensor) else x


# --- the reference -------------------------------------------------------------------------------

class Reference:
    """transformers' model and processor, with hooks on every stage the vendor records.

    On CUDA it is loaded the way the model card loads it -- ``from_pretrained(dtype=...)`` with
    ``device_map`` on the GPU, the processor run on the host and its output moved to the model's
    device -- and every recorded tensor is copied to the host as it is caught.
    """

    def __init__(self, source: str, revision: str | None, device: str = "cpu",
                 dtype: torch.dtype = torch.float32) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        kw = {} if revision is None else {"revision": revision}
        self.device = device
        self.processor = AutoProcessor.from_pretrained(source, **kw)
        placement = {} if device == "cpu" else {"device_map": device}
        self.model = AutoModelForImageTextToText.from_pretrained(
            source, dtype=dtype, attn_implementation="sdpa", **placement, **kw).eval()
        if device == "cpu":  # re-allocated, so MKL sees the alignment a fresh tensor has
            with torch.no_grad():
                for parameter in self.model.parameters():
                    parameter.data = parameter.data.clone()
        self.caught: dict[str, torch.Tensor] = {}
        self.positions: list[torch.Tensor] = []
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
        # The positions the decoder is called with: the prefill's M-RoPE ids, then each step's.
        text.register_forward_pre_hook(self._position_ids, with_kwargs=True)

    def _pre(self, name):
        def hook(module, args, kwargs):
            if name not in self.caught:
                self.caught[name] = _host(args[0] if args else kwargs["hidden_states"])
        return hook

    def _post(self, name):
        def hook(module, args, out):
            if name not in self.caught:
                self.caught[name] = _host(out[0] if isinstance(out, tuple) else out)
        return hook

    def _position_ids(self, module, args, kwargs):
        positions = kwargs.get("position_ids")
        if positions is not None and positions.shape[0] == 4:  # a text row in front of T, H, W
            positions = positions[1:]
        self.positions.append(_host(positions))

    def static(self) -> dict:
        """What does not depend on the input: each parameter's ``requires_grad``, the rotary tables."""
        m = self.model.model
        return {"requires_grad": {n.removeprefix("model."): p.requires_grad
                                  for n, p in self.model.named_parameters() if not n.startswith("lm_head")},
                "inv_freq_vision": _host(m.visual.rotary_pos_emb.inv_freq),
                "inv_freq_text": _host(m.language_model.rotary_emb.inv_freq),
                "inv_freq_device": (m.visual.rotary_pos_emb.inv_freq.device.type,
                                    m.language_model.rotary_emb.inv_freq.device.type)}

    def run(self, case: Case) -> dict:
        """``generate`` as the model card calls it (``generate`` itself runs under ``no_grad``)."""
        from PIL import Image

        image = Image.fromarray(case.rgb)
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                 {"type": "text", "text": case.prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt",
                                size={"longest_edge": case.max_pixels, "shortest_edge": 65_536})
        pixel_values, input_ids = inputs["pixel_values"].clone(), inputs["input_ids"].clone()
        inputs = inputs.to(self.model.device)
        s = case.sampling
        how = (dict(do_sample=False, temperature=None, top_p=None, top_k=None) if s.greedy else
               dict(do_sample=True, temperature=s.temperature, top_p=s.top_p, top_k=s.top_k))
        self.caught, self.positions = {}, []
        if not s.greedy:
            torch.manual_seed(s.seed)
        out = self.model.generate(**inputs, max_new_tokens=case.steps, output_logits=True,
                                  return_dict_in_generate=True, **how)
        ids = out.sequences[0, input_ids.shape[1]:].tolist()
        return {"pixel_values": pixel_values, "input_ids": input_ids,
                "grid": tuple(inputs["image_grid_thw"][0].tolist()), **self.caught,
                "positions": self.positions[0], "rope_deltas": int(self.model.model.rope_deltas.item()),
                "decode_positions": self.positions[1:],
                "logits": [_host(step[0]) for step in out.logits], "ids": ids,
                "text": self.processor.decode(ids, skip_special_tokens=True)}


# --- the vendor ----------------------------------------------------------------------------------

class HostRecord(dict):
    """The vendor's record, every tensor copied to the host as it is written (the GPU stays free)."""

    def __setitem__(self, key, value):
        super().__setitem__(key, _host(value))


def vendor_static(vendor: Generator) -> dict:
    m = vendor.model
    return {"requires_grad": {n: p.requires_grad for n, p in m.named_parameters()},
            "inv_freq_vision": _host(m.visual.inv_freq), "inv_freq_text": _host(m.language_model.inv_freq),
            "inv_freq_device": (m.visual.inv_freq.device.type, m.language_model.inv_freq.device.type)}


def run_vendor(vendor: Generator, case: Case) -> dict:
    """The vendor's answer to *case*, with every stage and step, in the reference's layout."""
    record = HostRecord()
    calls: list[torch.Tensor] = []
    hook = vendor.model.language_model.register_forward_pre_hook(
        lambda module, args: calls.append(_host(args[1])))
    try:
        answer = vendor.generate(case.rgb, case.prompt, case.steps, case.sampling, case.max_pixels,
                                 record=record)
    except Exception as error:  # a perturbation can also break a shape
        return {"error": f"raised {type(error).__name__}: {error}"}
    finally:
        hook.remove()
    prefill = calls[0]
    return {**record, "positions": prefill,
            "rope_deltas": int(prefill.max()) + 1 - record["input_ids"].shape[1],
            "decode_positions": calls[1:], "ids": answer.ids, "text": answer.text, "answer": answer}


# --- comparing -----------------------------------------------------------------------------------

def _tensor_diff(name: str, a: torch.Tensor, b: torch.Tensor) -> list[str]:
    if a.shape != b.shape:
        return [f"{name}: shape {tuple(a.shape)} != {tuple(b.shape)}"]
    if a.dtype != b.dtype:
        return [f"{name}: dtype {a.dtype} != {b.dtype}"]
    if not torch.equal(a, b):
        return [f"{name}: max |d| {(a.double() - b.double()).abs().max().item():.3e}"]
    return []


def static_diff(got: dict, want: dict) -> list[str]:
    """``requires_grad`` per parameter and the rotary frequency tables."""
    failures = []
    if got["requires_grad"] != want["requires_grad"]:
        names = set(got["requires_grad"]) ^ set(want["requires_grad"])
        flipped = [n for n in got["requires_grad"]
                   if n in want["requires_grad"] and got["requires_grad"][n] != want["requires_grad"][n]]
        failures.append(f"requires_grad: {len(flipped)} parameter(s) differ, {len(names)} unmatched names")
    for name in ("inv_freq_vision", "inv_freq_text"):
        failures += _tensor_diff(name, got[name], want[name])
    if got["inv_freq_device"] != want["inv_freq_device"]:
        failures.append(f"inv_freq_device: {got['inv_freq_device']} != {want['inv_freq_device']}")
    return failures


def static_count() -> int:
    return 4  # requires_grad, the two tables, where they live


def diff(got: dict, want: dict, spec) -> list[str]:
    """Every difference between two runs of one case, first stage first."""
    if "error" in got:
        return [got["error"]]
    if got["grid"] != want["grid"]:
        return [f"grid {got['grid']} != {want['grid']}"]
    failures = []
    for stage in stage_names(spec):
        failures += _tensor_diff(stage, got[stage], want[stage])
        if stage == "embeds":  # the positions are computed next, before the first decoder layer
            failures += _tensor_diff("positions", got["positions"], want["positions"])
            if got["rope_deltas"] != want["rope_deltas"]:
                failures.append(f"rope_deltas: {got['rope_deltas']} != {want['rope_deltas']}")
    for step in range(min(len(got["logits"]), len(want["logits"]))):
        if not torch.equal(got["logits"][step], want["logits"][step]):
            d = (got["logits"][step] - want["logits"][step]).abs().max().item()
            failures.append(f"step {step} logits: max |d| {d:.3e}")
            break
    for step, (a, b) in enumerate(zip(got["decode_positions"], want["decode_positions"])):
        if a.shape != b.shape or not torch.equal(a, b):
            failures.append(f"step {step + 1} positions: {a.flatten().tolist()} != {b.flatten().tolist()}")
            break
    if got["ids"] != want["ids"]:
        first = next((i for i, (x, y) in enumerate(zip(got["ids"], want["ids"])) if x != y),
                     min(len(got["ids"]), len(want["ids"])))
        failures.append(f"generated ids differ from token {first}")
    if got["text"] != want["text"]:
        failures.append("decoded text differs")
    return failures


def comparisons(spec, want: dict) -> int:
    """How many things :func:`diff` compares for *want*."""
    return (len(stage_names(spec)) + 2 + len(want["logits"]) + len(want["decode_positions"])
            + 2)  # positions, rope_deltas ... ids, text


def against_mozo(variant: str, case: Case, want: dict, device: str, dtype: str) -> list[str]:
    """mozo's adapter, through the registry and the weights tree, must write the same answer.

    ``auto`` must pick ``torch-fp32`` on the CPU and ``torch-bf16`` on CUDA; CUDA fp32 is asked for.
    """
    import mozo
    from mozo.adapters.qwen3_vl import Qwen3VLPredictor

    expected = "torch-bf16" if dtype == "bf16" else "torch-fp32"
    if device == "cuda" and dtype == "fp32":
        model = Qwen3VLPredictor(variant, device=device, runtime="torch-fp32")
    else:
        model = mozo.get_model(f"qwen3_vl/{variant}", device=device)
    s = case.sampling
    got = model.predict(case.rgb, case.prompt, max_new_tokens=case.steps, temperature=s.temperature,
                        top_p=s.top_p, top_k=s.top_k, seed=s.seed, max_pixels=case.max_pixels)
    failures = []
    if model.runtime != expected:
        failures.append(f"chose {model.runtime} on {device}, expected {expected}")
    if got.text != want["text"]:
        failures.append("the adapter's text differs")
    finished = "stop" if want["ids"] and want["ids"][-1] in Tokens().stop else "length"
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
        try:
            yield
        finally:
            vtokenizer._word.cache_clear()


# --- perturbations that exist only on CUDA, or only in a half precision ---------------------------

@contextlib.contextmanager
def _setting(module, name, value):
    saved = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, saved)


@contextlib.contextmanager
def _sdpa_math():
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel([SDPBackend.MATH]):
        yield


@contextlib.contextmanager
def _frozen(vendor: Generator):
    """Every parameter at ``requires_grad=False`` (upstream's are left trainable)."""
    parameters = list(vendor.model.parameters())
    for parameter in parameters:
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter in parameters:
            parameter.requires_grad_(True)


@contextlib.contextmanager
def _generator_on_host():
    """``torch.multinomial`` drawn on the host from a CPU generator with the same seed."""
    real = torch.multinomial
    host: dict[int, torch.Generator] = {}

    def multinomial(probs, num_samples, replacement=False, *, generator=None, out=None):
        g = host.setdefault(id(generator), torch.Generator().manual_seed(generator.initial_seed()))
        return real(probs.cpu(), num_samples, replacement, generator=g).to(probs.device)

    with mock.patch.object(torch, "multinomial", multinomial):
        yield


def _rmsnorm_in_half(self, x):
    """The text RMSNorm in the activations' dtype rather than in float32."""
    variance = x.pow(2).mean(-1, keepdim=True)
    return self.weight * (x * torch.rsqrt(variance + self.eps))


def _vision_rotary_in_half(q, k, cos, sin):
    """The vision rotary applied in the tower's dtype rather than in float32."""
    cos, sin = cos.unsqueeze(-2).to(q.dtype), sin.unsqueeze(-2).to(q.dtype)
    return (q * cos) + (vvision.rotate_half(q) * sin), (k * cos) + (vvision.rotate_half(k) * sin)


def _vision_rotary_on_host(original):
    """The vision tower's cos/sin computed on the host and moved to the GPU."""
    def rotary(self, x, positions):
        cos, sin = original(self, torch.empty(0, dtype=x.dtype), positions.cpu())
        return cos.to(x.device), sin.to(x.device)
    return rotary


def _text_rotary_on_host(original):
    """The text decoder's cos/sin computed on the host and moved to the GPU."""
    def rotary(self, x, positions):
        cos, sin = original(self, torch.empty(0, dtype=x.dtype), positions.cpu())
        return cos.to(x.device), sin.to(x.device)
    return rotary


ANY, CUDA, FP32_CUDA, HALF = "any", "cuda", "fp32-cuda", "half"


def _applies(where: str, device: str, dtype: str) -> bool:
    return {ANY: True, CUDA: device == "cuda", FP32_CUDA: device == "cuda" and dtype == "fp32",
            HALF: dtype != "fp32"}[where]


#: name -> (perturbation of the loaded vendor, the fixture that exposes it, the stage that must
#: fail first, where it applies).
FALSIFICATIONS = {
    "no-antialias": (lambda v: mock.patch.object(vimage, "tvF", _no_antialias()), "small", "pixel_values", ANY),
    "resize-floor": (lambda v: mock.patch.object(vimage, "smart_resize", _smart_resize_floor), "small", "grid",
                     ANY),
    "pos-embed-half-pixel": (lambda v: mock.patch.object(vvision, "_axis_taps", _axis_taps_half_pixel),
                             "small", "patch_embed", ANY),
    "vision-gelu-exact": (lambda v: mock.patch.object(vvision.MLP, "forward", _vision_mlp_exact),
                          "small", "vision_block_0", ANY),
    "merger-norm-eps": (lambda v: mock.patch.object(vvision.PatchMerger, "forward", _merger_norm_eps),
                        "small", "deepstack_0", ANY),
    "vision-rope-positions": (lambda v: mock.patch.object(vvision.VisionTower, "_rotary",
                                                          _scaled_rotary(vvision.VisionTower._rotary)),
                              "small", "vision_block_0", ANY),
    "mrope-blocks": (lambda v: mock.patch.object(vlanguage.TextDecoder, "_rotary", _recompose_blocks),
                     "small", "text_layer_0", ANY),
    "deepstack-one-layer-late": (lambda v: _deepstack_shifted(), "small", "text_layer_0", ANY),
    "text-rope-theta": (lambda v: mock.patch.object(vlanguage.TextDecoder, "_rotary", _text_theta(1_000_000.0)),
                        "small", "text_layer_0", ANY),
    "rmsnorm-eps": (lambda v: mock.patch.object(vlanguage.RMSNorm, "forward", _rmsnorm_eps(1e-5)),
                    "small", "text_layer_0", ANY),
    # A sampled token that differs is fed back, so the first difference is a later step's logits.
    "top-p-before-top-k": (lambda v: _warp_order_swapped(), "sampled", "step", ANY),
    "seed-off-by-one": (lambda v: _seed_off_by_one(), "open", "step", ANY),
    "tokenizer-no-nfc": (lambda v: _no_nfc(), "nfc", "input_ids", ANY),
    # requires_grad changes which matmul path torch takes on CUDA for some shapes (HQ-SAM).
    "requires-grad-off": (_frozen, "small", "requires_grad", ANY),
    # CUDA only: the hardware's own choices, and work moved between the host and the device.
    "matmul-tf32-on": (lambda v: _setting(torch.backends.cuda.matmul, "allow_tf32", True),
                       "small", "vision_block_0", FP32_CUDA),
    "cudnn-tf32-off": (lambda v: _setting(torch.backends.cudnn, "allow_tf32", False),
                       "small", "patch_embed", FP32_CUDA),
    "sdpa-math": (lambda v: _sdpa_math(), "small", "vision_block_0", CUDA),
    # Transcendentals on the host: CUDA's cos/sin differ from the CPU's in the last bit. In bfloat16
    # the text tables are rounded to bfloat16 (as upstream does), which hides that; the vision
    # tables stay float32 and multiply float32 queries, so they show it in both.
    "vision-rope-on-host": (lambda v: mock.patch.object(vvision.VisionTower, "_rotary",
                                                        _vision_rotary_on_host(vvision.VisionTower._rotary)),
                            "small", "vision_block_0", CUDA),
    "text-rope-on-host": (lambda v: mock.patch.object(vlanguage.TextDecoder, "_rotary",
                                                      _text_rotary_on_host(vlanguage.TextDecoder._rotary)),
                          "small", "text_layer_0", FP32_CUDA),
    "sampling-on-host": (lambda v: _generator_on_host(), "open", "step", CUDA),
    # Half precision only: where upstream computes in float32 inside a bfloat16 model.
    "rmsnorm-in-half": (lambda v: mock.patch.object(vlanguage.RMSNorm, "forward", _rmsnorm_in_half),
                        "small", "text_layer_0", HALF),
    "vision-rotary-in-half": (lambda v: mock.patch.object(vvision, "_apply_rotary", _vision_rotary_in_half),
                              "small", "vision_block_0", HALF),
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
    """The vision tower's rotary positions scaled by 1 + 1e-7."""
    return lambda self, x, positions: original(self, x, positions * 1.0000001)


def _rmsnorm_eps(eps):
    """The text decoder's RMSNorm (and its q/k norms) at another epsilon."""
    def forward(self, x):
        dtype = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        return self.weight * x.to(dtype)
    return forward


def falsifier_cases() -> dict[str, Case]:
    from PIL import Image

    photo = load_image(str(FIXTURE))
    small = np.array(Image.fromarray(photo).resize((320, 214), Image.BICUBIC))
    return {
        "small": Case("small", small, "Describe this image.", 4),
        # A hot distribution cut by both warpers, so their order changes the kept set at most steps
        # (top_k=3/top_p=0.6 showed it on the CPU and in CUDA fp32, never in 12 bf16 tokens; seed 2
        # parts within 4-5 tokens on the CPU and in CUDA fp32 and bf16).
        "sampled": Case("sampled", small, "Describe this image.", 12, Sampling(2.0, 0.8, 10, 2)),
        # The whole distribution at a high temperature, so two seeds part within a few tokens.
        "open": Case("open", small, "Describe this image.", 12, Sampling(1.5, 1.0, 0, 5)),
        "nfc": Case("nfc", small, "Café or café?", 3),
    }


def falsify(names: list[str], want: dict, vendor: Generator) -> int:
    """Each perturbation must make the comparison fail, at the stage it reaches first.

    *want* holds the reference's ``static`` and its run of each falsifier fixture. The unperturbed
    vendor must match it first, or a "caught" would mean nothing.
    """
    fixtures = falsifier_cases()
    needed = sorted({FALSIFICATIONS[name][1] for name in names})
    baseline = static_diff(vendor_static(vendor), want["static"])
    for fixture in needed:
        baseline += [f"{fixture}: {f}" for f in diff(run_vendor(vendor, fixtures[fixture]),
                                                     want["results"][fixture], vendor.spec)]
    if baseline:
        print(f"  the unperturbed vendor does not match: {baseline[:3]}")
        return len(names)
    missed = 0
    for name in names:
        perturbation, fixture, stage, _ = FALSIFICATIONS[name]
        with perturbation(vendor):
            failures = static_diff(vendor_static(vendor), want["static"])
            failures += diff(run_vendor(vendor, fixtures[fixture]), want["results"][fixture], vendor.spec)
        first = failures[0] if failures else "nothing"
        caught = bool(failures) and first.startswith(stage)
        missed += not caught
        print(f"  {'caught' if caught else 'MISSED'} {name:26s} first failure: {first[:90]}"
              + (f" ({len(failures)} in all)" if failures else "")
              + ("" if caught else f"   (expected {stage})"))
    return missed


def fuzz(count: int, tok) -> int:
    """The tokenizer on *count* random strings, and its decoder on random ids, against the reference."""
    ours = vtokenizer.Tokenizer()
    rng = random.Random(0)
    pools = ["abcxyz ABC'sStTmMdDlLreREve", "0123456789٠١٢३४५", " \t\n\r  　\u0085 ",
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


# --- running -------------------------------------------------------------------------------------

def profiled_case() -> Case:
    """The case whose CUDA kernels are listed on both sides: small, so the profile stays short."""
    from PIL import Image

    photo = load_image(str(FIXTURE))
    small = np.array(Image.fromarray(photo).resize((640, 427), Image.BICUBIC))
    return Case("profiled", small, "Describe this image.", 3)


def reference_source(path: str | None, variant: str) -> tuple[str, str | None]:
    """A local snapshot (one directory, or one per variant inside it), else the Hub at the pinned commit."""
    if path:
        per_variant = Path(path) / f"qwen3-vl-{variant}-instruct"
        if per_variant.is_dir():
            return str(per_variant), None
        if variant == "2b" and (Path(path) / "config.json").is_file():
            return path, None
    return repository(variant), COMMITS[variant]


def signature(chosen: dict[str, Case]) -> list:
    """What a cached reference run must have been made from: every case's inputs, the image hashed."""
    import hashlib
    return [(key, case.prompt, case.steps, case.sampling, case.max_pixels,
             hashlib.sha256(np.ascontiguousarray(case.rgb).tobytes()).hexdigest(), case.rgb.shape)
            for key, case in sorted(chosen.items())]


def reference_outputs(args, variant: str, device: str, dtype: str, which: str,
                      chosen: dict[str, Case]) -> dict:
    """The reference's run of every case in *chosen* (each ``--repeat`` times), cached when asked.

    Also what does not depend on the input, the global settings after it ran, and on CUDA the
    kernels it launches for :func:`profiled_case`. The model is released before returning.
    """
    spec = SPECS[variant]
    path = Path(args.reference_cache) / f"{variant}-{device}-{dtype}-{which}.pt" if args.reference_cache else None
    if path is not None and path.is_file():
        cached = torch.load(path, weights_only=False)
        if (cached["versions"], cached["environment"], cached["repeat"], cached["cases"]) == (
                versions(), environment(device), args.repeat, signature(chosen)):
            print(f"  reference: reused {path}")
            return cached
        print(f"  reference: {path} was made otherwise; running again")
    started = time.perf_counter()
    source, revision = reference_source(args.reference, variant)
    reference = Reference(source, revision, device, DTYPES[dtype])
    out = {"versions": versions(), "environment": environment(device), "repeat": args.repeat,
           "cases": signature(chosen), "static": reference.static(), "results": {}, "repeats": [], "repeat_comparisons": 0,
           "kernels": None}
    for key, case in chosen.items():
        first = reference.run(case)
        for _ in range(args.repeat - 1):
            out["repeats"] += [f"reference/{key}: {f}" for f in diff(reference.run(case), first, spec)]
            out["repeat_comparisons"] += comparisons(spec, first)
        out["results"][key] = first
    if device == "cuda":
        out["kernels"] = cuda_kernels(functools.partial(reference.run, profiled_case()))
    out["settings"] = global_settings()
    del reference
    free_gpu()
    print(f"  reference: {len(chosen)} case(s) x {args.repeat} in {time.perf_counter() - started:.0f}s")
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, path)
    return out


def fits(checkpoint: Path, device: str, dtype: str) -> bool:
    """Whether the weights alone, in *dtype*, leave room on the GPU (the bfloat16 file is half fp32's)."""
    if device != "cuda":
        return True
    weights = Path(checkpoint).stat().st_size * (2 if dtype == "fp32" else 1)
    return weights < 0.9 * torch.cuda.get_device_properties(0).total_memory


def attention_kernels(kernels: collections.Counter) -> dict[str, int]:
    return {name: n for name, n in sorted(kernels.items()) if ATTENTION_KERNEL.search(name)}


def gate(args, variant: str, device: str, dtype: str, settings: dict) -> tuple[list[str], int, int]:
    """One variant in one dtype on one device: (problems, comparisons, repeat comparisons)."""
    spec = SPECS[variant]
    checkpoint = resolve("qwen3_vl", variant, "torch-bf16")
    every = cases(args.quick)
    if variant != "2b" and not args.quick:
        every = [case for case in every if case.also_4b]
    chosen = {case.name: case for case in every}
    want = reference_outputs(args, variant, device, dtype, "quick" if args.quick else "full", chosen)
    problems = list(want["repeats"])
    if want["settings"] != settings:
        problems.append(f"the reference changed global settings: {want['settings']}")

    vendor = Generator(checkpoint, spec, device, DTYPES[dtype])
    problems += [f"static: {f}" for f in static_diff(vendor_static(vendor), want["static"])]
    total, repeated = static_count(), want["repeat_comparisons"]
    for key, case in chosen.items():
        started = time.perf_counter()
        got = run_vendor(vendor, case)
        failures = diff(got, want["results"][key], spec)
        for _ in range(args.repeat - 1):
            failures += [f"vendor repeat: {f}" for f in diff(run_vendor(vendor, case), got, spec)]
            repeated += comparisons(spec, want["results"][key])
        stages = comparisons(spec, want["results"][key])
        total += stages
        r = want["results"][key]
        print(f"  {variant} {dtype} {case.name}: {len(r['input_ids'][0])} prompt tokens, "
              f"{len(r['ids'])} written, {stages} comparisons, "
              f"{'identical' if not failures else f'{len(failures)} FAILED'} "
              f"({time.perf_counter() - started:.0f}s)")
        problems += [f"[{variant}/{dtype}/{case.name}] {f}" for f in failures]
    if device == "cuda":
        ours = cuda_kernels(functools.partial(run_vendor, vendor, profiled_case()))
        theirs = want["kernels"]
        a, b = attention_kernels(ours), attention_kernels(theirs)
        print(f"  attention kernels (profiled case), reference: {b}")
        if a != b:
            problems.append(f"[{variant}/{dtype}] attention kernels differ: vendor {a}")
        only_ours, only_theirs = set(ours) - set(theirs), set(theirs) - set(ours)
        print(f"  other kernels: {len(set(ours) & set(theirs))} shared, {len(only_ours)} vendor-only, "
              f"{len(only_theirs)} reference-only")
        for name in sorted(only_ours):
            print(f"    vendor only:    {name[:110]}")
        for name in sorted(only_theirs):
            print(f"    reference only: {name[:110]}")
        total += 1
    if global_settings() != settings:
        problems.append(f"[{variant}/{dtype}] global settings changed: {global_settings()}")
    del vendor
    free_gpu()

    for number, (key, case) in enumerate(chosen.items()):
        if number == 0 or "sampled" in case.name:
            problems += [f"[{variant}/{dtype}/{case.name}] mozo: {f}"
                         for f in against_mozo(variant, case, want["results"][key], device, dtype)]
            import mozo
            mozo.clear()
            free_gpu()
    return problems, total, repeated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variants", nargs="*", default=list(SPECS))
    parser.add_argument("--reference", help="local snapshot directory of the Hub repository")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dtype", nargs="+", choices=list(DTYPES),
                        help="default: fp32 on the CPU; fp32 and bf16 on CUDA")
    parser.add_argument("--repeat", type=int, help="runs per case per side (default 1 on CPU, 2 on CUDA)")
    parser.add_argument("--reference-cache", metavar="DIR", help="keep and reuse the reference's outputs")
    parser.add_argument("--quick", action="store_true", help="one small fixture per variant")
    parser.add_argument("--falsify", nargs="+", metavar="NAME",
                        choices=[*FALSIFICATIONS, "all"], help="perturbations that must be caught")
    parser.add_argument("--fuzz", type=int, default=0, metavar="N", help="tokenizer strings to fuzz")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    dtypes = args.dtype or (["fp32", "bf16"] if args.device == "cuda" else ["fp32"])
    if args.device == "cpu" and dtypes != ["fp32"]:
        parser.error("bf16 is a CUDA runtime only; the CPU gate is fp32")
    if args.repeat is None:
        args.repeat = 2 if args.device == "cuda" else 1
    torch.set_num_threads(args.threads)
    settings = global_settings()  # before transformers is imported
    if args.device == "cpu" and os.environ.get("MKL_CBWR") != "AVX2,STRICT":
        print("warning: MKL_CBWR is not AVX2,STRICT; MKL may pick paths by alignment\n")
    print(f"environment: {environment(args.device)}")
    print(f"reference: {versions()}")
    print(f"global settings: {settings}\n")

    problems: list[str] = []
    total = repeated = 0
    skipped = []
    for variant in args.variants:
        try:
            checkpoint = resolve("qwen3_vl", variant, "torch-bf16")
        except WeightsError as error:
            print(f"{variant}: {error}")
            return 1
        if args.fuzz:
            from transformers import AutoProcessor
            source, revision = reference_source(args.reference, variant)
            tok = AutoProcessor.from_pretrained(source, **({} if revision is None else {"revision": revision})).tokenizer
            problems += [f"[{variant}] tokenizer"] * fuzz(args.fuzz, tok)
        for dtype in dtypes:
            if not fits(checkpoint, args.device, dtype):
                skipped.append(f"{variant} {dtype}")
                print(f"{variant} {dtype}: not gated -- its weights alone do not fit this GPU")
                continue
            print(f"{variant} {dtype} on {args.device}:")
            if args.falsify:
                names = [n for n in (list(FALSIFICATIONS) if "all" in args.falsify else args.falsify)
                         if _applies(FALSIFICATIONS[n][3], args.device, dtype)]
                fixtures = falsifier_cases()
                needed = {FALSIFICATIONS[name][1] for name in names}
                want = reference_outputs(args, variant, args.device, dtype, "falsify",
                                         {k: v for k, v in fixtures.items() if k in needed})
                vendor = Generator(checkpoint, SPECS[variant], args.device, DTYPES[dtype])
                print(f"  falsifying {len(names)}")
                missed = falsify(names, want, vendor)
                del vendor
                free_gpu()
                if missed:
                    problems.append(f"[{variant}/{dtype}] {missed} perturbation(s) not caught")
                continue
            found, n, r = gate(args, variant, args.device, dtype, settings)
            problems += found
            total += n
            repeated += r
    if skipped:
        print(f"\nnot gated on this GPU: {', '.join(skipped)}")
    if problems:
        print(f"\n{len(problems)} FAILURES:")
        for problem in problems[:40]:
            print(f"  {problem}")
        return 1
    if args.falsify:
        print("\nevery perturbation was caught, at the stage it reaches first.")
    elif not args.fuzz or total:
        print(f"\n{total:,} comparisons, all identical"
              + (f"; each side repeated itself on {repeated:,} more." if repeated else "."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

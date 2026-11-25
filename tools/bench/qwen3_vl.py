#!/usr/bin/env python3
"""Qwen3-VL: how fast, how much memory, and what reduced precision does to the answers.

Two measurements.

``latency`` -- one image and one prompt, a fixed number of new tokens, greedy: mozo's runtimes
against ``transformers`` (when installed) on the same device, the same weights and inputs,
interleaved across trials so drift moves both sides together, the device synchronised. Prefill
(image through to the first token) is reported apart from the per-token decode, because the
image is most of the cost.

    python tools/bench/qwen3_vl.py latency 2b --device cuda --runtimes torch-fp32 torch-bf16

``vqa`` -- the task's accuracy, for deciding whether a reduced precision may be offered: VQAv2
questions on COCO images, the short-answer prompt Qwen evaluates VQAv2 with, greedy, scored with
VQA accuracy (``min(1, humans who gave that answer / 3)``, after VQA's answer normalisation,
simplified to lower-case, punctuation and articles). Each runtime is also compared with fp32
answer by answer, which is what a user switching precision would notice.

    python tools/bench/qwen3_vl.py vqa 2b --device cuda --runtimes torch-fp32 torch-bf16 \\
        --subset data/vqa/subset300.json --images data/vqa/images --out results.json

The subset is built from VQAv2's val annotations (CC BY 4.0) on COCO val2017 images, and is
measurement data only: nothing of it ships.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from common import synchronise  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "images" / "example.jpg"
PROMPT = "Describe this image."
VQA_PROMPT = "{question}\nAnswer the question using a single word or phrase."


def _peak(device: str) -> str:
    if device.startswith("cuda"):
        return f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
    import resource
    return f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB RSS"


def _mozo(variant: str, device: str, runtime: str):
    """mozo's adapter at *runtime*; ``torch-fp16``, which mozo does not offer, measured through the
    vendored package with the adapter's weights."""
    from mozo.adapters.qwen3_vl import Qwen3VLPredictor

    if runtime != "torch-fp16":
        return Qwen3VLPredictor(variant, device=device, runtime=runtime)
    from mozo.generation import Generation
    from mozo.image import load_image
    from mozo.vendors.qwen3_vl_deploy import SPECS, Generator
    from mozo.weights import resolve

    generator = Generator(resolve("qwen3_vl", variant, "torch-bf16"), SPECS[variant], device,
                          torch.float16)

    class Half:
        def predict(self, image, prompt, max_new_tokens=128):
            a = generator.generate(load_image(image), prompt, max_new_tokens)
            return Generation(a.text, a.finish_reason, a.prompt_tokens, a.generated_tokens)

    return Half()


def latency(args) -> None:
    """Each side loaded, warmed, timed and freed in turn: a 12 GB GPU does not hold two at once,
    so the sides are not interleaved -- run it on a quiet machine."""
    from PIL import Image

    rgb = np.array(Image.open(FIXTURE).convert("RGB"))

    def mozo_side(runtime):
        model = _mozo(args.variant, args.device, runtime)
        return model, lambda steps: model.predict(rgb, PROMPT, max_new_tokens=steps)

    def reference_side(dtype):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        proc = AutoProcessor.from_pretrained(args.reference)
        ref = AutoModelForImageTextToText.from_pretrained(
            args.reference, dtype=getattr(torch, dtype), attn_implementation="sdpa").to(args.device).eval()
        image = Image.fromarray(rgb)
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                 {"type": "text", "text": PROMPT}]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[image], return_tensors="pt",
                      size={"longest_edge": 2_621_440, "shortest_edge": 65_536}).to(args.device)

        def generate(steps):
            with torch.inference_mode():
                return ref.generate(**inputs, max_new_tokens=steps, min_new_tokens=steps,
                                    do_sample=False, temperature=None, top_p=None, top_k=None)
        return ref, generate

    sides = [(f"mozo {r}", lambda r=r: mozo_side(r)) for r in args.runtimes]
    if args.reference:
        sides += [(f"transformers {d}", lambda d=d: reference_side(d)) for d in args.reference_dtypes]
    print(f"{args.variant} on {args.device}, {args.tokens} new tokens, median of {args.trials}:")
    for name, build in sides:
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        try:
            holder, run = build()
            run(1)
            prefill, total = [], []
            for _ in range(args.trials):
                for steps, into in ((1, prefill), (args.tokens, total)):
                    synchronise(args.device)
                    started = time.perf_counter()
                    run(steps)
                    synchronise(args.device)
                    into.append(time.perf_counter() - started)
        except torch.OutOfMemoryError:
            print(f"  {name:24s} did not fit")
            holder = run = None
            continue
        p, t = statistics.median(prefill), statistics.median(total)
        print(f"  {name:24s} prefill {p * 1000:8.0f} ms   total {t:7.2f} s   "
              f"decode {(t - p) / max(1, args.tokens - 1) * 1000:6.1f} ms/token   peak {_peak(args.device)}",
              flush=True)
        del holder, run


def _normalise(answer: str) -> str:
    answer = answer.lower().strip().rstrip(".")
    answer = re.sub(r"[^\w\s]", "", answer)
    return " ".join(w for w in answer.split() if w not in {"a", "an", "the"})


def _accuracy(answer: str, humans: list[str]) -> float:
    got = _normalise(answer)
    return min(1.0, sum(_normalise(h) == got for h in humans) / 3)


def vqa(args) -> None:
    from mozo.image import load_image

    questions = json.loads(Path(args.subset).read_text())[: args.limit]
    images = Path(args.images)
    results = {}
    for runtime in args.runtimes:
        model = _mozo(args.variant, args.device, runtime)
        answers, started = [], time.perf_counter()
        for q in questions:
            g = model.predict(load_image(str(images / q["image"])), VQA_PROMPT.format(question=q["question"]),
                              max_new_tokens=16)
            answers.append(g.text)
        took = time.perf_counter() - started
        score = statistics.mean(_accuracy(a, q["answers"]) for a, q in zip(answers, questions))
        results[runtime] = {"answers": answers, "accuracy": score, "seconds": took}
        print(f"  {args.variant} {runtime:11s} VQA accuracy {score * 100:5.1f}% on {len(questions)} "
              f"questions, {took / len(questions):.2f} s each, peak {_peak(args.device)}", flush=True)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    base = args.baseline if args.baseline in results else None
    if base:
        for runtime, r in results.items():
            if runtime == base:
                continue
            same = sum(a == b for a, b in zip(r["answers"], results[base]["answers"]))
            print(f"  {runtime} vs {base}: {same}/{len(questions)} answers identical")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"variant": args.variant, "device": args.device, "questions": len(questions),
             "results": results}, indent=1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    lat = sub.add_parser("latency")
    lat.add_argument("variant")
    lat.add_argument("--device", default="cpu")
    lat.add_argument("--runtimes", nargs="+", default=["torch-fp32"])
    lat.add_argument("--tokens", type=int, default=32)
    lat.add_argument("--trials", type=int, default=3)
    lat.add_argument("--reference", help="a local snapshot of the Hub repository, to time transformers")
    lat.add_argument("--reference-dtypes", nargs="+", default=["float32"])
    q = sub.add_parser("vqa")
    q.add_argument("variant")
    q.add_argument("--device", default="cuda")
    q.add_argument("--runtimes", nargs="+", default=["torch-fp32", "torch-bf16"])
    q.add_argument("--baseline", default="torch-fp32")
    q.add_argument("--subset", required=True)
    q.add_argument("--images", required=True)
    q.add_argument("--limit", type=int, default=300)
    q.add_argument("--out")
    args = parser.parse_args()
    (latency if args.mode == "latency" else vqa)(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Measure what Granite Embedding R2 costs to serve, against the implementation it was extracted from.

Bootstrap tooling; never ships. ``tools/verify/granite_embedding_r2.py`` answers whether mozo
returns the same vectors as ``transformers``; this answers what it costs to get them. The results
are recorded in the vendor's ``PROVENANCE.md``, under *Speed* -- one place, so a re-run updates
one table.

    python tools/bench/granite_embedding_r2.py
    python tools/bench/granite_embedding_r2.py --devices cpu mps --reference

**Both sides run the same weights**, on the same device, from text to a unit vector: tokenizer,
model and normalisation, because that is the work a caller asks for. The reference is
``transformers``' tokenizer and ``ModernBertModel`` in float32 with SDPA attention, the
configuration the gate compares against. Any gap is this extraction against ``transformers``.

**How it measures.** Every timing synchronises the device, because Metal and CUDA queue work
asynchronously. Each side is warmed once per length, then the two are interleaved across trials so
thermal drift moves both together, and each side's median is taken over all its runs. The
tokenizer is timed **cold** -- its word cache cleared before every run -- because a document being
indexed is new text, and the warm figure measures dictionary lookups.

**The numbers are one machine's.** Per-token cost is not flat and its shape depends on the device.
Measure where you deploy. Nothing here is a pass/fail gate.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault("MOZO_BASE_URL", f"file://{ROOT / 'weights'}")

from common import median_ms, synchronise  # noqa: E402
from fetch.granite_embedding_r2 import COMMITS, repository  # noqa: E402
from verify.granite_embedding_r2 import prefix_within  # noqa: E402

from mozo.vendors.granite_embedding_r2_deploy import SPECS, Encoder  # noqa: E402
from mozo.vendors.granite_embedding_r2_deploy.tokenizer import Tokenizer, _word  # noqa: E402
from mozo.weights import WeightsError, resolve  # noqa: E402

#: Input lengths to time, in tokens: a search query, the usual retrieval chunks, and the cap.
LENGTHS = (32, 512, 1024, 2048, 8192)

#: How many short queries to time as one call: a batch on the CPU, one at a time elsewhere.
QUERIES = 64


def documents(tokenizer) -> dict[int, str]:
    """Documentation text for each of :data:`LENGTHS`, exactly that many tokens long.

    Built once: the search re-tokenizes, and one tokenizer word cache warmed by it is cleared before
    anything about the tokenizer is timed.
    """
    words = " ".join((ROOT / "docs" / name).read_text(encoding="utf-8")
                     for name in ("models.md", "workflows.md")).split(" ")
    out = {}
    for tokens in LENGTHS:
        text = prefix_within(tokenizer, words, tokens)
        out[tokens] = text + " x" * (tokens - len(tokenizer.encode(text)))  # " x" is one token
    return out


def reference_for(variant: str):
    """``transformers``' model and tokenizer for *variant*, loaded once, on the CPU."""
    from transformers import AutoModel, AutoTokenizer

    repo, commit = repository(variant), COMMITS[variant]
    model = AutoModel.from_pretrained(repo, revision=commit, dtype=torch.float32,
                                      attn_implementation="sdpa").eval()
    return model, AutoTokenizer.from_pretrained(repo, revision=commit)


def interleaved(sides: dict, device: str, runs: int, trials: int) -> dict[str, float]:
    """Each side's median over ``trials * runs`` timings, warmed once and taken in turns."""
    for work in sides.values():
        work()
        synchronise(device)
    timings = {name: [] for name in sides}
    for _ in range(trials):
        for name, work in sides.items():
            timings[name].append(median_ms(work, device, runs=runs, warmup=0))
    return {name: statistics.median(values) for name, values in timings.items()}


def cold(encode) -> float:
    """Median time to tokenize with the word cache emptied first, in milliseconds."""
    timings = []
    for _ in range(20):
        _word.cache_clear()
        start = time.perf_counter()
        encode()
        timings.append((time.perf_counter() - start) * 1000)
    return statistics.median(timings)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="+", default=list(SPECS))
    parser.add_argument("--devices", nargs="+", default=["cpu"])
    parser.add_argument("--reference", action="store_true",
                        help="also time transformers on the same weights, interleaved")
    parser.add_argument("--trials", type=int, default=3, help="interleaved trials per length")
    args = parser.parse_args()

    # Built before anything is timed, so no device's first-text figure pays for the tables.
    texts = documents(Tokenizer())
    queries = [f"how do I {verb} a video at {n} frames a second?"
               for n in range(QUERIES // 4) for verb in ("run", "read", "track", "blur")]

    for variant in args.variants:
        try:
            checkpoint = resolve("granite_embedding_r2", variant, "torch-fp32")
        except WeightsError as error:
            print(f"{variant}: {error}")
            continue
        reference = reference_for(variant) if args.reference else None

        for device in args.devices:
            print(f"\n{variant} on {device} ({torch.get_num_threads()} threads, "
                  f"torch {torch.__version__})")
            ours = Encoder(checkpoint, SPECS[variant], device=device)
            started = time.perf_counter()
            ours.encode_text("warm up")
            synchronise(device)
            print(f"  load + first text: {(time.perf_counter() - started) * 1000:.0f} ms")

            if reference:
                model, tokenizer = reference[0].to(device), reference[1]

                def theirs(text):
                    ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
                    with torch.inference_mode():
                        hidden = model(input_ids=ids).last_hidden_state[:, 0]
                    return F.normalize(hidden, dim=1).cpu()

            print(f"  {'tokens':>6} {'mozo':>10} {'per token':>10}"
                  + (f" {'reference':>10} {'mozo/ref':>9}" if reference else ""))
            for tokens, text in texts.items():
                sides = {"mozo": lambda: ours.encode_text(text)}
                if reference:
                    sides["reference"] = lambda: theirs(text)
                times = interleaved(sides, device, runs=max(2, 2048 // tokens), trials=args.trials)
                line = f"  {tokens:6d} {times['mozo']:8.1f}ms {times['mozo'] / tokens * 1000:8.1f}us"
                if reference:
                    line += f" {times['reference']:8.1f}ms {times['mozo'] / times['reference']:8.2f}x"
                print(line)

            batch_ms = median_ms(lambda: ours.encode_text(queries), device, runs=3, warmup=1)
            print(f"  {QUERIES} short queries in one call: {batch_ms:.0f} ms "
                  f"({batch_ms / QUERIES:.2f} ms each)")

        page = texts[2048]
        warm = median_ms(lambda: Tokenizer().encode(page), "cpu", runs=20, warmup=1)
        print(f"\n  tokenizer alone, 2048 tokens: {cold(lambda: Tokenizer().encode(page)):.2f} ms "
              f"cold, {warm:.2f} ms warm")
        if reference:
            # Warm too: the Rust tokenizer keeps a cache of its own across calls.
            reference_ms = median_ms(lambda: reference[1](page)["input_ids"], "cpu", runs=20)
            print(f"  transformers' tokenizer, same page, warm: {reference_ms:.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Measure Granite Embedding R2 on the CUDA runtimes mozo could publish: speed, drift, retrieval.

Bootstrap tooling; never ships, publishes nothing. It answers whether a graph runtime -- ONNX
Runtime on CUDA, or a TensorRT engine at fp32 or fp16 -- is worth publishing for a variant, and
the answer is recorded in the vendor's ``PROVENANCE.md`` under *Export*.

    python tools/bench/granite_embedding_r2_runtimes.py multilingual-small
    python tools/bench/granite_embedding_r2_runtimes.py multilingual-small --languages 5 \\
        --runtimes torch-fp32,tensorrt-fp16

**Quality is retrieval, not parity.** An fp16 vector moves by about 1e-3, and what matters is
whether search still finds the same things. So each runtime embeds Belebele (CC-BY-SA-4.0, from
the Hub) -- per language, its 900 questions against its 488 passages -- and reports nDCG@10 and
recall@1, averaged over languages. *Mixed* is the index a deployment would actually have when
runtimes differ: passages from this runtime, questions from torch fp32.

**Every text runs alone**, as the encoder runs it. Graphs are exported from the vendored encoder
into a scratch directory, the fp16 one under CUDA autocast; engines are strongly typed (TensorRT
11+), TF32 off, as mozo builds them. Speed is a median over synchronised runs at fixed lengths.
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import statistics
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault("MOZO_BASE_URL", f"file://{ROOT / 'weights'}")

from common import median_ms  # noqa: E402
from export.granite_embedding_r2 import Graph, export, session  # noqa: E402
from verify.granite_embedding_r2 import belebele_rows  # noqa: E402

from mozo.runtimes import TensorRTRunner  # noqa: E402
from mozo.vendors.granite_embedding_r2_deploy import SPECS, Encoder  # noqa: E402
from mozo.weights import resolve  # noqa: E402

#: Belebele languages, in order: the first *n* are measured. Twenty scripts in the first twenty.
LANGUAGES = ("eng_Latn", "deu_Latn", "fra_Latn", "spa_Latn", "rus_Cyrl", "arb_Arab", "zho_Hans",
             "jpn_Jpan", "kor_Hang", "hin_Deva", "ben_Beng", "tha_Thai", "vie_Latn", "tur_Latn",
             "ell_Grek", "heb_Hebr", "swh_Latn", "tel_Telu", "khm_Khmr", "amh_Ethi")

#: Lengths, in tokens, to time at.
LENGTHS = (16, 64, 256, 1024, 4096, 8192)

RUNTIMES = ("torch-fp32", "torch-fp16", "onnx-fp32", "tensorrt-fp32", "tensorrt-fp16")


def belebele(language: str) -> tuple[list[str], list[str], list[int]]:
    """``(passages, questions, the passage each question is about)`` for one language."""
    rows = belebele_rows(language)
    passages = list(dict.fromkeys(row["flores_passage"] for row in rows))
    index = {passage: i for i, passage in enumerate(passages)}
    return passages, [row["question"] for row in rows], [index[r["flores_passage"]] for r in rows]


def retrieval(data: dict, questions: dict, passages: dict) -> dict[str, float]:
    """nDCG@10 and recall@1, in percent, averaged over languages."""
    ndcg, recall = [], []
    for texts, asked, gold in data.values():
        scores = np.stack([questions[q] for q in asked]) @ np.stack([passages[p] for p in texts]).T
        ranks = (scores > scores[np.arange(len(gold)), gold][:, None]).sum(1)
        ndcg.append(np.mean(np.where(ranks < 10, 1 / np.log2(ranks + 2), 0)))
        recall.append(np.mean(ranks == 0))
    return {"nDCG@10": 100 * float(np.mean(ndcg)), "R@1": 100 * float(np.mean(recall))}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variant", choices=list(SPECS))
    parser.add_argument("--languages", type=int, default=20)
    parser.add_argument("--runtimes", default=",".join(RUNTIMES))
    parser.add_argument("--max-length", type=int, default=8192,
                        help="the longest input a TensorRT engine is built for")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    encoder = Encoder(resolve("granite_embedding_r2", args.variant, "torch-fp32"),
                      SPECS[args.variant], device="cuda")
    graph = Graph(encoder.network()).eval()
    data = {language: belebele(language) for language in LANGUAGES[:args.languages]}
    texts = list(dict.fromkeys(t for passages, asked, _ in data.values() for t in passages + asked))
    ids = {text: encoder.tokenizer.encode(text) for text in texts}
    print(f"{args.variant}, {len(data)} languages, {len(texts):,} texts, "
          f"median {statistics.median(map(len, ids.values()))} tokens, "
          f"{torch.cuda.get_device_name()}")
    body = encoder.tokenizer.encode(" ".join(texts))

    scratch = Path(tempfile.mkdtemp(prefix="granite-runtimes-"))

    def runner(name: str):
        half = name.endswith("fp16")
        if name.startswith("torch"):
            def run(row):
                with torch.inference_mode(), torch.autocast("cuda", torch.float16, enabled=half):
                    return graph(torch.tensor([row], device="cuda")).float()
            return run
        path = scratch / f"{'fp16' if half else 'fp32'}.onnx"
        if not path.exists():
            export(graph, path, half=half)
        if name == "onnx-fp32":
            ort_session = session(path, "cuda", threads=1)
            return lambda row: torch.from_numpy(
                ort_session.run(None, {"input_ids": np.array([row], dtype=np.int64)})[0])
        engine = TensorRTRunner(path, "cuda", engines=scratch,
                                shapes={"input_ids": ((1, 1), (1, 256), (1, args.max_length))},
                                probe=(torch.tensor([ids[texts[0]]], device="cuda"),))
        return lambda row: engine(torch.tensor([row], device="cuda"))[0].float()

    base = None
    for name in args.runtimes.split(","):
        run = runner(name)
        vectors = {text: run(ids[text]).cpu().numpy()[0] for text in texts}
        base = base or vectors
        drift = np.array([np.abs(vectors[t] - base[t]).max() for t in texts])
        cosine = min(float(vectors[t] @ base[t]) for t in texts)
        alone, mixed = retrieval(data, vectors, vectors), retrieval(data, base, vectors)
        speeds = []
        for length in (n for n in LENGTHS if n <= args.max_length):
            row = [body[0]] + body[1:length - 1] + [body[-1]]
            ms = median_ms(lambda: run(row), "cuda", runs=max(5, 2000 // length), warmup=3)
            speeds.append(f"{length}: {ms:.2f} ms")
        print(f"{name:14s} nDCG@10 {alone['nDCG@10']:.3f}  R@1 {alone['R@1']:.3f}  mixed nDCG@10 "
              f"{mixed['nDCG@10']:.3f}  max|d| {drift.max():.2e}  p99 {np.quantile(drift, .99):.2e}"
              f"  min cos {cosine:.6f}\n               {', '.join(speeds)}")
        # Drop this runtime before building the next: an ONNX Runtime session or an engine still
        # holding its GPU memory leaves the next engine's context too little to start.
        run = None  # noqa: F841
        gc.collect()
        torch.cuda.empty_cache()
    shutil.rmtree(scratch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build Granite Embedding R2's ONNX graph from the vendored encoder, measure it, and publish nothing.

    python tools/export/granite_embedding_r2.py                       # CPU
    python tools/export/granite_embedding_r2.py --device cuda         # needs onnxruntime-gpu
    python tools/export/granite_embedding_r2.py --keep /tmp/granite   # leave the graphs

This runs on a machine you control and never ships. **It writes nothing into ``weights/``**: like
``tools/export/ben2.py``, it is where a measured answer lives, so the answer can be re-run rather
than doubted. The answer, and the verdict it supports -- ``EXECUTES = ("torch",)`` -- are recorded
in one place, the vendor's ``PROVENANCE.md`` under *Export*.

The graph is the whole path after the tokenizer -- ``input_ids`` to the unit vector -- with batch
and sequence length dynamic. Both attention masks are built inside it from the length, which is
why the vendor builds the sliding mask at every length rather than skipping it below 64 tokens:
a trace would freeze whichever side of that branch its example input took.

On CUDA, torch is handed its input already on the device while ONNX Runtime copies a few kilobytes
in and out per call; at these sizes that is noise, and it is the cost a caller would pay.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault("MOZO_BASE_URL", f"file://{ROOT / 'weights'}")

from bench.granite_embedding_r2 import interleaved  # noqa: E402

from mozo.runtimes import providers_for  # noqa: E402
from mozo.vendors.granite_embedding_r2_deploy import SPECS, Encoder  # noqa: E402
from mozo.weights import resolve  # noqa: E402

#: The opset every mozo graph is written at.
OPSET = 17

#: Lengths to check parity at: both sides of the attention window, and up to the context.
PARITY_LENGTHS = (2, 17, 63, 64, 65, 66, 129, 512, 2048, 8192)

#: Lengths to time at, dense where ONNX Runtime and torch cross over.
SPEED_LENGTHS = (32, 64, 128, 256, 512, 1024, 2048, 8192)


class Graph(nn.Module):
    """``input_ids (N, L)`` -> unit vectors ``(N, width)``: the encoder, pooling and normalising,
    as ``Encoder.encode_text`` does after the tokenizer."""

    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        self.network = network

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(input_ids)[:, 0], p=2, dim=1)


def export(graph: Graph, path: Path) -> None:
    """Trace *graph* to ONNX with batch and length dynamic."""
    example = torch.randint(5, 50000, (1, 100))
    with torch.no_grad():
        torch.onnx.export(graph, (example,), str(path), opset_version=OPSET, dynamo=False,
                          input_names=["input_ids"], output_names=["embedding"],
                          dynamic_axes={"input_ids": {0: "batch", 1: "length"},
                                        "embedding": {0: "batch"}})


def session(path: Path, device: str, threads: int):
    """An ONNX Runtime session on *device*, refusing to fall back to the CPU without saying so."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    # ONNX Runtime's workers spin after each call by default, and spinning threads steal the CPU
    # from whichever side runs next -- torch, or another session -- in an interleaved benchmark.
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    created = ort.InferenceSession(str(path), options, providers=providers_for(device))
    if device.startswith("cuda") and "CUDAExecutionProvider" not in created.get_providers():
        raise SystemExit("ONNX Runtime fell back to the CPU; install onnxruntime-gpu, or these "
                         "would be CPU numbers labelled cuda")
    return created


def fuse(path: Path, width: int, heads: int, device: str) -> tuple[Path, dict[str, int]]:
    """Ask ONNX Runtime's transformer optimizer to fuse, as it would a BERT graph."""
    from onnxruntime.transformers import optimizer

    model = optimizer.optimize_model(str(path), model_type="bert", num_heads=heads,
                                     hidden_size=width, opt_level=1,
                                     use_gpu=device.startswith("cuda"))
    target = path.with_name(path.stem + "-fused.onnx")
    model.save_model_to_file(str(target))
    return target, {op: n for op, n in model.get_fused_operator_statistics().items() if n}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default="small", choices=list(SPECS))
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--keep", type=Path, help="leave the graphs here rather than deleting them")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    spec = SPECS[args.variant]
    encoder = Encoder(resolve("granite_embedding_r2", args.variant, "torch-fp32"), spec)
    graph = Graph(encoder.network())
    scratch = Path(args.keep or tempfile.mkdtemp(prefix="granite-onnx-"))
    scratch.mkdir(parents=True, exist_ok=True)
    path = scratch / f"{args.variant}.onnx"

    started = time.perf_counter()
    export(graph, path)  # on the CPU: the graph itself is device-agnostic
    graph.to(args.device)
    print(f"exported in {time.perf_counter() - started:.1f}s, {path.stat().st_size / 1e6:.1f} MB, "
          f"opset {OPSET}")

    plain = session(path, args.device, args.threads)
    print(f"ONNX Runtime providers: {plain.get_providers()}")

    # Real text, as the ids the tokenizer gives it, cut to exact lengths with its own markers.
    words = " ".join((ROOT / "docs" / name).read_text(encoding="utf-8")
                     for name in ("models.md", "workflows.md"))
    first, *body, last = encoder.tokenizer.encode(words)
    if len(body) < 2 * max(PARITY_LENGTHS):
        raise SystemExit(f"the docs give {len(body)} tokens; the parity check needs "
                         f"{2 * max(PARITY_LENGTHS)}")

    def ids(length: int, batch: int = 1) -> np.ndarray:
        step = max(length - 2, 1)  # at length 2 a row is the two markers alone
        return np.array([[first] + body[start:start + length - 2] + [last]
                         for start in range(0, batch * step, step)], dtype=np.int64)

    def delta(runner, x: np.ndarray) -> float:
        """Largest difference between *runner*'s unit vectors and the vendored encoder's."""
        with torch.inference_mode():
            want = graph(torch.from_numpy(x).to(args.device)).cpu().numpy()
        return float(np.abs(runner.run(None, {"input_ids": x})[0] - want).max())

    print("\nparity, unit vectors, ONNX Runtime against the vendored torch encoder:")
    worst = 0.0
    for length in PARITY_LENGTHS:
        x = ids(length, batch=2 if length <= 2048 else 1)
        worst = max(worst, found := delta(plain, x))
        print(f"  {length:5d} tokens x{x.shape[0]}  max|d| {found:.2e}")

    candidates = {"onnx": plain}
    try:
        fused_path, fused_ops = fuse(path, spec.width, spec.heads, args.device)
    except Exception as error:  # noqa: BLE001 -- a failed optimisation is a result, not a crash
        print(f"\nONNX Runtime's transformer optimizer failed: {type(error).__name__}: {error}")
    else:
        print(f"\nONNX Runtime's transformer optimizer fused: {fused_ops or 'nothing'}")
        # Only attention could change the picture; the other fusions measured as noise.
        if fused_ops.get("Attention") or fused_ops.get("MultiHeadAttention"):
            candidates["onnx-fused"] = session(fused_path, args.device, args.threads)
            print(f"  fused parity at 512 tokens x2: max|d| "
                  f"{delta(candidates['onnx-fused'], ids(512, batch=2)):.2e}")

    threads = f", {args.threads} threads" if args.device == "cpu" else ""
    print(f"\nspeed, {args.device}{threads}, each side warmed, interleaved over 3 trials:")
    print(f"  {'tokens':>6} {'torch':>10}" + "".join(f" {name:>18}" for name in candidates))
    for length in SPEED_LENGTHS:
        x = ids(length)
        tensor = torch.from_numpy(x).to(args.device)

        def torch_side():
            with torch.inference_mode():
                return graph(tensor).cpu()

        sides = {"torch": torch_side} | {
            name: (lambda runner=runner: runner.run(None, {"input_ids": x}))
            for name, runner in candidates.items()}
        medians = interleaved(sides, args.device, runs=max(5, 2048 // length), trials=3)
        print(f"  {length:6d} {medians['torch']:8.1f}ms" + "".join(
            f" {medians[name]:9.1f}ms ({medians[name] / medians['torch']:.2f}x)"
            for name in candidates))

    if not args.keep:
        shutil.rmtree(scratch)
    print(f"\nworst parity {worst:.2e}. Nothing was published"
          + (f"; graphs left in {scratch}." if args.keep else "."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

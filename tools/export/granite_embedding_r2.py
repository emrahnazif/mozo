#!/usr/bin/env python3
"""Build Granite Embedding R2's graphs from the vendored encoder: ONNX to measure, CoreML to publish.

    python tools/export/granite_embedding_r2.py                       # ONNX, CPU
    python tools/export/granite_embedding_r2.py --device cuda         # needs onnxruntime-gpu
    python tools/export/granite_embedding_r2.py --keep /tmp/granite   # leave the graphs
    python tools/export/granite_embedding_r2.py --coreml              # publish coreml-fp32

This runs on a machine you control and never ships. **The ONNX run writes nothing into
``weights/``**: like ``tools/export/ben2.py``, it is where a measured answer lives, so the answer
can be re-run rather than doubted. **``--coreml`` publishes**: it writes ``coreml-fp32.zip`` into
the current revision, and only once every length has agreed with torch. Both answers are recorded
in one place, the vendor's ``PROVENANCE.md`` under *Export*.

The CoreML package holds :class:`~mozo.vendors.granite_embedding_r2_deploy.graph.Bucketed` at each
of :data:`LENGTHS`, one function apiece (``tokens_<N>``), sharing one copy of the weights. Fixed
shapes because CoreML runs a flexible one on the CPU alone; several because one fixed length would
pad every query to 8,192 tokens.

The graph is the whole path after the tokenizer -- ``input_ids`` to the unit vector -- with batch
and sequence length dynamic. Both attention masks are built inside it from the length, which is
why the vendor builds the sliding mask at every length rather than skipping it below 64 tokens:
a trace would freeze whichever side of that branch its example input took.

On CUDA, torch is handed its input already on the device while ONNX Runtime copies a few kilobytes
in and out per call; at these sizes that is noise, and it is the cost a caller would pay.
"""

from __future__ import annotations

import argparse
import copy
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
from verify.granite_embedding_r2 import docs_text  # noqa: E402

from fetch.granite_embedding_r2 import REVISION  # noqa: E402

from mozo.runtimes import CoreMLRunner, providers_for  # noqa: E402
from mozo.vendors.granite_embedding_r2_deploy import SPECS, Encoder  # noqa: E402
from mozo.vendors.granite_embedding_r2_deploy.graph import Bucketed, padded  # noqa: E402
from mozo.weights import resolve  # noqa: E402

#: The opset every mozo graph is written at.
OPSET = 17

#: Lengths to check parity at: both sides of the attention window, and up to the context.
PARITY_LENGTHS = (2, 17, 63, 64, 65, 66, 129, 512, 2048, 8192)

#: Lengths to time at, dense where ONNX Runtime and torch cross over.
SPEED_LENGTHS = (32, 64, 128, 256, 512, 1024, 2048, 8192)

#: The fixed lengths the CoreML package holds. Doubling: a text runs at most twice its length, and
#: a text that only just spills into the next one costs 1.0-1.34x torch on MPS, measured.
LENGTHS = (64, 128, 256, 512, 1024, 2048, 4096, 8192)

#: The artifact ``--coreml`` publishes.
ARTIFACT = "coreml-fp32"

#: Multifunction packages need macOS 15. SAM 3's graph already requires it.
TARGET = "macOS15"

#: How far a CoreML vector may sit from torch's before the package is refused. Measured at 1.8e-07
#: or less; ten times the ONNX graph's worst, which is the same arithmetic-order disagreement.
TOLERANCE = 3e-06


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


def doc_ids(tokenizer):
    """``ids(length, batch=1)``: rows of real documentation, each exactly *length* ids, as int64."""
    first, *body, last = tokenizer.encode(docs_text())
    if len(body) < 2 * max(PARITY_LENGTHS):
        raise SystemExit(f"the docs give {len(body)} tokens; the parity check needs "
                         f"{2 * max(PARITY_LENGTHS)}")

    def ids(length: int, batch: int = 1) -> np.ndarray:
        step = max(length - 2, 1)  # at length 2 a row is the two markers alone
        return np.array([[first] + body[start:start + length - 2] + [last]
                         for start in range(0, batch * step, step)], dtype=np.int64)

    return ids


def convert(network: nn.Module, scratch: Path) -> Path:
    """One package holding :class:`Bucketed` at every one of :data:`LENGTHS`, weights shared."""
    import coremltools as ct

    graph, functions = Bucketed(network).eval(), ct.utils.MultiFunctionDescriptor()
    for length in LENGTHS:
        with torch.no_grad():
            traced = torch.jit.trace(graph, (torch.zeros((1, length), dtype=torch.long),
                                             torch.tensor([length], dtype=torch.int32)))
        single = scratch / f"tokens_{length}.mlpackage"
        ct.convert(traced,
                   inputs=[ct.TensorType(name="input_ids", shape=(1, length), dtype=np.int32),
                           ct.TensorType(name="length", shape=(1,), dtype=np.int32)],
                   outputs=[ct.TensorType(name="embedding", dtype=np.float32)],
                   convert_to="mlprogram", compute_precision=ct.precision.FLOAT32,
                   minimum_deployment_target=getattr(ct.target, TARGET)).save(str(single))
        functions.add_function(str(single), src_function_name="main",
                               target_function_name=f"tokens_{length}")
    functions.default_function_name = "tokens_512"
    staged = scratch / "package.mlpackage"
    ct.utils.save_multifunction(functions, str(staged))
    return staged


def check(staged: Path, network: nn.Module, ids) -> float:
    """Every length against torch, at the shortest text that needs it and the longest it holds.

    Prints each, timed against torch on MPS where there is one, and returns the worst max|d|.
    """
    reference = Graph(network)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    on_device = Graph(copy.deepcopy(network).to(device))
    print(f"\n{'length':>6} {'text':>5} {'load':>6} {'max|d|':>8} {'CoreML':>9} "
          f"{'torch ' + device:>10}")
    worst = 0.0
    for bucket in LENGTHS:
        started = time.perf_counter()
        runner = CoreMLRunner(staged, function=f"tokens_{bucket}")
        load = time.perf_counter() - started
        for length in (bucket // 2 + 1, bucket):
            row = ids(length)
            graph_ids, real = padded(row[0], bucket), np.array([length], dtype=np.int32)
            tokens = torch.from_numpy(row)
            with torch.inference_mode():
                want = reference(tokens).numpy()
            found = float(np.abs(runner(graph_ids, length=real)[0] - want).max())
            worst = max(worst, found)
            on = tokens.to(device)

            def torch_side():
                with torch.inference_mode():
                    return on_device(on).cpu()

            medians = interleaved({"coreml": lambda: runner(graph_ids, length=real),
                                   "torch": torch_side},
                                  device, runs=3 if bucket >= 4096 else 10, trials=3)
            print(f"{bucket:6d} {length:5d} {load:5.1f}s {found:8.1e} {medians['coreml']:7.1f}ms "
                  f"{medians['torch']:8.1f}ms ({medians['coreml'] / medians['torch']:.2f}x)")
    return worst


def coreml(variant: str, revision: str) -> None:
    """Build the CoreML package, hold every length to torch, and only then publish it.

    Written to a scratch directory and moved into place only once everything has agreed: a
    rejected package left in the revision directory would be hashed and published by
    ``tools/generate_manifest.py``, which has no way to know it was turned down.
    """
    encoder = Encoder(resolve("granite_embedding_r2", variant, "torch-fp32"), SPECS[variant])
    network = encoder.network()
    with tempfile.TemporaryDirectory() as scratch:
        started = time.perf_counter()
        staged = convert(network, Path(scratch))
        size = sum(f.stat().st_size for f in staged.rglob("*") if f.is_file())
        print(f"converted {len(LENGTHS)} lengths in {time.perf_counter() - started:.0f}s, "
              f"{size / 1e6:.0f} MB at {TARGET}")
        worst = check(staged, network, doc_ids(encoder.tokenizer))
        if worst > TOLERANCE:
            raise SystemExit(f"worst max|d| {worst:.1e} is over {TOLERANCE:.0e}; not published")
        destination = ROOT / "weights" / "granite_embedding_r2" / variant / revision
        shutil.make_archive(str(destination / ARTIFACT), "zip", root_dir=staged)
        print(f"\nworst max|d| {worst:.1e}; published {destination / ARTIFACT}.zip")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default="small", choices=list(SPECS))
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--keep", type=Path, help="leave the graphs here rather than deleting them")
    parser.add_argument("--coreml", action="store_true",
                        help=f"build, check and publish {ARTIFACT} into revision {REVISION}")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.coreml:
        coreml(args.variant, REVISION)
        return 0

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
    ids = doc_ids(encoder.tokenizer)

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

#!/usr/bin/env python3
"""Verify the vendored Depth Anything 3 against the authors' repository, exactly.

    # CPU (secondary gate)
    MKL_CBWR=AVX2,STRICT python tools/verify/depth_anything_v3.py \\
        --upstream /path/to/Depth-Anything-3 --checkpoints /path/to/hf-snapshots \\
        --images /path/to/photos [--variants small] [--falsify] [--no-autocast]
    # CUDA (primary gate): fp32, plus upstream's own half precisions under autocast
    python tools/verify/depth_anything_v3.py --device cuda --dtype fp32 fp16 bf16 \\
        --upstream ... --checkpoints ... --images ... --reference-cache refs/ [--falsify]

``--upstream`` is a checkout of the commit PROVENANCE.md records (refused otherwise);
``--checkpoints`` holds one directory per upstream repository name (``DA3-SMALL/`` with its
``config.json`` and ``model.safetensors``, as ``hf download`` writes them), which upstream's own
``DepthAnything3.from_pretrained`` loads. Nothing here installs upstream: its ``src/`` is put on
``sys.path``, and the two modules its API imports only for exporting files and aligning to given
camera poses (``utils.export``, ``utils.pose_align``, which need moviepy, trimesh, evo...) are
stubbed. Neither is reached by single-image inference.

Upstream and the vendor run in this one process, on ``--device``, with the same torch. Every
intermediate is recorded as its shape, dtype and the sha256 of its bytes (stricter than
``torch.equal``: it also tells ``-0.0`` from ``0.0``); the final depth maps are kept whole.

Checks, per ``--dtype``, every one exact -- no tolerance:

1. **Standalone** -- the vendored tree imports nothing outside stdlib, torch, torchvision, numpy
   and cv2, and nothing absolute of its own.
2. **Every stage** -- per image: upstream's preprocessed tensor (``InputProcessor``, as
   ``inference`` calls it) against the vendor's; then the output of *every module the two
   networks share by qualified name*, every call in order (blocks, attention, RoPE, norms, head
   projections, fusion blocks, convolutions...); the final depth upstream's ``OutputProcessor``
   returns against the vendor's ``predict``. Upstream's sky fill samples the global RNG, seeded 0
   first; the vendor's samples a generator seeded 0.

   * ``fp32``: upstream's network is called directly with autocast off; the vendor as mozo's
     ``torch-fp32`` runtime runs it.
   * ``fp16`` / ``bf16``: upstream's own ``DepthAnything3.forward``, which enters
     ``torch.autocast`` itself and picks bf16 when ``torch.cuda.is_bf16_supported()``, else fp16
     (that check is answered to pick the dtype asked for); the vendor under the same autocast.
     mozo publishes no torch half runtime for this family; this proves the vendor follows
     upstream's official half path too.
3. **Upstream's own API, autocast on** (with ``fp32``) -- ``DepthAnything3.inference([image])``
   unmodified, in the dtype upstream picks on this device, against the vendor's ``predict`` under
   the same autocast. Skipped with ``--no-autocast``.
4. **mozo** (with ``fp32``) -- ``DepthAnythingV3Predictor`` through the registry, the weights
   resolver (point ``MOZO_BASE_URL`` at a staged ``weights/`` tree) and the adapter on
   ``--device``, against the vendor's depth resized and inverted here. Skipped with ``--no-mozo``.
5. **Determinism** -- upstream twice and the vendor twice on the same image, byte-identical,
   every recorded intermediate.

Global settings are mozo's production ones: whatever torch defaults to after ``import mozo``.
Upstream's ``api`` sets ``torch.backends.cudnn.benchmark = False`` at import; the settings are
snapshotted before that import and restored after it, and any it changed are printed.

``--reference-cache DIR`` keeps upstream's records per variant (``DIR/<variant>.pt``), keyed by
device, dtype, image pixels and environment, so the falsifiers -- and a rerun -- reuse one
reference run: upstream does not change between them. Build it with an unperturbed run.

``--falsify`` then perturbs one thing at a time on the vendor's side and requires the gate to
fail at the stage it reaches, and not before (fp32 only). On CUDA it adds perturbations of the
hardware arithmetic alone: TF32 for matmuls on, TF32 for cuDNN off, channels-last convolution
inputs, SDPA forced to its math kernel, attention written out by hand.

On the CPU, run with ``MKL_CBWR=AVX2,STRICT``: MKL otherwise chooses sgemm kernels by buffer
alignment, so two processes -- or two models in one -- can differ in the last bit.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Callable, NamedTuple
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent.parent
os.environ.setdefault("MKL_CBWR", "AVX2,STRICT")
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.stdout.reconfigure(line_buffering=True)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from mozo.vendors import depth_anything_v3_deploy as vendor  # noqa: E402
from mozo.vendors.depth_anything_v3_deploy import predictor as vendor_predictor  # noqa: E402
from common import digest, fixtures  # noqa: E402
from mozo.image import load_image  # noqa: E402
from mozo.weights import resolve  # noqa: E402

VENDOR = ROOT / "mozo" / "vendors" / "depth_anything_v3_deploy"
ALLOWED_THIRD_PARTY = {"torch", "torchvision", "numpy", "cv2"}

#: mozo variant -> upstream's Hugging Face repository name.
UPSTREAM = {"small": "DA3-SMALL", "base": "DA3-BASE", "mono-large": "DA3MONO-LARGE"}

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

#: Bump when what the cache holds changes shape.
CACHE_VERSION = 1


def _pinned_commit() -> str:
    for line in (VENDOR / "PROVENANCE.md").read_text().splitlines():
        if "Upstream commit" in line:
            return line.split("`")[1]
    raise SystemExit("PROVENANCE.md does not record an upstream commit")


def check_standalone() -> list[str]:
    problems = []
    for path in sorted(VENDOR.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    continue
                names = [node.module or ""]
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                if top in sys.stdlib_module_names or top in ALLOWED_THIRD_PARTY:
                    continue
                problems.append(f"{path.relative_to(ROOT)}:{node.lineno} imports {name}")
    return problems


# --- global settings ----------------------------------------------------------------------------

def settings() -> dict[str, object]:
    """The process-wide switches that change GPU arithmetic, as they stand."""
    out: dict[str, object] = {
        "cudnn.benchmark": torch.backends.cudnn.benchmark,
        "cudnn.deterministic": torch.backends.cudnn.deterministic,
        "cudnn.allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    for name in ("flash", "mem_efficient", "math", "cudnn"):
        out[f"sdp.{name}"] = getattr(torch.backends.cuda, f"{name}_sdp_enabled")()
    return out


def restore(saved: dict[str, object]) -> None:
    torch.backends.cudnn.benchmark = saved["cudnn.benchmark"]
    torch.backends.cudnn.deterministic = saved["cudnn.deterministic"]
    torch.backends.cudnn.allow_tf32 = saved["cudnn.allow_tf32"]
    torch.backends.cuda.matmul.allow_tf32 = saved["matmul.allow_tf32"]
    torch.set_float32_matmul_precision(saved["float32_matmul_precision"])
    torch.use_deterministic_algorithms(saved["deterministic_algorithms"])
    for name in ("flash", "mem_efficient", "math", "cudnn"):
        getattr(torch.backends.cuda, f"enable_{name}_sdp")(saved[f"sdp.{name}"])


def environment(device: torch.device) -> str:
    import torchvision
    line = (f"torch {torch.__version__}, torchvision {torchvision.__version__}, numpy "
            f"{np.__version__}, cv2 {cv2.__version__}")
    if device.type == "cuda":
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                                 "--format=csv,noheader", f"--id={device.index or 0}"],
                                capture_output=True, text=True)
        line += (f"; CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}, "
                 f"{torch.cuda.get_device_name(device)}, driver "
                 f"{driver.stdout.strip() or 'unknown'}")
    else:
        line += f"; MKL_CBWR={os.environ['MKL_CBWR']}"
    return line


# --- upstream -----------------------------------------------------------------------------------

def load_upstream_package(checkout: Path, allow_other_commit: bool):
    commit = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    if commit != _pinned_commit() and not allow_other_commit:
        raise SystemExit(f"{checkout} is at {commit}, PROVENANCE.md records {_pinned_commit()}. "
                         "Pass --allow-other-commit to compare anyway.")
    sys.path.insert(0, str(checkout / "src"))
    # Imported by api.py for writing files and for aligning to caller-given poses; neither runs
    # for single-image inference, and both pull in heavy optional dependencies.
    for name, attr in (("depth_anything_3.utils.export", "export"),
                       ("depth_anything_3.utils.pose_align", "align_poses_umeyama")):
        stub = types.ModuleType(name)
        setattr(stub, attr, lambda *a, **k: (_ for _ in ()).throw(RuntimeError(f"{name} stub")))
        sys.modules[name] = stub
    saved = settings()
    from depth_anything_3.api import DepthAnything3
    changed = {k: (saved[k], v) for k, v in settings().items() if v != saved[k]}
    restore(saved)
    return DepthAnything3, commit, changed


def realloc(module: torch.nn.Module) -> torch.nn.Module:
    """Give every parameter fresh storage, so BLAS sees the same alignment on both sides."""
    with torch.no_grad():
        for p in module.parameters():
            p.data = p.data.clone()
        for name, b in module.named_buffers():
            b.data = b.data.clone()
    return module


# --- recording ----------------------------------------------------------------------------------

def record(t: torch.Tensor) -> tuple:
    """A tensor as its shape, dtype and the sha256 of its bytes."""
    shape = tuple(t.shape)
    t = t.detach().contiguous().cpu().reshape(-1)
    return (shape, str(t.dtype), hashlib.sha256(t.view(torch.uint8).numpy()).hexdigest())


def _tensors(value) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return [t for k in sorted(value) for t in _tensors(value[k])]
    if isinstance(value, (tuple, list)):
        return [t for v in value for t in _tensors(v)]
    return []


class Calls(NamedTuple):
    #: name -> one list of tensor records per call
    by_name: dict[str, list[list[tuple]]]
    #: (name, call index) in the order the calls ran
    order: list[tuple[str, int]]


@contextlib.contextmanager
def recording(model: torch.nn.Module, prefix: str, names: set[str]):
    """Record every call's tensor outputs of each module in *names* (keys without *prefix*)."""
    calls = Calls({}, [])
    handles = []
    for name, module in model.named_modules():
        if not name.startswith(prefix) or name[len(prefix):] not in names:
            continue
        key = name[len(prefix):]

        def hook(_m, _i, out, key=key):
            per = calls.by_name.setdefault(key, [])
            calls.order.append((key, len(per)))
            per.append([record(t) for t in _tensors(out)])
        handles.append(module.register_forward_hook(hook))
    try:
        yield calls
    finally:
        for h in handles:
            h.remove()


def compare_calls(ours: Calls, theirs: Calls) -> tuple[int, list[str]]:
    """Count equal module calls and name the differing ones, in the order the vendor ran them."""
    shared = set(ours.by_name) & set(theirs.by_name)
    equal, bad = 0, []
    for key in sorted(shared):
        a, b = ours.by_name[key], theirs.by_name[key]
        if len(a) != len(b):
            bad.append((-1, f"{key}: {len(a)} calls vs {len(b)}"))
    for step, (key, i) in enumerate(ours.order):
        if key not in shared or i >= len(theirs.by_name[key]):
            continue
        if ours.by_name[key][i] == theirs.by_name[key][i]:
            equal += 1
        else:
            bad.append((step, f"{key}[{i}]"))
    return equal, [text for _, text in sorted(bad)]


# --- the gate -----------------------------------------------------------------------------------

class Stage(NamedTuple):
    name: str
    ok: bool
    detail: str


def _delta(a: torch.Tensor, b: torch.Tensor) -> str:
    if a.shape != b.shape:
        return f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
    return f"max|d| {float((a.float() - b.float()).abs().max()):.3g}"


def _equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and record(a) == record(b)


def shared_modules(upstream, ours: vendor.Predictor) -> set[str]:
    # "head" itself is left out: upstream's any-view head also returns the ray branch mozo does
    # not build. Everything inside it that both have is compared.
    return ({n[len("model."):] for n, _ in upstream.named_modules() if n.startswith("model.")}
            & {n for n, _ in ours.model.named_modules()}) - {"", "head"}


def fresh_caches(upstream) -> None:
    """Empty upstream's RoPE caches, as a fresh process has them.

    ``RotaryPositionEmbedding2D.frequency_cache`` is keyed by ``(dim, seq_len, device, dtype)``,
    where dtype is the tokens' -- fp32 under autocast too, since the QK-norm before it is a
    LayerNorm -- so the table built by the *first* call is reused whatever autocast is active
    later. Built under autocast its ``einsum`` runs in half precision on CUDA. Without this, an
    fp32 call followed by an autocast call gives upstream an fp32 table under autocast, and the
    order of the gate's own calls would decide upstream's answer.
    """
    for module in upstream.modules():
        if isinstance(getattr(module, "frequency_cache", None), dict):
            module.frequency_cache.clear()
        for value in vars(module).values():
            if isinstance(getattr(value, "position_cache", None), dict):
                value.position_cache.clear()


def _upstream_once(upstream, rgb: np.ndarray, dtype: str, shared: set[str]):
    fresh_caches(upstream)
    up_cpu, _, _ = upstream._preprocess_inputs([rgb], None, None, 504, "upper_bound_resize")
    up_input, _, _ = upstream._prepare_model_inputs(up_cpu, None, None)
    with torch.inference_mode():
        torch.manual_seed(0)
        with recording(upstream, "model.", shared) as calls:
            if dtype == "fp32":
                raw = upstream.model(up_input, None, None, [], False, False, "saddle_balanced")
            else:
                # Upstream's own half path: forward() enters torch.autocast in bf16 when the GPU
                # supports it, else fp16. The check is answered to pick the dtype asked for.
                with mock.patch.object(torch.cuda, "is_bf16_supported",
                                       lambda *a, **k: dtype == "bf16"):
                    raw = upstream.forward(up_input, None, None, [], False, False,
                                           "saddle_balanced")
        depth = torch.from_numpy(upstream._convert_to_prediction(raw).depth[0])
    return record(up_input), calls, depth


def reference(upstream, rgb: np.ndarray, dtype: str, shared: set[str], device: torch.device,
              autocast: bool) -> dict:
    """Upstream's records for one image at one dtype, run twice to show it is deterministic."""
    first = _upstream_once(upstream, rgb, dtype, shared)
    second = _upstream_once(upstream, rgb, dtype, shared)
    ref = {"input": first[0], "calls": first[1], "depth": first[2],
           "deterministic": (first[0] == second[0] and first[1].by_name == second[1].by_name
                             and _equal(first[2], second[2]))}
    if autocast and dtype == "fp32":
        # Upstream's unmodified API, in the dtype it picks itself on this device.
        picked = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        fresh_caches(upstream)
        torch.manual_seed(0)
        ref["api"] = torch.from_numpy(upstream.inference([rgb]).depth[0])
        ref["api_dtype"] = picked
    return ref


def _ours_once(ours: vendor.Predictor, rgb: np.ndarray, dtype: str, shared: set[str]):
    our_input = ours.preprocess(rgb)
    with torch.inference_mode(), recording(ours.model, "", shared) as calls, \
            torch.autocast(ours.device.type, dtype=DTYPES[dtype], enabled=dtype != "fp32"):
        depth = ours.predict(rgb).cpu()
    return record(our_input), calls, depth


def gate_image(ref: dict, ours: vendor.Predictor, rgb: np.ndarray, dtype: str,
               shared: set[str], full: bool = True) -> list[Stage]:
    """Compare the vendor with upstream's records *ref*. *full* adds the checks falsification
    does not need (upstream's API, determinism, shape inputs)."""
    stages: list[Stage] = []
    our_input, mine, our_depth = _ours_once(ours, rgb, dtype, shared)
    stages.append(Stage("input", our_input == ref["input"], f"{our_input[0][-2:]}"))
    equal, bad = compare_calls(mine, ref["calls"])
    n = len(set(mine.by_name) & set(ref["calls"].by_name))
    stages.append(Stage("modules", not bad, f"{equal} calls of {n} modules equal"
                        + (f"; first differing: {bad[:3]}" if bad else "")))
    same = _equal(ref["depth"], our_depth)
    stages.append(Stage(f"depth {dtype}", same, "" if same else _delta(ref["depth"], our_depth)))
    if not full:
        return stages

    if "api" in ref:
        with torch.autocast(ours.device.type, dtype=ref["api_dtype"]):
            our_auto = ours.predict(rgb).cpu()
        same = _equal(ref["api"], our_auto)
        detail = (str(ref["api_dtype"]).replace("torch.", "") if same
                  else _delta(ref["api"], our_auto))
        stages.append(Stage("api autocast", same, detail))

    again = _ours_once(ours, rgb, dtype, shared)
    stages.append(Stage("deterministic",
                        ref["deterministic"] and again[0] == our_input
                        and again[1].by_name == mine.by_name and _equal(again[2], our_depth),
                        "" if ref["deterministic"] else "upstream is not"))
    if dtype == "fp32":
        # The shape-only tensors a graph is handed: computed beforehand, they must give exactly
        # what the network computes for itself.
        x = ours.preprocess(rgb).to(ours.device)
        with torch.inference_mode():
            a = ours.model(x)
            b = ours.model(x, ours.model.shape_inputs(*x.shape[-2:], device=x.device))
        stages.append(Stage("shape inputs", all(_equal(a[k], b[k]) for k in a), ""))
    return stages


def gate_mozo(variant: str, ours: vendor.Predictor, images: list[np.ndarray],
              device: torch.device) -> Stage:
    from mozo.adapters.depth_anything_v3 import DepthAnythingV3Predictor

    model = DepthAnythingV3Predictor(variant, device=str(device), runtime="torch-fp32")
    realloc(model._predictor.model)
    worst, failures = 0.0, 0
    for rgb in images:
        expected = (1.0 / vendor.resize(ours.predict(rgb), rgb.shape[:2])).cpu().numpy()
        got = model.predict(rgb)
        if got.shape != expected.shape or got.tobytes() != expected.tobytes():
            failures += 1
            if got.shape == expected.shape:
                worst = max(worst, float(np.abs(got - expected).max()))
    return Stage("mozo", failures == 0,
                 f"{len(images) - failures}/{len(images)} images" +
                 (f", max|d| {worst:.3g}" if failures else ""))


# --- reference cache ----------------------------------------------------------------------------

def _image_key(name: str, rgb: np.ndarray) -> str:
    return f"{name}|{rgb.shape}|{hashlib.sha256(rgb.tobytes()).hexdigest()[:16]}"


class References:
    """Upstream's records for one variant, from ``--reference-cache`` or computed (and saved)."""

    def __init__(self, path: Path | None, meta: dict, make: Callable[[np.ndarray, str], dict]):
        self.path, self.meta, self.make, self.dirty = path, meta, make, False
        self.entries: dict[str, dict] = {}
        if path is not None and path.exists():
            saved = torch.load(path, weights_only=False)
            if saved.get("meta") != meta:
                raise SystemExit(f"{path} was made under {saved.get('meta')}, not {meta}. "
                                 "Delete it, or use another --reference-cache.")
            self.entries = saved["entries"]
            self.shared = saved["shared"]

    def get(self, name: str, rgb: np.ndarray, dtype: str) -> tuple[dict, bool]:
        """The records, and whether they came from the cache."""
        key = f"{dtype}|{_image_key(name, rgb)}"
        if key in self.entries:
            return self.entries[key], True
        self.entries[key] = self.make(rgb, dtype)
        self.dirty = True
        return self.entries[key], False

    def save(self, shared: set[str]) -> None:
        if self.path is not None and self.dirty:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"meta": self.meta, "shared": shared, "entries": self.entries}, self.path)
            self.dirty = False


# --- falsification ------------------------------------------------------------------------------

class Falsification(NamedTuple):
    name: str
    #: The first stage that must fail; every stage before it must still pass.
    stage: str
    applies: Callable[[str, str], bool]          # (variant, device type)
    patch: Callable[[vendor.Predictor], contextlib.AbstractContextManager]


def _setattr(obj, name, value):
    return mock.patch.object(obj, name, value)


def _first_block_norm_eps(p):
    return _setattr(p.model.backbone.pretrained.blocks[0].norm1, "eps", 1e-5)


def _rope_frequency(p):
    return _setattr(p.model.backbone.pretrained.rope, "base_frequency", 101.0)


def _no_interpolate_offset(p):
    return _setattr(p.model.backbone.pretrained, "interpolate_offset", 0.0)


def _qk_norm_eps(p):
    return _setattr(p.model.backbone.pretrained.blocks[4].attn.q_norm, "eps", 1e-6)


def _final_norm_eps(p):
    return _setattr(p.model.backbone.pretrained.norm, "eps", 1e-6)


def _pos_embed_ratio(p):
    return _setattr(vendor.dualdpt, "UV_RATIO", vendor.dualdpt.UV_RATIO * 1.0000001)


def _area_to_linear(p):
    # A stand-in for the vendor's own ``cv2`` name only: patching cv2.resize itself would change
    # upstream's preprocessing too, and the gate would compare two equally wrong inputs.
    def resize(img, size, interpolation=None):
        if interpolation == cv2.INTER_AREA:
            interpolation = cv2.INTER_LINEAR
        return cv2.resize(img, size, interpolation=interpolation)
    proxy = types.SimpleNamespace(**{k: getattr(cv2, k) for k in ("INTER_AREA", "INTER_CUBIC")},
                                  resize=resize)
    return _setattr(vendor_predictor, "cv2", proxy)


def _ties_round_down(p):
    def nearest(x, q):
        down = (x // q) * q
        up = down + q
        return up if abs(up - x) < abs(x - down) else down
    return _setattr(vendor_predictor, "_nearest_multiple", nearest)


def _sky_seed(p):
    return _setattr(vendor_predictor, "SKY_SEED", 1)


def _sky_threshold(p):
    return _setattr(vendor.net, "SKY_THRESHOLD", 0.31)


def _camera_token_unused(p):
    real = p.model.backbone.pretrained.camera_token
    return _setattr(p.model.backbone.pretrained, "camera_token",
                    torch.nn.Parameter(real.detach().flip(1)))


# Hardware arithmetic only, on the vendor's side only: the same values, other kernels. Each wraps
# the vendor network's forward, so upstream (when it is run rather than read from the cache) keeps
# mozo's production settings.

@contextlib.contextmanager
def _around_forward(p, enter: Callable[[], contextlib.AbstractContextManager]):
    real = p.model.forward

    def forward(*args, **kwargs):
        with enter():
            return real(*args, **kwargs)
    with _setattr(p.model, "forward", forward):
        yield


@contextlib.contextmanager
def _flag(obj, name, value):
    saved = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, saved)


def _matmul_tf32_on(p):
    return _around_forward(p, lambda: _flag(torch.backends.cuda.matmul, "allow_tf32", True))


def _cudnn_tf32_off(p):
    return _around_forward(p, lambda: _flag(torch.backends.cudnn, "allow_tf32", False))


def _sdpa_math(p):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    return _around_forward(p, lambda: sdpa_kernel(SDPBackend.MATH))


@contextlib.contextmanager
def _hand_attention(p):
    # The vendored DINOv2 attention's own unfused path: q * scale, q @ k^T, softmax, @ v.
    attns = [m for m in p.model.modules() if hasattr(m, "fused_attn")]
    with contextlib.ExitStack() as stack:
        for m in attns:
            stack.enter_context(_setattr(m, "fused_attn", False))
        yield


@contextlib.contextmanager
def _channels_last(p):
    # Every convolution's input in channels-last memory: torch.equal values, other strides, so
    # cuDNN chooses other kernels.
    def pre(_m, args):
        return (args[0].contiguous(memory_format=torch.channels_last),) + tuple(args[1:])
    handles = [m.register_forward_pre_hook(pre) for m in p.model.modules()
               if isinstance(m, torch.nn.Conv2d)]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def _any(v, d):
    return True


def _anyview(v, d):
    return v != "mono-large"


def _cuda(v, d):
    return d == "cuda"


FALSIFICATIONS = (
    Falsification("block norm eps 1e-6 -> 1e-5", "modules", _any, _first_block_norm_eps),
    Falsification("final norm eps 1e-5 -> 1e-6", "modules", _any, _final_norm_eps),
    Falsification("pos-embed interpolate_offset 0.1 -> 0", "modules", _any,
                  _no_interpolate_offset),
    Falsification("INTER_AREA -> INTER_LINEAR", "input", _any, _area_to_linear),
    Falsification("multiple-of-14 ties round down", "input", _any, _ties_round_down),
    Falsification("RoPE frequency 100 -> 101", "modules", _anyview, _rope_frequency),
    Falsification("QK-norm eps 1e-5 -> 1e-6", "modules", _anyview, _qk_norm_eps),
    Falsification("reference/source camera tokens swapped", "modules", _anyview,
                  _camera_token_unused),
    Falsification("head UV pos-embed ratio x1.0000001", "modules", _anyview, _pos_embed_ratio),
    Falsification("sky seed 0 -> 1", "depth fp32", lambda v, d: v == "mono-large", _sky_seed),
    Falsification("sky threshold 0.3 -> 0.31", "depth fp32", lambda v, d: v == "mono-large",
                  _sky_threshold),
    Falsification("matmul TF32 on (vendor side)", "modules", _cuda, _matmul_tf32_on),
    Falsification("cuDNN TF32 off (vendor side)", "modules", _cuda, _cudnn_tf32_off),
    Falsification("channels-last conv inputs", "modules", _cuda, _channels_last),
    Falsification("SDPA forced to math", "modules", _cuda, _sdpa_math),
    Falsification("attention written out by hand", "modules", _cuda, _hand_attention),
)


#: The images falsification runs on: a photograph (shrunk with INTER_AREA), one whose sky (25k
#: pixels for mono-large) sits beside more than 100,000 other pixels, so the sky fill samples, and
#: a tie in the multiple-of-14 rounding. A sky beside fewer non-sky pixels is never sampled, and
#: the seed then changes nothing to catch.
FALSIFY_IMAGES = ("example.jpg", "pixabay_cc0_seonyudo-633518_1280.jpg",
                  "tie-504x217 (rounds 217 up)")


def falsify(variant: str, refs: References, ours: vendor.Predictor,
            images: list[tuple[str, np.ndarray]], shared: set[str]) -> bool:
    order = ["input", "modules", "depth fp32"]
    caught_all = True
    for f in FALSIFICATIONS:
        if not f.applies(variant, ours.device.type):
            continue
        with f.patch(ours):
            results = [s for name, rgb in images
                       for s in gate_image(refs.get(name, rgb, "fp32")[0], ours, rgb, "fp32",
                                           shared, full=False)]
        failed = {s.name for s in results if not s.ok and s.name in order}
        first = next((name for name in order if name in failed), None)
        ok = first == f.stage
        caught_all &= ok
        print(f"  falsify {f.name:42s} first failing stage: {first or 'none':10s} "
              f"{'caught' if ok else 'NOT CAUGHT AS EXPECTED (want ' + f.stage + ')'}")
    return caught_all


# --- main ---------------------------------------------------------------------------------------

def read_images(paths: list[Path]) -> list[tuple[str, np.ndarray]]:
    """Decode through mozo's own entry point, the RGB array a user's call starts from."""
    return [(path.name, load_image(str(path))) for path in paths]


def synthetic() -> list[tuple[str, np.ndarray]]:
    """Awkward inputs: the resize branches, a tie in the multiple-of-14 rounding, flat input."""
    rng = np.random.default_rng(0)
    return [
        ("noise-640x480", rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)),
        ("tiny-90x60 (enlarged)", rng.integers(0, 256, (60, 90, 3), dtype=np.uint8)),
        ("exact-504x300 (no first resize)", rng.integers(0, 256, (300, 504, 3), dtype=np.uint8)),
        ("portrait-300x700", rng.integers(0, 256, (700, 300, 3), dtype=np.uint8)),
        ("wide-2000x150", rng.integers(0, 256, (150, 2000, 3), dtype=np.uint8)),
        ("tie-504x217 (rounds 217 up)", rng.integers(0, 256, (217, 504, 3), dtype=np.uint8)),
        ("black-640x480", np.zeros((480, 640, 3), np.uint8)),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs="*", default=[],
                        help="image files or directories (default: tests/fixtures/images)")
    parser.add_argument("--variants", nargs="*", default=list(UPSTREAM))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", nargs="+", choices=tuple(DTYPES), default=["fp32"],
                        help="fp32 is mozo's torch runtime; fp16/bf16 run upstream's own "
                             "autocast path on both sides")
    parser.add_argument("--reference-cache", type=Path,
                        help="directory keeping upstream's records per variant")
    parser.add_argument("--falsify", action="store_true")
    parser.add_argument("--no-autocast", action="store_true")
    parser.add_argument("--no-mozo", action="store_true")
    parser.add_argument("--allow-other-commit", action="store_true")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    problems = check_standalone()
    print(f"standalone: {'PASS' if not problems else 'FAIL'}")
    for p in problems:
        print("  ", p)

    DepthAnything3, commit, changed = load_upstream_package(args.upstream,
                                                            args.allow_other_commit)
    env = environment(device)
    print(f"upstream {commit}; {env}")
    print(f"settings (mozo's production, restored after importing upstream): {settings()}")
    if changed:
        print(f"  upstream's import changed {changed}; restored")

    paths = [] if args.images else fixtures()
    for item in args.images:
        paths += sorted(p for p in item.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                        ) if item.is_dir() else [item]
    images = read_images(paths) + synthetic()
    production = settings()

    all_ok = not problems
    for variant in args.variants:
        repo = UPSTREAM[variant]
        # The artifact mozo publishes, through mozo's resolver (MOZO_BASE_URL may point at a
        # staged weights/ tree).
        weights = resolve("depth_anything_v3", variant, "torch-fp32")
        ours = vendor.Predictor.from_pretrained(variant, weights=weights)
        realloc(ours.model).to(device)
        print(f"\n== {variant} ({repo}, mozo weights sha256 {digest(weights)[:12]}...)")

        holder: dict = {}

        def upstream_model():
            if "model" not in holder:
                holder["model"] = realloc(DepthAnything3.from_pretrained(
                    str(args.checkpoints / repo)).eval()).to(device)
            return holder["model"]

        meta = {"version": CACHE_VERSION, "variant": variant, "commit": commit, "env": env,
                "device": str(device), "autocast": not args.no_autocast,
                "settings": production}
        refs = References(
            args.reference_cache / f"{variant}.pt" if args.reference_cache else None, meta,
            lambda rgb, dtype: reference(upstream_model(), rgb, dtype, shared, device,
                                         not args.no_autocast))
        shared = getattr(refs, "shared", None) or shared_modules(upstream_model(), ours)

        counts: dict[str, list[int]] = {}
        cached = 0
        for dtype in args.dtype:
            for name, rgb in images:
                try:
                    ref, hit = refs.get(name, rgb, dtype)
                    cached += hit
                    stages = gate_image(ref, ours, rgb, dtype, shared)
                except Exception as error:  # noqa: BLE001 - reported as a failing image
                    stages = [Stage("error", False, repr(error))]
                line = "  ".join(f"{s.name} {'ok' if s.ok else 'FAIL'}"
                                 + (f" ({s.detail})" if s.detail and (not s.ok or s.name in
                                                                      {"input", "modules"})
                                    else "")
                                 for s in stages)
                print(f"  {dtype} {name:34s} {line}")
                for s in stages:
                    key = s.name if s.name.startswith("depth") else f"{s.name} {dtype}"
                    counts.setdefault(key, [0, 0])
                    counts[key][0] += s.ok
                    counts[key][1] += 1
            refs.save(shared)
        if not args.no_mozo and "fp32" in args.dtype:
            stage = gate_mozo(variant, ours, [rgb for _, rgb in images], device)
            print(f"  mozo adapter: {'ok' if stage.ok else 'FAIL'} ({stage.detail})")
            counts["mozo"] = [int(stage.ok), 1]
        ok = all(passed == total for passed, total in counts.values())
        print(f"  {variant}: " + ", ".join(f"{k} {p}/{t}" for k, (p, t) in counts.items())
              + f" -> {'PASS' if ok else 'FAIL'}"
              + (f"  ({cached} reference records from the cache)" if cached else ""))
        all_ok &= ok
        if args.falsify:
            chosen = [(name, rgb) for name, rgb in images if name in FALSIFY_IMAGES]
            if len(chosen) != len(FALSIFY_IMAGES):
                raise SystemExit(f"--falsify needs {FALSIFY_IMAGES} among the images")
            all_ok &= falsify(variant, refs, ours, chosen, shared)
            refs.save(shared)
        if settings() != production:
            print(f"  global settings moved during the run: {settings()}; restored")
            restore(production)
            all_ok = False
        holder.clear()
        del ours
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\n{'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

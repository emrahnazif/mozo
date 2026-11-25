#!/usr/bin/env python3
"""Verify the vendored Depth Anything 3 against the authors' repository, exactly.

    MKL_CBWR=AVX2,STRICT python tools/verify/depth_anything_v3.py \\
        --upstream /path/to/Depth-Anything-3 --checkpoints /path/to/hf-snapshots \\
        --images /path/to/photos [--variants small] [--falsify] [--no-autocast]

``--upstream`` is a checkout of the commit PROVENANCE.md records (refused otherwise);
``--checkpoints`` holds one directory per upstream repository name (``DA3-SMALL/`` with its
``config.json`` and ``model.safetensors``, as ``hf download`` writes them), which upstream's own
``DepthAnything3.from_pretrained`` loads. Nothing here installs upstream: its ``src/`` is put on
``sys.path``, and the two modules its API imports only for exporting files and aligning to given
camera poses (``utils.export``, ``utils.pose_align``, which need moviepy, trimesh, evo...) are
stubbed. Neither is reached by single-image inference.

Checks, every one ``torch.equal`` -- no tolerance:

1. **Standalone** -- the vendored tree imports nothing outside stdlib, torch, torchvision, numpy
   and cv2, and nothing absolute of its own.
2. **Every stage, fp32** -- per image: upstream's preprocessed tensor (``InputProcessor``, as
   ``inference`` calls it) against the vendor's; then the output of *every module the two
   networks share by qualified name*, every call in order (blocks, attention, RoPE, norms, head
   projections, fusion blocks, convolutions...); the head's depth (and sky); the final depth
   upstream's ``OutputProcessor`` returns against the vendor's ``predict``. Upstream's network is
   called directly with autocast off, after ``torch.manual_seed(0)`` (its sky fill samples the
   global RNG; the vendor's samples a generator seeded 0).
3. **Upstream's own API, autocast on** -- ``DepthAnything3.inference([image])`` unmodified,
   which runs under ``torch.autocast`` in the dtype upstream picks, against the vendor's
   ``predict`` under the same autocast.
4. **mozo** -- ``DepthAnythingV3Predictor`` through the registry, the weights resolver (point
   ``MOZO_BASE_URL`` at a staged ``weights/`` tree) and the adapter, against the vendor's depth
   resized and inverted here. Skipped with ``--no-mozo``.
5. **Determinism** -- the vendor twice on the same image, byte-identical.

``--falsify`` then perturbs one constant at a time and requires the gate to fail at the stage
that constant reaches, and not before.

Run on the CPU, with ``MKL_CBWR=AVX2,STRICT``: MKL otherwise chooses sgemm kernels by buffer
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
sys.stdout.reconfigure(line_buffering=True)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from mozo.vendors import depth_anything_v3_deploy as vendor  # noqa: E402
from mozo.vendors.depth_anything_v3_deploy import predictor as vendor_predictor  # noqa: E402
from mozo.weights import resolve  # noqa: E402

VENDOR = ROOT / "mozo" / "vendors" / "depth_anything_v3_deploy"
ALLOWED_THIRD_PARTY = {"torch", "torchvision", "numpy", "cv2"}

#: mozo variant -> (upstream repository name, upstream model name).
UPSTREAM = {"small": ("DA3-SMALL", "da3-small"), "base": ("DA3-BASE", "da3-base"),
            "mono-large": ("DA3MONO-LARGE", "da3mono-large")}


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
    from depth_anything_3.api import DepthAnything3
    return DepthAnything3, commit


def realloc(module: torch.nn.Module) -> torch.nn.Module:
    """Give every parameter fresh storage, so BLAS sees the same alignment on both sides."""
    with torch.no_grad():
        for p in module.parameters():
            p.data = p.data.clone()
        for name, b in module.named_buffers():
            b.data = b.data.clone()
    return module


# --- recording ----------------------------------------------------------------------------------

def _tensors(value) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return [t for k in sorted(value) for t in _tensors(value[k])]
    if isinstance(value, (tuple, list)):
        return [t for v in value for t in _tensors(v)]
    return []


@contextlib.contextmanager
def recording(model: torch.nn.Module, prefix: str, names: set[str] | None = None):
    """Record every call's tensor outputs of each module, keyed by name without *prefix*."""
    calls: dict[str, list[list[torch.Tensor]]] = {}
    handles = []
    for name, module in model.named_modules():
        if not name.startswith(prefix):
            continue
        key = name[len(prefix):]
        if names is not None and key not in names:
            continue

        def hook(_m, _i, out, key=key):
            calls.setdefault(key, []).append([t.detach().clone() for t in _tensors(out)])
        handles.append(module.register_forward_hook(hook))
    try:
        yield calls
    finally:
        for h in handles:
            h.remove()


def compare_calls(ours: dict, theirs: dict) -> tuple[int, list[str]]:
    """Count equal module calls and name the first differing ones."""
    equal, bad = 0, []
    for key in sorted(set(ours) & set(theirs)):
        a, b = ours[key], theirs[key]
        if len(a) != len(b):
            bad.append(f"{key}: {len(a)} calls vs {len(b)}")
            continue
        for i, (xs, ys) in enumerate(zip(a, b)):
            if len(xs) != len(ys) or not all(x.shape == y.shape and torch.equal(x, y)
                                             for x, y in zip(xs, ys)):
                worst = max((float((x.float() - y.float()).abs().max())
                             for x, y in zip(xs, ys) if x.shape == y.shape), default=float("nan"))
                bad.append(f"{key}[{i}]: max|d| {worst:.3g}")
            else:
                equal += 1
    return equal, bad


# --- the gate -----------------------------------------------------------------------------------

class Stage(NamedTuple):
    name: str
    ok: bool
    detail: str


def _delta(a: torch.Tensor, b: torch.Tensor) -> str:
    if a.shape != b.shape:
        return f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
    return f"max|d| {float((a - b).abs().max()):.3g}"


def gate_image(variant: str, upstream, ours: vendor.Predictor, rgb: np.ndarray,
               autocast: bool) -> list[Stage]:
    stages: list[Stage] = []
    # 1. preprocessing, exactly as inference() calls it
    up_cpu, _, _ = upstream._preprocess_inputs([rgb], None, None, 504, "upper_bound_resize")
    up_input, _, _ = upstream._prepare_model_inputs(up_cpu, None, None)
    our_input = ours.preprocess(rgb)
    same = up_input.shape == our_input.shape and torch.equal(up_input, our_input)
    stages.append(Stage("input", same, f"{tuple(our_input.shape[-2:])}"))

    # 2. every shared module, fp32, then the final depth
    shared = ({n[len("model."):] for n, _ in upstream.named_modules() if n.startswith("model.")}
              & {n for n, _ in ours.model.named_modules()}) - {"", "head"}
    # "head" itself is left out: upstream's any-view head also returns the ray branch mozo does
    # not build. Everything inside it that both have is compared.
    with torch.inference_mode():
        torch.manual_seed(0)
        with recording(upstream, "model.", shared) as theirs:
            raw = upstream.model(up_input, None, None, [], False, False, "saddle_balanced")
        up_depth = torch.from_numpy(upstream._convert_to_prediction(raw).depth[0])
        with recording(ours.model, "", shared) as mine:
            our_depth = ours.predict(rgb)
    equal, bad = compare_calls(mine, theirs)
    stages.append(Stage("modules", not bad, f"{equal} calls of {len(set(mine) & set(theirs))} "
                        "modules equal"
                        + (f"; first differing: {bad[:3]}" if bad else "")))
    same = up_depth.shape == our_depth.shape and torch.equal(up_depth, our_depth)
    stages.append(Stage("depth fp32", same, "" if same else _delta(up_depth, our_depth)))

    # 3. upstream's unmodified API, under its own autocast
    if autocast:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        torch.manual_seed(0)
        api_depth = torch.from_numpy(upstream.inference([rgb]).depth[0])
        with torch.autocast(device_type="cpu", dtype=dtype):
            our_auto = ours.predict(rgb)
        same = api_depth.shape == our_auto.shape and torch.equal(api_depth, our_auto)
        detail = str(dtype).replace("torch.", "") if same else _delta(api_depth, our_auto)
        stages.append(Stage("depth autocast", same, detail))

    # 5. determinism, and the shape-only tensors a graph is handed: computed beforehand, they
    #    must give exactly what the network computes for itself.
    again = ours.predict(rgb)
    stages.append(Stage("deterministic", torch.equal(again, our_depth), ""))
    with torch.inference_mode():
        a = ours.model(our_input)
        b = ours.model(our_input, ours.model.shape_inputs(*our_input.shape[-2:]))
    stages.append(Stage("shape inputs", all(torch.equal(a[k], b[k]) for k in a), ""))
    return stages


def gate_mozo(variant: str, ours: vendor.Predictor, images: list[np.ndarray]) -> Stage:
    from mozo.adapters.depth_anything_v3 import DepthAnythingV3Predictor

    model = DepthAnythingV3Predictor(variant, device="cpu", runtime="torch-fp32")
    realloc(model._predictor.model)
    worst, failures = 0.0, 0
    for rgb in images:
        expected = (1.0 / vendor.resize(ours.predict(rgb), rgb.shape[:2])).numpy()
        got = model.predict(rgb)
        if got.shape != expected.shape or not np.array_equal(got, expected):
            failures += 1
            if got.shape == expected.shape:
                worst = max(worst, float(np.abs(got - expected).max()))
    return Stage("mozo", failures == 0,
                 f"{len(images) - failures}/{len(images)} images" +
                 (f", max|d| {worst:.3g}" if failures else ""))


# --- falsification ------------------------------------------------------------------------------

class Falsification(NamedTuple):
    name: str
    #: The first stage that must fail; every stage before it must still pass.
    stage: str
    applies: Callable[[str], bool]
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
    real = vendor.dualdpt.uv_embedding
    return _setattr(vendor.dualdpt, "uv_embedding",
                    lambda *a, ratio=0.1: real(*a[:7], ratio=(a[7] if len(a) > 7 else ratio)
                                               * 1.0000001))


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
    real = vendor.net.compute_sky_mask
    return _setattr(vendor.net, "compute_sky_mask",
                    lambda sky, threshold=0.3: real(sky, threshold=0.31))


def _camera_token_unused(p):
    real = p.model.backbone.pretrained.camera_token
    return _setattr(p.model.backbone.pretrained, "camera_token",
                    torch.nn.Parameter(real.detach().flip(1)))


FALSIFICATIONS = (
    Falsification("block norm eps 1e-6 -> 1e-5", "modules", lambda v: True, _first_block_norm_eps),
    Falsification("final norm eps 1e-5 -> 1e-6", "modules", lambda v: True, _final_norm_eps),
    Falsification("pos-embed interpolate_offset 0.1 -> 0", "modules", lambda v: True,
                  _no_interpolate_offset),
    Falsification("INTER_AREA -> INTER_LINEAR", "input", lambda v: True, _area_to_linear),
    Falsification("multiple-of-14 ties round down", "input", lambda v: True, _ties_round_down),
    Falsification("RoPE frequency 100 -> 101", "modules", lambda v: v != "mono-large",
                  _rope_frequency),
    Falsification("QK-norm eps 1e-5 -> 1e-6", "modules", lambda v: v != "mono-large",
                  _qk_norm_eps),
    Falsification("reference/source camera tokens swapped", "modules",
                  lambda v: v != "mono-large", _camera_token_unused),
    Falsification("head UV pos-embed ratio x1.0000001", "modules", lambda v: v != "mono-large",
                  _pos_embed_ratio),
    Falsification("sky seed 0 -> 1", "depth fp32", lambda v: v == "mono-large", _sky_seed),
    Falsification("sky threshold 0.3 -> 0.31", "depth fp32", lambda v: v == "mono-large",
                  _sky_threshold),
)


#: The images falsification runs on: a photograph (shrunk with INTER_AREA), one with sky over
#: more than 100,000 other pixels (the sky sample), and a tie in the multiple-of-14 rounding.
FALSIFY_IMAGES = ("example.jpg", "pixabay_cc0_jinyun-mountain-2783957_1280.jpg",
                  "tie-504x217 (rounds 217 up)")


def falsify(variant: str, upstream, ours: vendor.Predictor, images: list[np.ndarray]) -> bool:
    order = ["input", "modules", "depth fp32"]
    caught_all = True
    for f in FALSIFICATIONS:
        if not f.applies(variant):
            continue
        with f.patch(ours):
            results = [s for rgb in images
                       for s in gate_image(variant, upstream, ours, rgb, autocast=False)]
        failed = {s.name for s in results if not s.ok and s.name in order}
        first = next((name for name in order if name in failed), None)
        ok = first == f.stage
        caught_all &= ok
        print(f"  falsify {f.name:42s} first failing stage: {first or 'none':10s} "
              f"{'caught' if ok else 'NOT CAUGHT AS EXPECTED (want ' + f.stage + ')'}")
    return caught_all


# --- main ---------------------------------------------------------------------------------------

def read_images(paths: list[Path]) -> list[tuple[str, np.ndarray]]:
    out = []
    for path in paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"cannot read {path}")
        out.append((path.name, np.ascontiguousarray(bgr[..., ::-1])))
    return out


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
    parser.add_argument("--falsify", action="store_true")
    parser.add_argument("--no-autocast", action="store_true")
    parser.add_argument("--no-mozo", action="store_true")
    parser.add_argument("--allow-other-commit", action="store_true")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    problems = check_standalone()
    print(f"standalone: {'PASS' if not problems else 'FAIL'}")
    for p in problems:
        print("  ", p)

    DepthAnything3, commit = load_upstream_package(args.upstream, args.allow_other_commit)
    import torchvision
    print(f"upstream {commit}; torch {torch.__version__}, torchvision {torchvision.__version__}, "
          f"numpy {np.__version__}, cv2 {cv2.__version__}; MKL_CBWR={os.environ['MKL_CBWR']}")

    paths = []
    for item in args.images or [ROOT / "tests" / "fixtures" / "images"]:
        paths += sorted(p for p in item.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                        ) if item.is_dir() else [item]
    images = read_images(paths) + synthetic()

    all_ok = not problems
    for variant in args.variants:
        repo, _ = UPSTREAM[variant]
        upstream = realloc(DepthAnything3.from_pretrained(str(args.checkpoints / repo)).eval())
        # The artifact mozo publishes, through mozo's resolver (MOZO_BASE_URL may point at a
        # staged weights/ tree).
        weights = resolve("depth_anything_v3", variant, "torch-fp32")
        ours = vendor.Predictor.from_pretrained(variant, weights=weights)
        realloc(ours.model)
        digest = hashlib.sha256(weights.read_bytes()).hexdigest()[:12]
        print(f"\n== {variant} ({repo}, mozo weights sha256 {digest}...)")
        counts: dict[str, list[int]] = {}
        for name, rgb in images:
            try:
                stages = gate_image(variant, upstream, ours, rgb, autocast=not args.no_autocast)
            except Exception as error:  # noqa: BLE001 - reported as a failing image
                stages = [Stage("error", False, repr(error))]
            line = "  ".join(f"{s.name} {'ok' if s.ok else 'FAIL'}"
                             + (f" ({s.detail})" if s.detail and (not s.ok or s.name in
                                                                  {"input", "modules"}) else "")
                             for s in stages)
            print(f"  {name:34s} {line}")
            for s in stages:
                counts.setdefault(s.name, [0, 0])
                counts[s.name][0] += s.ok
                counts[s.name][1] += 1
        if not args.no_mozo:
            stage = gate_mozo(variant, ours, [rgb for _, rgb in images])
            print(f"  mozo adapter: {'ok' if stage.ok else 'FAIL'} ({stage.detail})")
            counts["mozo"] = [int(stage.ok), 1]
        ok = all(passed == total for passed, total in counts.values())
        print(f"  {variant}: " + ", ".join(f"{k} {p}/{t}" for k, (p, t) in counts.items())
              + f" -> {'PASS' if ok else 'FAIL'}")
        all_ok &= ok
        if args.falsify:
            chosen = [rgb for name, rgb in images if name in FALSIFY_IMAGES]
            if len(chosen) != len(FALSIFY_IMAGES):
                raise SystemExit(f"--falsify needs {FALSIFY_IMAGES} among the images")
            all_ok &= falsify(variant, upstream, ours, chosen)
        del upstream, ours

    print(f"\n{'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

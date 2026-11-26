#!/usr/bin/env python3
"""Check that mozo's PP-OCRv6 returns exactly what PaddleOCR does with the same networks.

Three paths reach the same weights and this compares all three, stage by stage, with no
tolerance anywhere.

``mozo/vendors/ppocrv6_deploy`` is one. **PaddleOCR 3.7.0 is the second**: its own OCR pipeline
(``PaddleOCR(engine="paddle")``: PaddleX's resize, normalise, DB post-processing with pyclipper,
sorting, cropping, batching and CTC decoding), with each network forward handed to the PyTorch
implementation PaddlePaddle publishes in transformers 5.20.2, on PaddlePaddle's own
``model.safetensors``. And mozo itself is the third: registry, weights, adapter, PixelFlow.

**Why that reference, and not either PaddleOCR engine as it ships.** ``engine="paddle"`` runs the
networks in Paddle, whose arithmetic is not torch's -- the probability maps differ by ~1e-4 at the
same weights -- so nothing torch computes can equal it bit for bit. ``engine="transformers"`` is
torch end to end, but its pre- and post-processing are different code from the Paddle pipeline's
(torchvision resizing, a re-derived unclip, and every line in a batch stretched to the widest
one's width), and it is the Paddle pipeline whose numbers PaddleOCR publishes. So the gate holds
every stage to upstream code: the networks to transformers' modules, everything around them to
PaddleX's. What Paddle's own arithmetic changes is measured, not gated, by tools/bench/ppocrv6.py.

**pyclipper is the reference here and absent from mozo.** The vendor reimplements Clipper's
round-join offset; ``--fuzz N`` additionally checks it against pyclipper on N random boxes.

**On CUDA (the primary gate) and on the CPU.** Both sides run in one process on one device, in
fp32, mozo's only torch precision for this family. On CUDA the global switches stay as mozo's
runtime leaves them (TF32 on for cuDNN convolutions, off for matmuls, ``cudnn.benchmark`` off,
every SDPA backend allowed), each side first runs twice and must repeat itself byte for byte, and
the GPU and the stack are printed with the result. The reference's network forwards move to the
GPU the way PaddleX's own transformers engine moves them: the model once, each batch with
``.to(device)``.

``--falsify`` re-runs the vendor under deliberate perturbations and requires each that applies
to fail at the stage it should; four of them change only the hardware's arithmetic (TF32
matmuls, the SDPA kernel, the attention written out, channels-last inputs). ``--reference-cache
DIR`` keeps the reference's records and reuses them across runs.

Run from the repository root, in an environment with the reference installed::

    pip install paddleocr==3.7.0 paddlex==3.7.2 paddlepaddle==3.3.1 transformers==5.20.2
    python tools/verify/ppocrv6.py --device cuda --all --images icdar2015/ --limit 10 --falsify
    python tools/verify/ppocrv6.py                 # medium, on the CPU
    python tools/verify/ppocrv6.py --all --fuzz 100000
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import pickle
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))

from mozo.image import load_image  # noqa: E402
from mozo.vendors.ppocrv6_deploy import SPECS, VARIANTS, Reader  # noqa: E402
from mozo.vendors.ppocrv6_deploy import boxes as _boxes  # noqa: E402
from mozo.vendors.ppocrv6_deploy import image as _image  # noqa: E402
from mozo.weights import resolve  # noqa: E402

from conftest import FIXTURE, TEXT_FIXTURES  # noqa: E402

# The Hub commits tools/fetch/ppocrv6.py fuses, so the reference loads the same tensors.
from fetch.ppocrv6 import COMMITS, _repo  # noqa: E402


#: Extra pages from ``--images``: the fixtures have at most three lines, so a dense page is
#: what exercises more than one recogniser batch and hundreds of unclips.
EXTRA: list[Path] = []


def images() -> list[Path]:
    rendered = sorted(TEXT_FIXTURES.glob("*.png"))
    return rendered + ([FIXTURE] if FIXTURE.is_file() else []) + EXTRA


class Comparison:
    def __init__(self, quiet: bool = False) -> None:
        self.failures: list[str] = []
        self.count = 0
        self.quiet = quiet

    def check(self, name: str, want, got) -> bool:
        self.count += 1
        same = _identical(want, got)
        if not same:
            self.failures.append(name)
            if not self.quiet:
                print(f"    {name:60s} DIFFERS  {_delta(want, got)}")
        return same

    def note(self, name: str, text: str) -> None:
        print(f"    {name:52s} {text}")


def _identical(want, got) -> bool:
    if isinstance(want, (str, float, int)) or isinstance(got, (str, float, int)):
        return type(want) is type(got) and want == got
    if isinstance(want, (list, tuple)) and isinstance(got, (list, tuple)):
        return len(want) == len(got) and all(_identical(a, b) for a, b in zip(want, got))
    a, b = np.asarray(want), np.asarray(got)
    return a.shape == b.shape and a.dtype == b.dtype and bool(np.array_equal(a, b))


def _delta(want, got) -> str:
    try:
        a, b = np.asarray(want), np.asarray(got)
        if a.shape != b.shape:
            return f"shape {a.shape} vs {b.shape}"
        if a.dtype != b.dtype:
            return f"dtype {a.dtype} vs {b.dtype}"
        return f"max|delta| = {np.abs(a.astype(float) - b.astype(float)).max()}"
    except (ValueError, TypeError):
        return f"{want!r} vs {got!r}"


def _transformers(variant: str, kind: str):
    """PaddlePaddle's PyTorch implementation on its own checkpoint, at the pinned commit."""
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForObjectDetection, AutoModelForTextRecognition

    path = snapshot_download(_repo(variant, kind), revision=COMMITS[(variant, kind)])
    cls = AutoModelForObjectDetection if kind == "det" else AutoModelForTextRecognition
    return cls.from_pretrained(path).eval()


class _TorchRunner:
    """Stands in for PaddleX's Paddle runner: ``runner(x=[batch]) -> [output]``.

    On a GPU it does what PaddleX's own transformers engine does (``TransformersPredictor``): the
    model moved to the device once, each batch moved to it with ``.to(device)``, the forward under
    ``inference_mode``."""

    def __init__(self, model, device: str) -> None:
        self.model = model.to(device)
        self.device = device
        self.outputs: list[np.ndarray] = []

    def __call__(self, x):
        pixels = torch.from_numpy(np.ascontiguousarray(x[0])).to(self.device)
        with torch.inference_mode():
            out = self.model(pixel_values=pixels)
        array = out.last_hidden_state.cpu().numpy()
        self.outputs.append(array)
        return [array]


def upstream_pipeline(variant: str, device: str):
    """PaddleOCR 3.7.0's pipeline at the defaults its model cards document, Paddle runners out.

    Paddle itself stays on the CPU: it only loads the graphs whose runners are replaced."""
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "huggingface")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(text_detection_model_name=f"PP-OCRv6_{variant}_det",
                    text_recognition_model_name=f"PP-OCRv6_{variant}_rec",
                    engine="paddle", device="cpu", enable_mkldnn=False,
                    use_doc_orientation_classify=False, use_doc_unwarping=False,
                    use_textline_orientation=False)
    pipeline = ocr.paddlex_pipeline._pipeline
    det = _TorchRunner(_transformers(variant, "det"), device)
    rec = _TorchRunner(_transformers(variant, "rec"), device)
    pipeline.text_det_model.runner = det
    pipeline.text_rec_model.runner = rec
    return ocr, pipeline, det, rec


#: One run of one side: for each page, every recorded stage by name, in pipeline order.
Record = dict[str, dict[str, object]]


def run_upstream(variant: str, device: str) -> Record:
    """Every stage of the reference on every page. Recorded once and compared against as often
    as needed: upstream does not change between falsifications."""
    ocr, pipeline, det_runner, rec_runner = upstream_pipeline(variant, device)
    det = pipeline.text_det_model
    post = det.post_op
    cases: Record = {}
    for path in images():
        bgr = np.ascontiguousarray(load_image(path)[..., ::-1])
        stages: dict[str, object] = {}

        imgs, shapes = det.pre_tfs["Resize"](
            imgs=[bgr], limit_side_len=pipeline.text_det_limit_side_len,
            limit_type=pipeline.text_det_limit_type, max_side_limit=pipeline.text_det_max_side_limit)
        x = det.pre_tfs["ToBatch"](imgs=det.pre_tfs["ToCHW"](imgs=det.pre_tfs["Normalize"](imgs=imgs)))[0]
        stages["detector input"] = x
        stages["resize record"] = shapes[0]
        prob = det_runner([x])[0]
        stages["probability map"] = prob

        # Every candidate that reaches the unclip, through pyclipper: DBPostProcess's own steps.
        contours, _ = cv2.findContours(((prob[0][0] > pipeline.text_det_thresh) * 255).astype(np.uint8),
                                       cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        n = 0
        for contour in contours[:post.max_candidates]:
            points, sside = post.get_mini_boxes(contour)
            if sside < post.min_size:
                continue
            points = np.array(points)
            stages[f"unclip {n} box"] = points
            stages[f"unclip {n}"] = post.unclip(points, pipeline.text_det_unclip_ratio)
            n += 1

        quads, scores = post([prob], shapes, thresh=pipeline.text_det_thresh,
                             box_thresh=pipeline.text_det_box_thresh,
                             unclip_ratio=pipeline.text_det_unclip_ratio)
        stages["quads"] = quads[0]
        stages["box scores"] = [float(s) for s in scores[0]]
        ordered = pipeline._sort_boxes(quads[0]) if len(quads[0]) else []
        stages["sorted quads"] = list(ordered)
        for i, crop in enumerate(pipeline._crop_by_polys(bgr, ordered) if len(ordered) else []):
            stages[f"line {i} crop"] = crop

        # The whole pipeline, end to end.
        rec_runner.outputs.clear()
        result = next(iter(ocr.predict(bgr))).json["res"]
        for i, probs in enumerate(rec_runner.outputs):
            stages[f"rec batch {i} probabilities"] = probs
        stages["region count"] = len(result["rec_texts"])
        for i, text in enumerate(result["rec_texts"]):
            stages[f"region {i} quad"] = np.asarray(result["rec_polys"][i]).tolist()
            stages[f"region {i} text"] = text
            stages[f"region {i} confidence"] = float(result["rec_scores"][i])
        cases[path.name] = stages
    return cases


def run_vendor(variant: str, checkpoint: Path, device: str) -> Record:
    """The vendor's same stages, keyed as :func:`run_upstream` keys them. Reads the module
    constants at call time, so a falsification patched around this call is seen."""
    mine = Reader(checkpoint, SPECS[variant], device=device)
    cases: Record = {}
    for path in images():
        rgb = load_image(path)
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        stages: dict[str, object] = {}

        page, shape = _image.resize_for_detector(bgr)
        x = _image.normalize_for_detector(page)[None].astype(np.float32, copy=False)
        stages["detector input"] = x
        stages["resize record"] = shape
        prob = mine.detect(np.ascontiguousarray(x))
        stages["probability map"] = prob

        contours, _ = cv2.findContours(((prob[0][0] > _boxes.THRESH) * 255).astype(np.uint8),
                                       cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        n = 0
        for contour in contours[:_boxes.MAX_CANDIDATES]:
            points, sside = _boxes.mini_box(contour)
            if sside < _boxes.MIN_SIZE:
                continue
            points = np.array(points)
            stages[f"unclip {n} box"] = points
            stages[f"unclip {n}"] = _boxes.unclip(points, _boxes.UNCLIP_RATIO)
            n += 1

        quads, scores = _boxes.quads(prob[0][0], shape)
        stages["quads"] = quads
        stages["box scores"] = [float(s) for s in scores]
        ordered = _boxes.sort_quads(quads) if len(quads) else []
        stages["sorted quads"] = list(ordered)
        for i, quad in enumerate(ordered):
            stages[f"line {i} crop"] = _image.crop_line(bgr, quad)

        batches = []
        recognise = mine.recognise

        def record(batch):
            batches.append(recognise(batch))
            return batches[-1]

        mine.recognise = record
        regions = mine(rgb)
        mine.recognise = recognise
        for i, probs in enumerate(batches):
            stages[f"rec batch {i} probabilities"] = probs
        stages["region count"] = len(regions)
        for i, region in enumerate(regions):
            stages[f"region {i} quad"] = region.quad.tolist()
            stages[f"region {i} text"] = region.text
            stages[f"region {i} confidence"] = region.confidence
        stages["_read"] = " | ".join(repr(r.text) for r in regions) or "(nothing)"
        cases[path.name] = stages
    return cases


def compare(want: Record, got: Record, comparison: Comparison, label: str = "") -> None:
    """Every stage of ``got`` against ``want``, page by page; a stage on one side only fails."""
    for page, expected in want.items():
        found = got.get(page, {})
        before = len(comparison.failures)
        keys = list(expected) + [k for k in found if k not in expected]
        checked = 0
        for key in keys:
            if key.startswith("_"):
                continue
            checked += 1
            name = f"{label}{page}: {key}"
            if key not in expected or key not in found:
                comparison.count += 1
                comparison.failures.append(name)
                if not comparison.quiet:
                    print(f"    {name:60s} only on {'our' if key in found else 'upstream'} side")
            else:
                comparison.check(name, expected[key], found[key])
        if not comparison.quiet:
            first = comparison.failures[before] if len(comparison.failures) > before else None
            print(f"  {label}{page:28s} {checked:5d} stages  "
                  f"{'identical' if first is None else 'FIRST DIFFERS: ' + first}")
            if "_read" in found:
                print(f"      read: {found['_read']}")
        if comparison.quiet and len(comparison.failures) > before:
            return


def fuzz_unclip(count: int, comparison: Comparison, seed: int = 0) -> None:
    """Random boxes in the shapes DB produces, through pyclipper and through the vendor."""
    from paddlex.inference.models.text_detection.processors import DBPostProcess

    post = DBPostProcess()
    rng = np.random.default_rng(seed)
    bad = 0
    for _ in range(count):
        cx, cy = rng.uniform(0, 4000, 2)
        w = rng.choice([rng.uniform(3, 12), rng.uniform(3, 1500)])
        h = rng.choice([rng.uniform(3, 8), rng.uniform(3, 200)])
        angle = rng.choice([0.0, 90.0, rng.uniform(-90, 90)])
        points, _ = post.get_mini_boxes(
            cv2.boxPoints(((cx, cy), (w, h), angle)).reshape(-1, 1, 2).astype(np.float32))
        points = np.array(points)
        ratio = float(rng.choice([1.5, rng.uniform(0.5, 3.0)]))
        want, got = post.unclip(points, ratio), _boxes.unclip(points, ratio)
        if not _identical(want, got):
            bad += 1
            if bad <= 5:
                print(f"    unclip differs on {points.tolist()} ratio {ratio}")
    comparison.count += count
    if bad:
        comparison.failures.extend(["fuzz"] * bad)
    comparison.note(f"pyclipper fuzz ({count} boxes)", f"{bad} differ")


def against_mozo(variant: str, device: str, comparison: Comparison) -> None:
    """The vendor against mozo's adapter, through PixelFlow's rounding (2 dp, 3 dp)."""
    from mozo.adapters.ppocrv6 import PPOCRv6Predictor

    model = PPOCRv6Predictor(variant, device=device, runtime="torch-fp32")
    reader = Reader(resolve("ppocrv6", variant, "torch-fp32"), SPECS[variant], device=device)
    for path in images():
        expected = reader(load_image(path))
        found = model.predict(path)
        print(f"  {path.name}")
        if not comparison.check("adapter count", len(expected), len(found)):
            continue
        for i, (region, detection) in enumerate(zip(expected, found)):
            quad = np.asarray(region.quad, dtype=float)
            hull = [round(float(v), 2) for v in
                    (quad[:, 0].min(), quad[:, 1].min(), quad[:, 0].max(), quad[:, 1].max())]
            comparison.check(f"adapter {i} bbox", hull, list(detection.bbox))
            comparison.check(f"adapter {i} segments", quad.round(2), np.asarray(detection.segments))
            comparison.check(f"adapter {i} text", region.text, detection.text)
            comparison.check(f"adapter {i} confidence", round(region.confidence, 3),
                             detection.confidence)


def repeatable(name: str, first: Record, second: Record) -> list[str]:
    """Two runs of one side, byte for byte (VENDORING 2.7): without this a GPU mismatch could be
    noise rather than a difference."""
    comparison = Comparison(quiet=True)
    for page, stages in first.items():
        for key, value in stages.items():
            if key.startswith("_"):
                continue
            if not _identical(value, second[page].get(key)):
                comparison.failures.append(f"{page}: {key}")
            comparison.count += 1
    total = comparison.count
    print(f"  {name}: run twice, {total - len(comparison.failures)}/{total} stages byte-identical")
    return comparison.failures


def upstream_record(variant: str, device: str, cache: Path | None) -> tuple[Record, list[str]]:
    """The reference, from ``cache`` when it holds this (variant, device, page set), else run --
    twice on a GPU, which must agree byte for byte -- and saved there."""
    key = hashlib.sha256("\n".join(p.name for p in images()).encode()).hexdigest()[:12]
    file = cache / f"ppocrv6-{variant}-{device}-fp32-{key}.pkl" if cache else None
    if file is not None and file.is_file():
        print(f"  reference from {file} (checked for repeatability when it was recorded)")
        return pickle.loads(file.read_bytes()), []
    record = run_upstream(variant, device)
    problems = []
    if device != "cpu":
        problems = repeatable("upstream", record, run_upstream(variant, device))
    if file is not None and not problems:
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(pickle.dumps(record))
    return record, problems


# -- falsification ----------------------------------------------------------------------------


@contextlib.contextmanager
def _patched(target, name: str, value):
    old = getattr(target, name)
    setattr(target, name, value)
    try:
        yield
    finally:
        setattr(target, name, old)


def _rotated_union(polygon):
    out = _UNION(polygon)
    return out[1:] + out[:1]


def _float64_decode(self, probs):
    out = []
    for index, prob in zip(probs.argmax(axis=-1), probs.max(axis=-1)):
        keep = np.ones(len(index), dtype=bool)
        keep[1:] = index[1:] != index[:-1]
        keep &= index != 0
        kept = prob[keep].astype(np.float64)
        out.append(("".join(self.symbols[i] for i in index[keep]),
                    np.mean(kept if len(kept) else [0]).tolist()))
    return out


@contextlib.contextmanager
def _tf32_matmuls():
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


def _math_attention():
    from torch.nn.attention import SDPBackend, sdpa_kernel
    return sdpa_kernel(SDPBackend.MATH)


def _eager_forward(self, x):
    """Attention written out, ``softmax(q k^T * scale) v``: the same function, other arithmetic."""
    batch, length, dim = x.size()
    q, k, v = self.qkv(x).reshape(batch, length, 3, self.heads, dim // self.heads).permute(2, 0, 3, 1, 4)
    out = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1) @ v
    return self.projection(out.transpose(1, 2).reshape(batch, length, -1))


def _default_attention_is_math(variant: str, device: str) -> bool:
    """Whether SDPA already falls back to its math kernel at this variant's head size: then
    forcing it changes nothing, and there is nothing to catch. On CUDA, fp32 attention has no
    flash kernel and the memory-efficient one wants a head size divisible by 8 (small's is 15)."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    rec = SPECS[variant].recogniser
    q = torch.zeros(1, rec.heads, 4, rec.hidden_size // rec.heads, device=device)
    try:
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
                          SDPBackend.CUDNN_ATTENTION]):
            torch.nn.functional.scaled_dot_product_attention(q, q, q)
    except RuntimeError:
        return device != "cpu"
    return False


def _channels_last_call(module, device):
    def run(batch):
        with torch.inference_mode():
            x = torch.from_numpy(batch).to(device).contiguous(memory_format=torch.channels_last)
            return module(x).float().cpu().numpy()
    return run


_UNION = _boxes._union


def falsifications(variant: str, device: str):
    """(what, the stage that must fail first, the patch, or None and why it does not apply).
    Each is applied around a vendor run only; the reference is the recorded one. The last four
    change only the arithmetic the hardware does -- matmul precision, the attention kernel, the
    attention written out, the memory layout cuDNN sees -- on the vendor's side."""
    from mozo.vendors.ppocrv6_deploy import predictor as _predictor
    from mozo.vendors.ppocrv6_deploy import rec as _rec
    from mozo.vendors.ppocrv6_deploy import text as _text

    cases = [
        ("B mean 0.485 -> 0.486", "detector input",
         lambda: _patched(_image, "_MEAN", (0.486, 0.456, 0.406))),
        ("LayerNorm eps 1e-6 -> 1e-5", "rec batch",
         lambda: _patched(_rec, "_LAYER_NORM_EPS", 1e-5)),
        ("Clipper arc tolerance 0.25 -> 0.3", "unclip",
         lambda: _patched(_boxes, "_ARC_TOLERANCE", 0.3)),
        ("union output rotated by one vertex", "unclip",
         lambda: _patched(_boxes, "_union", _rotated_union)),
        ("max_candidates 3000 -> 10", "unclip",
         lambda: _patched(_boxes, "MAX_CANDIDATES", 10)),
        ("confidence mean in float64", "confidence",
         lambda: _patched(_text.Alphabet, "decode", _float64_decode)),
        ("recogniser batch 6 -> 5", "rec batch",
         lambda: _patched(_predictor, "REC_BATCH", 5)),
        ("TF32 matmuls on", "rec batch",
         _tf32_matmuls if device != "cpu" else "TF32 is a CUDA setting"),
        ("SDPA forced to the math kernel", "rec batch", _math_attention),
        ("attention written out instead of SDPA", "rec batch",
         lambda: _patched(_rec.Attention, "forward", _eager_forward)),
        ("network inputs channels-last", "probability map",
         lambda: _patched(_predictor, "_torch_call", _channels_last_call)),
    ]
    svtr_only = ("LayerNorm", "SDPA", "attention")
    out = []
    for what, stage, patch in cases:
        if SPECS[variant].recogniser.head != "svtr" and what.startswith(svtr_only):
            patch = f"{variant}'s recogniser has no SVTR encoder: no LayerNorm, no attention"
        elif what.startswith("SDPA") and _default_attention_is_math(variant, device):
            patch = (f"SDPA already runs its math kernel at {variant}'s head size on {device} "
                     "(no fp32 flash kernel; the efficient one needs a multiple of 8)")
        out.append((what, stage, patch))
    return out


def falsify(variant: str, checkpoint: Path, upstream: Record, device: str) -> list[str]:
    """Each perturbation must fail, and the first failing stage must be the one named."""
    failures = []
    print(f"\n=== falsification, {variant} on {device}: each must be caught ===")
    for what, stage, patch in falsifications(variant, device):
        if isinstance(patch, str):
            print(f"  {what:38s} does not apply: {patch}")
            continue
        comparison = Comparison(quiet=True)
        with patch():
            compare(upstream, run_vendor(variant, checkpoint, device), comparison)
        first = comparison.failures[0] if comparison.failures else None
        if first is None:
            failures.append(f"{what}: not caught")
            print(f"  {what:38s} NOT CAUGHT")
        elif stage not in first.split(": ", 1)[1]:
            failures.append(f"{what}: caught at {first}, expected {stage}")
            print(f"  {what:38s} caught, but first at {first} (expected {stage})")
        else:
            print(f"  {what:38s} caught, first at {first}")
    return failures


def environment(device: str) -> None:
    import paddleocr
    import paddlex
    import transformers
    print(f"torch {torch.__version__}, transformers {transformers.__version__}, "
          f"paddleocr {paddleocr.__version__}, paddlex {paddlex.__version__}, cv2 {cv2.__version__}, "
          f"numpy {np.__version__}")
    if device == "cpu":
        print("device cpu")
        return
    import subprocess
    driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                            capture_output=True, text=True).stdout.strip().splitlines()
    print(f"device {device}: {torch.cuda.get_device_name(device)}, CUDA {torch.version.cuda}, "
          f"cuDNN {torch.backends.cudnn.version()}, driver {driver[0] if driver else '?'}")
    from torch.backends import cuda as cuda_backends
    print(f"  as mozo leaves them: cudnn.allow_tf32 {torch.backends.cudnn.allow_tf32}, "
          f"matmul.allow_tf32 {cuda_backends.matmul.allow_tf32}, "
          f"cudnn.benchmark {torch.backends.cudnn.benchmark}, "
          f"cudnn.deterministic {torch.backends.cudnn.deterministic}, "
          f"deterministic algorithms {torch.are_deterministic_algorithms_enabled()}, "
          f"SDPA flash/efficient/math {cuda_backends.flash_sdp_enabled()}/"
          f"{cuda_backends.mem_efficient_sdp_enabled()}/{cuda_backends.math_sdp_enabled()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variants", nargs="*", help=f"default: medium. Known: {VARIANTS}")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                        help="both sides run here, in one process (default: cpu)")
    parser.add_argument("--dtype", choices=("fp32",), default="fp32",
                        help="fp32 only: mozo runs PP-OCRv6 through torch in fp32 alone, on every "
                             "device (its fp16 is a TensorRT graph, not torch)")
    parser.add_argument("--fuzz", type=int, default=0, help="random boxes through pyclipper")
    parser.add_argument("--no-mozo", action="store_true", help="skip the adapter comparison")
    parser.add_argument("--images", type=Path, help="a folder of extra pages, e.g. ICDAR 2015")
    parser.add_argument("--limit", type=int, default=20, help="how many of --images")
    parser.add_argument("--falsify", action="store_true",
                        help="also require each deliberate perturbation of the vendor to fail")
    parser.add_argument("--reference-cache", type=Path,
                        help="a folder to keep the reference's records in and reuse them from")
    args = parser.parse_args()
    if args.images:
        EXTRA.extend(sorted(p for p in args.images.iterdir()
                            if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[:args.limit])
    wanted = VARIANTS if args.all else (args.variants or ["medium"])
    unknown = [v for v in wanted if v not in SPECS]
    if unknown:
        raise SystemExit(f"unknown variant(s) {unknown}. Known: {VARIANTS}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda, and torch sees no CUDA device")

    warnings.filterwarnings("ignore")
    environment(args.device)

    comparison = Comparison()
    problems: list[str] = []
    for variant in wanted:
        checkpoint = resolve("ppocrv6", variant, "torch-fp32")
        print(f"\n=== {variant} vs PaddleOCR's pipeline on PaddlePaddle's torch networks, "
              f"{args.device} {args.dtype} ===")
        upstream, unstable = upstream_record(variant, args.device, args.reference_cache)
        mine = run_vendor(variant, checkpoint, args.device)
        if args.device != "cpu":
            unstable += repeatable("mozo", mine, run_vendor(variant, checkpoint, args.device))
        problems += [f"not repeatable: {name}" for name in unstable]
        compare(upstream, mine, comparison)
        if not args.no_mozo:
            print(f"\n=== {variant} vs mozo's adapter, through PixelFlow ===")
            against_mozo(variant, args.device, comparison)
        if args.falsify:
            problems += falsify(variant, checkpoint, upstream, args.device)
    if args.fuzz:
        print("\n=== the unclip against pyclipper ===")
        fuzz_unclip(args.fuzz, comparison)

    print()
    for problem in problems:
        print(f"  {problem}")
    if comparison.failures or problems:
        print(f"{len(comparison.failures)} of {comparison.count} comparisons differ, "
              f"{len(problems)} other problems. PP-OCRv6 ({args.device} {args.dtype}): FAIL")
        for name in comparison.failures[:20]:
            print(f"  {name}")
        return 1
    print(f"{comparison.count} comparisons, every one identical. "
          f"PP-OCRv6 ({args.device} {args.dtype}): PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

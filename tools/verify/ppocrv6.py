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

Run from the repository root, in an environment with the reference installed::

    pip install paddleocr==3.7.0 paddlex==3.7.2 paddlepaddle==3.3.1 transformers==5.20.2
    python tools/verify/ppocrv6.py                 # medium
    python tools/verify/ppocrv6.py --all --fuzz 100000
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from mozo.image import load_image  # noqa: E402
from mozo.vendors.ppocrv6_deploy import SPECS, VARIANTS, Reader  # noqa: E402
from mozo.vendors.ppocrv6_deploy import boxes as _boxes  # noqa: E402
from mozo.vendors.ppocrv6_deploy import image as _image  # noqa: E402
from mozo.weights import resolve  # noqa: E402

from conftest import FIXTURE, TEXT_FIXTURES  # noqa: E402

#: The Hub commits tools/fetch/ppocrv6.py fuses, so the reference loads the same tensors.
from importlib import util as _util  # noqa: E402

_spec = _util.spec_from_file_location("_fetch", ROOT / "tools" / "fetch" / "ppocrv6.py")
_fetch = _util.module_from_spec(_spec)
_spec.loader.exec_module(_fetch)


#: Extra pages from ``--images``: the fixtures have at most three lines, so a dense page is
#: what exercises more than one recogniser batch and hundreds of unclips.
EXTRA: list[Path] = []


def images() -> list[Path]:
    rendered = sorted(TEXT_FIXTURES.glob("*.png"))
    return rendered + ([FIXTURE] if FIXTURE.is_file() else []) + EXTRA


class Comparison:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.count = 0

    def check(self, name: str, want, got) -> bool:
        self.count += 1
        same = _identical(want, got)
        if not same:
            self.failures.append(name)
            print(f"    {name:52s} DIFFERS  {_delta(want, got)}")
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

    repo = f"PaddlePaddle/PP-OCRv6_{variant}_{kind}_safetensors"
    path = snapshot_download(repo, revision=_fetch.COMMITS[(variant, kind)])
    cls = AutoModelForObjectDetection if kind == "det" else AutoModelForTextRecognition
    return cls.from_pretrained(path).eval()


class _TorchRunner:
    """Stands in for PaddleX's Paddle runner: ``runner(x=[batch]) -> [output]``."""

    def __init__(self, model) -> None:
        self.model = model
        self.outputs: list[np.ndarray] = []

    def __call__(self, x):
        with torch.inference_mode():
            out = self.model(pixel_values=torch.from_numpy(np.ascontiguousarray(x[0])))
        array = out.last_hidden_state.numpy()
        self.outputs.append(array)
        return [array]


def upstream_pipeline(variant: str):
    """PaddleOCR 3.7.0's pipeline at the defaults its model cards document, Paddle runners out."""
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "huggingface")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(text_detection_model_name=f"PP-OCRv6_{variant}_det",
                    text_recognition_model_name=f"PP-OCRv6_{variant}_rec",
                    engine="paddle", device="cpu", enable_mkldnn=False,
                    use_doc_orientation_classify=False, use_doc_unwarping=False,
                    use_textline_orientation=False)
    pipeline = ocr.paddlex_pipeline._pipeline
    det, rec = _TorchRunner(_transformers(variant, "det")), _TorchRunner(_transformers(variant, "rec"))
    pipeline.text_det_model.runner = det
    pipeline.text_rec_model.runner = rec
    return ocr, pipeline, det, rec


def against_upstream(variant: str, checkpoint: Path, comparison: Comparison) -> None:
    ocr, pipeline, det_runner, rec_runner = upstream_pipeline(variant)
    det, rec = pipeline.text_det_model, pipeline.text_rec_model
    mine = Reader(checkpoint, SPECS[variant], device="cpu")
    clip_checked = 0

    for path in images():
        rgb = load_image(path)
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        print(f"  {path.name}")

        # -- detector input ----------------------------------------------------------------
        up_imgs, up_shapes = det.pre_tfs["Resize"](
            imgs=[bgr], limit_side_len=pipeline.text_det_limit_side_len,
            limit_type=pipeline.text_det_limit_type, max_side_limit=pipeline.text_det_max_side_limit)
        up_x = det.pre_tfs["ToBatch"](imgs=det.pre_tfs["ToCHW"](
            imgs=det.pre_tfs["Normalize"](imgs=up_imgs)))[0]
        my_page, my_shape = _image.resize_for_detector(bgr)
        my_x = _image.normalize_for_detector(my_page)[None].astype(np.float32, copy=False)
        comparison.check("detector input", up_x, my_x)
        comparison.check("resize record", up_shapes[0], my_shape)

        # -- detector forward ----------------------------------------------------------------
        up_prob = det_runner([up_x])[0]
        my_prob = mine.detect(np.ascontiguousarray(my_x))
        comparison.check("probability map", up_prob, my_prob)

        # -- DB post-processing, with every unclip checked against pyclipper on its own --------
        up_quads, up_scores = det.post_op([up_prob], up_shapes, thresh=pipeline.text_det_thresh,
                                          box_thresh=pipeline.text_det_box_thresh,
                                          unclip_ratio=pipeline.text_det_unclip_ratio)
        my_quads, my_scores = _boxes.quads(my_prob[0][0], my_shape)
        comparison.check(f"quads (n={len(my_quads)})", up_quads[0], my_quads)
        comparison.check("box scores", [float(s) for s in up_scores[0]],
                         [float(s) for s in my_scores])
        clip_checked += _check_unclips(det.post_op, my_prob[0][0], comparison)

        up_sorted = pipeline._sort_boxes(up_quads[0]) if len(up_quads[0]) else []
        my_sorted = _boxes.sort_quads(my_quads) if len(my_quads) else []
        comparison.check("sorted quads", list(up_sorted), list(my_sorted))

        # -- crops ------------------------------------------------------------------------------
        up_crops = list(pipeline._crop_by_polys(bgr, up_sorted)) if len(up_sorted) else []
        my_crops = [_image.crop_line(bgr, q) for q in my_sorted]
        for i, (a, b) in enumerate(zip(up_crops, my_crops)):
            comparison.check(f"line {i} crop", a, b)

        # -- the whole pipeline, end to end ---------------------------------------------------
        rec_runner.outputs.clear()
        result = next(iter(ocr.predict(bgr))).json["res"]
        mine_rec = []
        mine_recognise = mine.recognise
        mine.recognise = lambda batch: mine_rec.append(mine_recognise(batch)) or mine_rec[-1]
        regions = mine(rgb)
        mine.recognise = mine_recognise
        for i, (a, b) in enumerate(zip(rec_runner.outputs, mine_rec)):
            comparison.check(f"rec batch {i} probabilities", a, b)
        if not comparison.check("region count", len(result["rec_texts"]), len(regions)):
            continue
        for i, region in enumerate(regions):
            comparison.check(f"region {i} quad", np.asarray(result["rec_polys"][i]).tolist(),
                             region.quad.tolist())
            comparison.check(f"region {i} text", result["rec_texts"][i], region.text)
            comparison.check(f"region {i} confidence", float(result["rec_scores"][i]),
                             region.confidence)
        comparison.note("read", " | ".join(repr(r.text) for r in regions) or "(nothing)")
    comparison.note("unclips checked against pyclipper", str(clip_checked))


def _check_unclips(post_op, prob: np.ndarray, comparison: Comparison) -> int:
    """Every candidate box on this map that reaches the unclip, through both implementations."""
    contours, _ = cv2.findContours(((prob > _boxes.THRESH) * 255).astype(np.uint8),
                                   cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    n = 0
    for contour in contours[:_boxes.MAX_CANDIDATES]:
        points, sside = post_op.get_mini_boxes(contour)
        if sside < _boxes.MIN_SIZE:
            continue
        points = np.array(points)
        n += 1
        comparison.check(f"unclip {n}", post_op.unclip(points, _boxes.UNCLIP_RATIO),
                         _boxes.unclip(points, _boxes.UNCLIP_RATIO))
    return n


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


def against_mozo(variant: str, comparison: Comparison) -> None:
    """The vendor against mozo's adapter, through PixelFlow's rounding (2 dp, 3 dp)."""
    from mozo.adapters.ppocrv6 import PPOCRv6Predictor

    model = PPOCRv6Predictor(variant, device="cpu", runtime="torch-fp32")
    reader = Reader(resolve("ppocrv6", variant, "torch-fp32"), SPECS[variant], device="cpu")
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variants", nargs="*", help=f"default: medium. Known: {VARIANTS}")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--fuzz", type=int, default=0, help="random boxes through pyclipper")
    parser.add_argument("--no-mozo", action="store_true", help="skip the adapter comparison")
    parser.add_argument("--images", type=Path, help="a folder of extra pages, e.g. ICDAR 2015")
    parser.add_argument("--limit", type=int, default=20, help="how many of --images")
    args = parser.parse_args()
    if args.images:
        EXTRA.extend(sorted(p for p in args.images.iterdir()
                            if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[:args.limit])
    wanted = VARIANTS if args.all else (args.variants or ["medium"])
    unknown = [v for v in wanted if v not in SPECS]
    if unknown:
        raise SystemExit(f"unknown variant(s) {unknown}. Known: {VARIANTS}")

    warnings.filterwarnings("ignore")
    import transformers
    import paddleocr
    import paddlex
    print(f"torch {torch.__version__}, transformers {transformers.__version__}, "
          f"paddleocr {paddleocr.__version__}, paddlex {paddlex.__version__}, cv2 {cv2.__version__}, "
          f"numpy {np.__version__}")

    comparison = Comparison()
    for variant in wanted:
        print(f"\n=== {variant} vs PaddleOCR's pipeline on PaddlePaddle's torch networks ===")
        against_upstream(variant, resolve("ppocrv6", variant, "torch-fp32"), comparison)
        if not args.no_mozo:
            print(f"\n=== {variant} vs mozo's adapter, through PixelFlow ===")
            against_mozo(variant, comparison)
    if args.fuzz:
        print("\n=== the unclip against pyclipper ===")
        fuzz_unclip(args.fuzz, comparison)

    print()
    if comparison.failures:
        print(f"{len(comparison.failures)} of {comparison.count} comparisons differ. PP-OCRv6: FAIL")
        for name in comparison.failures[:20]:
            print(f"  {name}")
        return 1
    print(f"{comparison.count} comparisons, every one identical. PP-OCRv6: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

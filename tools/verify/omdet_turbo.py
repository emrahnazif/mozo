#!/usr/bin/env python3
"""Check that mozo's OmDet-Turbo returns exactly what the published model does.

Three paths reach the same weights. ``mozo/vendors/omdet_turbo_deploy`` is one; ``transformers``'
``OmDetTurboForObjectDetection`` with its ``OmDetTurboProcessor`` (PIL image backend) and
``post_process_grounded_object_detection`` is the second -- the reference, see the vendor's
``PROVENANCE.md`` for why it rather than the authors' repository; mozo itself (registry, weights,
adapter, PixelFlow) is the third.

**Exact, at every stage**, on CUDA (the primary gate) and on the CPU: pixels, token ids and
masks, class embeddings, task states, the three Swin levels after their norms, the three encoder
levels, each of six decoder layers, the final boxes and logits, and the post-processed detections
at two thresholds; then the adapter against the vendor through PixelFlow's rounding. Both sides
run in one process, on one device, in fp32 -- mozo's only torch precision for this family.

On CUDA the global switches stay as mozo's runtime leaves them (TF32 for cuDNN convolutions on,
for matmuls off, ``cudnn.benchmark`` off, every SDPA backend allowed), and each side first runs
twice and must repeat itself byte for byte; the GPU and the stack are printed with the result.

What makes the comparison defined, and is pinned here:

- ``TIMM_FUSED_ATTN`` is removed from the environment before timm is imported: set to 2 it moves
  the reference's Swin onto ``scaled_dot_product_attention``, a different sum.
- ``MKL_CBWR=AVX2,STRICT`` before torch loads: MKL picks sgemm kernels by buffer alignment, so
  CPU torch is not reproducible across processes without it.
- The reference's parameters are re-allocated after loading (``from_pretrained``-style offsets
  change BLAS paths), and its language caches are emptied before every call: upstream caches each
  class prompt across calls, which makes its bits depend on call history.
- The processor is built with ``backend="pil"``; transformers 5's default torchvision backend
  resizes differently from the authors' pipeline (measured with ``--torchvision-distance``).

The reference is built from the published ``config.json`` values (reproduced below) and mozo's
own ``torch-fp32.pth``, its tokenizer from the vocabulary mozo vendors -- so this needs no network.
The vendored vocabulary was checked id for id against the ``vocab.json`` and ``merges.txt`` Om AI
Lab publishes (49,408 entries, 48,894 merges); its bytes are pinned below.

``--falsify`` re-runs the comparison under deliberate perturbations of the vendor and requires
each to fail, and at the stage it should. The reference is recorded once and reused.

Run from the repository root, in an environment with ``transformers`` and ``timm``::

    python tools/verify/omdet_turbo.py --device cuda --falsify
    python tools/verify/omdet_turbo.py
    python tools/verify/omdet_turbo.py --coco /data/coco --images 20 --falsify
"""

from __future__ import annotations

import os

os.environ.pop("TIMM_FUSED_ATTN", None)
os.environ.setdefault("MKL_CBWR", "AVX2,STRICT")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from dataclasses import replace  # noqa: E402
from typing import NamedTuple  # noqa: E402
from unittest import mock  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
os.environ.setdefault("MOZO_BASE_URL", f"file://{ROOT / 'weights'}")

import mozo  # noqa: E402
from conftest import FIXTURE, as_pixelflow_reports  # noqa: E402
from mozo.image import load_image  # noqa: E402
from mozo.vendors.omdet_turbo_deploy import SPECS, Detector  # noqa: E402
from mozo.vendors.omdet_turbo_deploy import decoder as decoder_module  # noqa: E402
from mozo.vendors.omdet_turbo_deploy import encoder as encoder_module  # noqa: E402
from mozo.vendors.omdet_turbo_deploy import postprocess as postprocess_module  # noqa: E402
from mozo.vendors.omdet_turbo_deploy import predictor as predictor_module  # noqa: E402
from mozo.vendors.omdet_turbo_deploy.postprocess import postprocess  # noqa: E402
from mozo.vendors.omdet_turbo_deploy.predictor import task_sentence  # noqa: E402
from mozo.vendors.omdet_turbo_deploy.text import encoder as text_module  # noqa: E402
from mozo.vendors.omdet_turbo_deploy.text.tokenizer import VOCAB_PATH, Tokenizer  # noqa: E402
from mozo.vendors.omdet_turbo_deploy.vision import swin as swin_module  # noqa: E402
from mozo.weights import resolve  # noqa: E402

#: OpenAI's ``bpe_simple_vocab_16e6.txt.gz``, the bytes the vendored tokenizer reads.
VOCAB_SHA256 = "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"

#: Low enough that marginal detections, where implementations diverge first, are compared too.
THRESHOLDS = (0.05, 0.3)

#: The published ``config.json`` of ``omlab/omdet-turbo-swin-tiny-hf`` @ 7fe93cec, minus the
#: entries ``OmDetTurboConfig`` derives or ignores.
CONFIG = {
    "apply_layernorm_after_vision_backbone": True, "backbone": "swin_tiny_patch4_window7_224",
    "backbone_kwargs": {"always_partition": True, "img_size": 640, "out_indices": [1, 2, 3]},
    "use_timm_backbone": True, "use_pretrained_backbone": False, "batch_norm_eps": 1e-05,
    "class_distance_type": "cosine", "class_embed_dim": 512, "conv_norm_activation": "gelu",
    "csp_activation": "silu", "d_model": 256, "decoder_activation": "relu",
    "decoder_dim_feedforward": 2048, "decoder_dropout": 0.0, "decoder_hidden_dim": 256,
    "decoder_num_heads": 8, "decoder_num_layers": 6, "decoder_num_points": 4,
    "disable_custom_kernels": False, "encoder_attention_heads": 8,
    "encoder_dim_feedforward": 2048, "encoder_dropout": 0.0,
    "encoder_feedforward_activation": "relu", "encoder_feedforward_dropout": 0.0,
    "encoder_in_channels": [192, 384, 768], "encoder_layers": 1,
    "encoder_projection_indices": [2], "eval_size": None, "hidden_expansion": 1,
    "image_size": 640, "layer_norm_eps": 1e-05, "learn_initial_query": False,
    "num_feature_levels": 3, "num_queries": 900, "positional_encoding_temperature": 10000,
    "task_encoder_hidden_dim": 1024, "text_config": {"model_type": "clip_text_model"},
    "text_projection_in_dim": 512, "text_projection_out_dim": 512,
    "vision_features_channels": [256, 256, 256],
}

#: The published ``preprocessor_config.json``, as image-processor arguments. Written out rather
#: than read off the vendor, so the reference does not inherit a vendor mistake.
PROCESSOR = {"do_resize": True, "size": {"height": 640, "width": 640}, "resample": 2,
             "do_rescale": False, "do_normalize": True, "image_mean": [123.675, 116.28, 103.53],
             "image_std": [58.395, 57.12, 57.375], "do_pad": False}

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

#: Each exists for a reason a shorter list would miss. One prompt is the only batch-one text
#: forward; the 80 COCO names make the task sentence overflow and truncate; ``"ΑΣ"`` is where
#: Python's lowercase and the reference's disagree (final sigma); the 75-token prompt sits exactly
#: at the cap; a comma inside a phrase changes the task sentence and not the class.
VOCABULARIES = (
    ["person"],
    ["cat", "remote", "person"],
    COCO_NAMES,
    ["Café Terrace", "naïve façade", "ΑΣ", "北京", "a person, holding a mug", "it's a dog!"],
    [" ".join(["word"] * 75)],
)


def images(coco: Path | None, count: int) -> dict[str, np.ndarray]:
    """The fixture photograph at three aspect ratios, three synthetic edge cases, COCO images."""
    full = load_image(str(FIXTURE))
    height, width = full.shape[:2]
    rng = np.random.default_rng(0)
    found = {
        "full": full, "square": full[: min(height, width), : min(height, width)],
        "tall": full[:, : width // 4],
        "black": np.zeros((480, 640, 3), np.uint8),
        "noise": rng.integers(0, 256, (333, 517, 3), dtype=np.uint8),
        "tiny": full[::97, ::97].copy(),
    }
    if coco is not None and count:
        files = sorted((coco / "val2017").glob("*.jpg"))
        for path in random.Random(0).sample(files, count):
            found[path.stem] = load_image(str(path))
    return found


def reference(checkpoint: Path):
    """``transformers``' OmDet-Turbo, its PIL processor, from mozo's weights and vocabulary."""
    from transformers import CLIPTokenizer, OmDetTurboConfig, OmDetTurboForObjectDetection
    from transformers import OmDetTurboProcessor
    from transformers.models.detr import DetrImageProcessorPil

    model = OmDetTurboForObjectDetection(OmDetTurboConfig(**CONFIG)).eval()
    # transformers 5 dropped the ``text_model.`` level from CLIPTextModel and renames the
    # published keys inside ``from_pretrained``; a direct load has to do the same, strictly.
    state = {key.replace("language_backbone.model.text_model.", "language_backbone.model."): value
             for key, value in torch.load(checkpoint, map_location="cpu", weights_only=True).items()}
    model.load_state_dict(state, strict=True)
    with torch.no_grad():
        for module in model.modules():
            for name, tensor in list(module.named_parameters(recurse=False)):
                tensor.data = tensor.data.clone()
            for name, tensor in list(module.named_buffers(recurse=False)):
                if tensor is not None:
                    setattr(module, name, tensor.clone())
    ours = Tokenizer()
    tokenizer = CLIPTokenizer(vocab=ours.encoder, merges=list(ours.ranks),
                              bos_token="<|startoftext|>", eos_token="<|endoftext|>",
                              unk_token="<|endoftext|>", pad_token="<|endoftext|>",
                              model_max_length=77)
    images = DetrImageProcessorPil(**PROCESSOR)
    return model, OmDetTurboProcessor(images, tokenizer)


def _empty_caches(model) -> None:
    for cache in (model.language_cache_class, model.language_cache_prompt):
        cache.cache.clear()
        cache.current_load = 0


def _compare(stages) -> list[str]:
    problems = []
    for name, want, got in stages:
        want, got = torch.as_tensor(want).cpu(), torch.as_tensor(got).cpu()
        if want.shape != got.shape:
            problems.append(f"{name}: shape {tuple(want.shape)} one side, {tuple(got.shape)} the other")
        elif not torch.equal(want, got):
            gap = (want.double() - got.double()).abs().max().item()
            problems.append(f"{name}: differs by {gap:.3e}")
    return problems


def _where(label: str, vocabulary) -> str:
    return f"{label}/{vocabulary[0][:14]}({len(vocabulary)})"


def run_upstream(model, processor, pictures, device) -> dict[str, dict[str, torch.Tensor]]:
    """The reference's every stage on every picture and vocabulary, on the CPU, keyed by case.

    Recorded once and compared against as often as needed: upstream does not change between
    falsifications."""
    from PIL import Image

    recorded: dict[str, object] = {}
    hooks = [model.vision_backbone.register_forward_hook(
                 lambda m, i, o: recorded.__setitem__("vision", o)),
             model.encoder.register_forward_hook(
                 lambda m, i, o: recorded.__setitem__("encoder", o.extracted_states))]
    hooks += [layer.register_forward_hook(lambda m, i, o, k=k: recorded.__setitem__(f"layer{k}", o[0]))
              for k, layer in enumerate(model.decoder.layers)]
    cases = {}
    try:
        for label, pixels in pictures.items():
            for vocabulary in VOCABULARIES:
                _empty_caches(model)
                encoded = processor(Image.fromarray(pixels), text=list(vocabulary),
                                    return_tensors="pt")
                pixel_values = encoded["pixel_values"]
                encoded = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                           for key, value in encoded.items()}
                with torch.no_grad():
                    out = model(**encoded)
                    text = model.get_language_embedding(
                        encoded["classes_input_ids"], encoded["classes_attention_mask"],
                        encoded["tasks_input_ids"], encoded["tasks_attention_mask"],
                        encoded["classes_structure"])
                stages = {
                    "pixels": pixel_values,
                    "class ids": encoded["classes_input_ids"],
                    "task ids": encoded["tasks_input_ids"],
                    "task mask": encoded["tasks_attention_mask"],
                    "class embeddings": text[0], "task states": text[1],
                    **{f"swin level {k}": recorded["vision"][k] for k in range(3)},
                    **{f"encoder level {k}": recorded["encoder"][k] for k in range(3)},
                    **{f"decoder layer {k}": recorded[f"layer{k}"] for k in range(6)},
                    "boxes": out.decoder_coord_logits[0], "logits": out.decoder_class_logits[0],
                }
                for threshold in THRESHOLDS:
                    found = processor.post_process_grounded_object_detection(
                        out, text_labels=list(vocabulary), target_sizes=[pixels.shape[:2]],
                        threshold=threshold)[0]
                    stages.update({f"detections@{threshold} {kind}": found[kind]
                                   for kind in ("boxes", "scores", "labels")})
                cases[_where(label, vocabulary)] = {key: torch.as_tensor(value).cpu()
                                                    for key, value in stages.items()}
    finally:
        for hook in hooks:
            hook.remove()
    return cases


def run_vendor(detector: Detector, pictures) -> dict[str, dict[str, torch.Tensor]]:
    """The vendor's same stages, keyed as :func:`run_upstream` keys them."""
    mine: dict[str, object] = {}
    image_path = detector.image
    hooks = [image_path.vision.register_forward_hook(lambda m, i, o: mine.__setitem__("vision", o)),
             image_path.encoder.register_forward_hook(lambda m, i, o: mine.__setitem__("encoder", o))]
    hooks += [layer.register_forward_hook(lambda m, i, o, k=k: mine.__setitem__(f"layer{k}", o[0]))
              for k, layer in enumerate(image_path.decoder.layers)]
    cases = {}
    try:
        for label, pixels in pictures.items():
            for vocabulary in VOCABULARIES:
                detector._prompts.clear()
                class_ids, _ = detector.tokenizer(list(vocabulary), truncate=False)
                task_ids, task_mask = detector.tokenizer([task_sentence(vocabulary)], truncate=True)
                classes, task = detector.encode_text(vocabulary)
                boxes, logits = detector.raw(pixels, vocabulary)
                stages = {
                    "pixels": predictor_module.preprocess(pixels),
                    "class ids": class_ids, "task ids": task_ids, "task mask": task_mask,
                    "class embeddings": classes, "task states": task,
                    **{f"swin level {k}": mine["vision"][k] for k in range(3)},
                    **{f"encoder level {k}": mine["encoder"][k] for k in range(3)},
                    **{f"decoder layer {k}": mine[f"layer{k}"] for k in range(6)},
                    "boxes": boxes, "logits": logits,
                }
                for threshold in THRESHOLDS:
                    found = postprocess(boxes, logits, pixels.shape[:2], threshold)
                    stages.update({f"detections@{threshold} {kind}": value
                                   for kind, value in zip(("boxes", "scores", "labels"), found)})
                cases[_where(label, vocabulary)] = {key: torch.as_tensor(value).cpu()
                                                    for key, value in stages.items()}
    finally:
        for hook in hooks:
            hook.remove()
    return cases


def against_upstream(upstream, detector: Detector, pictures, quiet=False):
    """Every stage of the vendor against the recorded reference. Returns (checked, problems)."""
    checked, problems = 0, []
    for where, mine in run_vendor(detector, pictures).items():
        want = upstream[where]
        found = _compare([(f"{where}: {stage}", want[stage], value) for stage, value in mine.items()])
        checked += len(mine)
        problems += found
        if not quiet:
            count = max(len(mine[f"detections@{t} scores"]) for t in THRESHOLDS)
            print(f"  {where:38} {count:4d} detections  "
                  f"{'identical' if not found else found[0]}")
    return checked, problems


def repeatable(name: str, first, second) -> list[str]:
    """Two runs of one side, byte for byte (VENDORING 2.7): without this a GPU mismatch could be
    noise rather than a difference."""
    problems = _compare([(f"{name} rerun {where}: {stage}", value, second[where][stage])
                         for where, stages in first.items() for stage, value in stages.items()])
    total = sum(len(stages) for stages in first.values())
    print(f"  {name}: run twice, {total - len(problems)}/{total} stages byte-identical")
    return problems


def against_mozo(detector: Detector, pictures, device: str) -> tuple[int, list[str]]:
    """mozo's own path -- ModelManager, adapter, PixelFlow -- against the vendor."""
    model = mozo.ModelManager().get_model("omdet_turbo", "tiny", device=device,
                                          runtime="torch-fp32")
    checked, problems = 0, []
    for label, pixels in pictures.items():
        for vocabulary in VOCABULARIES[:4]:
            boxes, scores, labels = detector(pixels, vocabulary, threshold=0.3)
            want_boxes, want_scores = as_pixelflow_reports(boxes, scores, labels)
            rows = model.predict(pixels, vocabulary, threshold=0.3).to_dict()
            got_boxes = np.array([row["bbox"] for row in rows], dtype=np.float64).reshape(-1, 4)
            got_scores = np.array([row["confidence"] for row in rows], dtype=np.float64)
            got_ids = np.array([row["class_id"] for row in rows], dtype=np.int64)
            names = [row["class_name"] for row in rows]
            where = f"mozo {label}/{vocabulary[0][:14]}"
            found = []
            if len(got_boxes) != len(want_boxes):
                found.append(f"{where}: {len(want_boxes)} detections one side, {len(got_boxes)} the other")
            else:
                if not np.array_equal(want_boxes, got_boxes):
                    found.append(f"{where}: boxes differ")
                if not np.array_equal(want_scores, got_scores):
                    found.append(f"{where}: scores differ")
                if not np.array_equal(labels.numpy(), got_ids):
                    found.append(f"{where}: class ids differ")
                if names != [vocabulary[i] for i in labels.tolist()]:
                    found.append(f"{where}: class names are not the prompts")
            checked += 4
            problems += found
    print(f"  mozo adapter: {checked} comparisons, {len(problems)} differ")
    return checked, problems


class Falsification(NamedTuple):
    """A deliberate error, and the first stage it must show up at."""

    name: str
    patch: object
    stage: str


def falsifications() -> list[Falsification]:
    def python_lower(self, text):
        import unicodedata

        import regex
        return regex.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).lower()

    def cv2_resize(image, size=640):
        import cv2
        resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
        pixels = (resized.astype(np.float32) - np.float32([123.675, 116.28, 103.53])) / np.float32(
            [58.395, 57.12, 57.375])
        return torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)))[None]

    def last_token(self, ids):
        hidden = self.hidden(ids)
        return hidden[:, -1] @ self.text_projection

    def warmer(features, classes):
        features = features / features.norm(dim=2, keepdim=True).clamp_min(1e-12)
        classes = classes / classes.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return torch.tensor(1 / 0.069).log().exp() * torch.bmm(features, classes)

    original_sincos = encoder_module.sincos_position
    original_mask = swin_module.shift_mask
    return [
        Falsification("str.lower() instead of the reference's case table",
                      mock.patch.object(Tokenizer, "clean", python_lower), "class ids"),
        Falsification("cv2 bilinear instead of PIL's", mock.patch.object(
            predictor_module, "preprocess", cv2_resize), "pixels"),
        Falsification("pool the last token instead of the first end marker",
                      mock.patch.object(text_module.TextTower, "classes", last_token),
                      "class embeddings"),
        # -100 and -99 both underflow in the softmax; -10 does not.
        Falsification("Swin shift mask -100 -> -10", mock.patch.object(
            swin_module, "shift_mask", lambda *a: original_mask(*a) * 0.1), "swin level"),
        Falsification("sine-cosine temperature 10000 -> 10001", mock.patch.object(
            encoder_module, "sincos_position",
            lambda w, h, d, t, *a: original_sincos(w, h, d, t + 1, *a)), "encoder level"),
        # Query selection ranks by the same similarity, and a scale does not change a ranking,
        # so this shows first in the logits.
        Falsification("cosine temperature 0.07 -> 0.069", mock.patch.object(
            decoder_module, "class_similarity", warmer), "logits"),
        # 0.011 would change nothing: the nearest anchor centre to the edge is 0.0125.
        Falsification("anchor validity margin 0.01 -> 0.02", mock.patch.dict(
            SPECS, {"tiny": replace(SPECS["tiny"], anchor_eps=0.02)}), "decoder layer"),
        Falsification("NMS IoU 0.5 -> 0.55", mock.patch.object(postprocess_module, "NMS", 0.55),
                      "detections"),
    ]


def falsify(checkpoint: Path, upstream, pictures, device: str) -> list[str]:
    """Each perturbation must fail, and the first failing stage must be the one named."""
    failures = []
    # A subset is enough to make each perturbation visible.
    subset = {key: pictures[key] for key in ("full", "tall")}
    for case in falsifications():
        with case.patch:
            # Rebuilt inside the patch, so construction-time code (the shift masks) is covered.
            detector = Detector(checkpoint, SPECS["tiny"], device=device)
            checked, problems = against_upstream(upstream, detector, subset, quiet=True)
        first = problems[0] if problems else ""
        caught = bool(problems) and case.stage in first.split(": ", 1)[1]
        print(f"  {'caught' if caught else 'MISSED'}  {case.name:52} "
              f"{len(problems):4d}/{checked} differ; first: {first[:90]}")
        if not caught:
            failures.append(case.name)
    return failures


def cuda_environment() -> str:
    """What a GPU result depends on, printed with it: the stack, the card, and the global switches
    (left as mozo's runtime leaves them)."""
    import subprocess

    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader",
                                 f"--id={torch.cuda.current_device()}"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        driver = "unknown"
    backends = torch.backends
    return (f"CUDA {torch.version.cuda}, cuDNN {backends.cudnn.version()}, "
            f"{torch.cuda.get_device_name()}, driver {driver}\n"
            f"TF32 matmul {backends.cuda.matmul.allow_tf32}, TF32 cuDNN {backends.cudnn.allow_tf32}, "
            f"cudnn.benchmark {backends.cudnn.benchmark}, "
            f"cudnn.deterministic {backends.cudnn.deterministic}, "
            f"deterministic algorithms {torch.are_deterministic_algorithms_enabled()}, "
            f"SDPA flash {backends.cuda.flash_sdp_enabled()} "
            f"mem-efficient {backends.cuda.mem_efficient_sdp_enabled()} "
            f"math {backends.cuda.math_sdp_enabled()}")


def torchvision_distance(pictures) -> None:
    """How far transformers 5's default image backend is from the reference's PIL one."""
    from PIL import Image
    from transformers.models.detr import DetrImageProcessor

    default = DetrImageProcessor(**PROCESSOR)
    for label, pixels in pictures.items():
        theirs = default(Image.fromarray(pixels), return_tensors="pt")["pixel_values"]
        gap = (theirs - predictor_module.preprocess(pixels)).abs()
        print(f"  {label:14} torchvision backend vs PIL: max {gap.max():.3f}, mean {gap.mean():.4f}"
              " (normalised units)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coco", type=Path, help="a COCO root holding val2017/, for more images")
    parser.add_argument("--images", type=int, default=4, help="COCO images to add (with --coco)")
    parser.add_argument("--falsify", action="store_true", help="also prove the gate can fail")
    parser.add_argument("--torchvision-distance", action="store_true",
                        help="report the default transformers image backend's distance")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                        help="where both sides run, in one process (default: cpu)")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32",
                        help="the precision both sides run in; OmDet-Turbo's torch runtime is fp32 "
                             "only, on every device")
    args = parser.parse_args()
    if args.dtype != "fp32":
        # mozo publishes no torch-fp16: autocast was slower than fp32 on CUDA (PROVENANCE). Its
        # half precision is the TensorRT fp16 engine, which is held to parity and AP, not bits.
        raise SystemExit(f"OmDet-Turbo has no {args.dtype} torch runtime to gate: mozo runs it "
                         "through torch in fp32 only, and in half precision only through "
                         "TensorRT, whose kernels are measured by parity and accuracy")

    try:
        import timm
        import transformers
    except ImportError:
        raise SystemExit("this gate compares against transformers and timm:\n"
                         "    pip install transformers timm")
    torch.set_grad_enabled(False)
    print(f"torch {torch.__version__}, transformers {transformers.__version__}, "
          f"timm {timm.__version__}, MKL_CBWR={os.environ.get('MKL_CBWR')}, {args.dtype}")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda: no CUDA device visible")
        print(cuda_environment())

    problems = []
    digest = hashlib.sha256(VOCAB_PATH.read_bytes()).hexdigest()
    if digest != VOCAB_SHA256:
        problems.append(f"vocabulary: {VOCAB_PATH.name} hashes to {digest}")
    checked = 1

    checkpoint = Path(resolve("omdet_turbo", "tiny", "torch-fp32"))
    with contextlib.redirect_stdout(io.StringIO()):
        model, processor = reference(checkpoint)
    model.to(args.device)
    detector = Detector(checkpoint, SPECS["tiny"], device=args.device)
    pictures = images(args.coco, args.images)
    upstream = run_upstream(model, processor, pictures, args.device)

    if args.device == "cuda":
        # Production settings, untouched on both sides; each must repeat itself before a
        # mismatch between them can mean anything.
        print("\ndeterminism: each side twice, same process, same settings")
        problems += repeatable("transformers", upstream,
                               run_upstream(model, processor, pictures, args.device))
        problems += repeatable("omdet_turbo_deploy", run_vendor(detector, pictures),
                               run_vendor(detector, pictures))
        checked += 2 * sum(len(stages) for stages in upstream.values())

    print(f"\ntiny: omdet_turbo_deploy against transformers, {args.device} {args.dtype}")
    upstream_checked, upstream_problems = against_upstream(upstream, detector, pictures)
    mozo_checked, mozo_problems = against_mozo(detector, pictures, args.device)
    checked += upstream_checked + mozo_checked
    problems += upstream_problems + mozo_problems

    if args.torchvision_distance:
        print("\ntransformers' default (torchvision) image backend")
        torchvision_distance(pictures)

    missed = []
    if args.falsify:
        print("\nfalsification: each must fail, first at the stage named")
        missed = falsify(checkpoint, upstream, pictures, args.device)

    print()
    for problem in problems[:20]:
        print(f"  {problem}")
    if problems or missed:
        if missed:
            print(f"falsification missed: {missed}")
        print(f"{len(problems)} of {checked} comparisons disagree. OmDet-Turbo: FAIL")
        return 1
    print(f"{checked} comparisons, every one identical to the published model. OmDet-Turbo: PASS"
          + (f"; {len(falsifications())} falsifications caught" if args.falsify else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

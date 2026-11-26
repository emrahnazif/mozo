#!/usr/bin/env python3
"""Check that mozo's HQ-SAM returns exactly what the published model does.

Three paths reach the same weights. ``SysCV/sam-hq`` itself, driven through its own
``SamPredictor``; ``mozo/vendors/sam_hq_deploy`` -- build a ``Segmenter``, hand it an image and a
prompt; and mozo end to end -- registry, weights, adapter, PixelFlow, and the ``sam_hq_boxes``
workflow node. Between upstream and the vendor sit a TinyViT with timm removed, a resize that
calls Pillow instead of torchvision, a decoder whose selection moved out of the module, and a
predictor rewritten around a cache -- and any of those could quietly change a number.

The comparison is exact (``torch.equal``), at every stage: the resized image, the image
embedding, the early feature the HQ decoder fuses, the decoder's raw tokens, and the final
logits, IoU and masks for each prompt.

**On CUDA (the primary gate) and on the CPU.** Both sides run in one process, on one device, in
fp32 -- mozo's only torch precision for HQ-SAM (``torch-fp32`` is the only runtime it publishes,
and nothing derives a half-precision one). On CUDA the global switches stay as mozo's runtime
leaves them, which is torch's defaults (TF32 off for matmuls, on for cuDNN convolutions,
``cudnn.benchmark`` off); each side first records everything twice and must repeat itself byte
for byte; the GPU and the stack are printed with the result.

**One thing mozo returns that upstream does not, and how it is checked.** With
``multimask_output=True`` upstream's ``MaskDecoderHQ.forward`` keeps only the SAM candidate with
the highest predicted IoU and adds the HQ token's mask to it -- one mask. mozo's promptable
contract returns three candidates, so the vendor adds the HQ mask to *each* of the three. The
best of mozo's three is compared against upstream's one answer; all three are compared against
upstream's own raw decoder output (``predict_masks``) put through the same addition and upstream's
``postprocess_masks``. Both must be bit-identical.

Run from the repository root, in an environment with torch, torchvision, Pillow and timm
(upstream's TinyViT imports it)::

    git clone https://github.com/SysCV/sam-hq ~/src/sam-hq
    git -C ~/src/sam-hq checkout <commit in PROVENANCE.md>
    python tools/verify/sam_hq.py --upstream ~/src/sam-hq --device cuda       # the primary gate
    python tools/verify/sam_hq.py --upstream ~/src/sam-hq                     # CPU, every variant
    python tools/verify/sam_hq.py --upstream ~/src/sam-hq tiny base
    python tools/verify/sam_hq.py --upstream ~/src/sam-hq --device cuda --falsify \\
        --reference-cache /tmp/sam_hq-ref                     # each perturbation must FAIL

``--falsify`` runs the gate once per deliberate perturbation of the vendored side and passes only
if every one of them is caught -- a gate that has never failed has not been shown to work. The
reference does not change between perturbations, so it is recorded once per variant (as digests,
under ``--reference-cache`` when given, and reused by later runs on the same stack).
``--swap-models`` keeps one network on the GPU at a time: upstream is recorded and released before
the vendor is built, and the vendor before the adapter.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from mozo.image import load_image  # noqa: E402
from mozo.vendors.sam_hq_deploy import Segmenter  # noqa: E402
from mozo.vendors.sam_hq_deploy import image as vendor_image  # noqa: E402

sys.path.insert(0, str(ROOT / "tests"))
from conftest import FIXTURE  # noqa: E402

FAMILY = "sam_hq"
PROVENANCE = ROOT / "mozo" / "vendors" / "sam_hq_deploy" / "PROVENANCE.md"

#: mozo variant -> upstream's ``sam_model_registry`` key.
UPSTREAM_NAMES = {"tiny": "vit_tiny", "base": "vit_b", "large": "vit_l", "huge": "vit_h"}

#: Prompts, as fractions of the image so they follow it rather than its size. Each exercises
#: something the others do not: SAM's box goes to the prompt encoder as a box (not folded into the
#: points, as SAM 2 does), clicks alone get a padding token and clicks with a box do not, and
#: ``multimask_output=False`` reads token 0 instead of choosing among 1..3.
PROMPTS: tuple[tuple[str, list | None, list | None, list | None, bool], ...] = (
    ("one positive", [[0.62, 0.55]], [1], None, True),
    ("positive and negative", [[0.62, 0.55], [0.58, 0.72]], [1, 0], None, True),
    ("box", None, None, [0.55, 0.42, 0.72, 0.78], True),
    ("box, single mask", None, None, [0.55, 0.42, 0.72, 0.78], False),
    ("box and negative", [[0.58, 0.72]], [0], [0.55, 0.42, 0.72, 0.78], True),
    ("lone negative", [[0.62, 0.55]], [0], None, True),
    ("one positive, single mask", [[0.62, 0.55]], [1], None, False),
)

#: The detector rows handed to the ``sam_hq_boxes`` node: boxes as fractions of the image, a
#: score and a class each. Out of class order, so a node that renumbered rows would show.
NODE_ROWS = (
    ([0.55, 0.42, 0.72, 0.78], 0.91, 2),
    ([0.10, 0.10, 0.50, 0.60], 0.64, 0),
    ([0.30, 0.20, 0.95, 0.90], 0.37, 1),
)

#: Deliberate perturbations of the vendored side, each of which must make the gate fail. Each
#: names the stage it should first fail at, so a falsifier caught somewhere unexpected is visible.
#: The second field is which trunk a perturbation applies to (``"vit"``, ``"tiny"`` or both); the
#: third, ``"cuda"`` for those only a GPU run can catch -- a global switch, a memory layout or a
#: host/device split that changes kernels on CUDA and changes nothing on a CPU.
FALSIFIERS = {
    "resize-filter": ("PIL BILINEAR -> BICUBIC in the resize (fails at: resized image)", None, None),
    "resize-rounding": ("round-half-even instead of upstream's int(x + 0.5) "
                        "(fails at: resized image, half-pixel size)", None, None),
    "layernorm-eps": ("ViT LayerNorm eps 1e-6 -> 1e-5 (fails at: image embedding)", "vit", None),
    "attention-bias": ("TinyViT attention biases left at zero, upstream's stale-buffer case "
                       "(fails at: image embedding)", "tiny", None),
    "interm-index": ("fuse the last global block's feature instead of the first "
                     "(fails at: hq features)", "vit", None),
    "hq-token-only": ("HQ token alone instead of SAM + HQ (fails at: final logits)", None, None),
    "coords-float32": ("scale prompts in float32 instead of float64 (fails at: prompt embedding)",
                       None, None),
    "no-crop": ("skip cropping the letterbox padding before the final resize "
                "(fails at: final masks)", None, None),
    "all-trainable": ("every parameter requires grad, not only HQ-SAM's additions: torch.matmul "
                      "stops folding (fails at: batched boxes on cuda; the requires-grad stage on "
                      "cpu)", None, None),
    "tf32-matmul": ("TF32 matmuls on for the vendor only; mozo leaves torch's default, off "
                    "(fails at: image embedding)", None, "cuda"),
    "cudnn-tf32-off": ("TF32 cuDNN convolutions off for the vendor only; mozo leaves torch's "
                       "default, on (fails at: image embedding)", None, "cuda"),
    "channels-last": ("encoder input in channels-last memory, same values: cuDNN picks other "
                      "kernels (fails at: image embedding)", None, "cuda"),
    # Replaces "mask resize on the host": CUDA's and the CPU's bilinear F.interpolate agreed bit
    # for bit on every mask here (0 of 396 stages moved, all four variants), so a host resize is
    # uncatchable because it is not wrong. Decoding off the device upstream decodes on is.
    "decode-host": ("prompt encoder and decoder on the host, the encoder on the GPU "
                    "(fails at: final logits)", None, "cuda"),
}


def prompt_for(spec, shape: tuple[int, int]) -> dict:
    """Turn one :data:`PROMPTS` entry into pixel coordinates for ``shape``."""
    _, points, labels, box, multimask = spec
    height, width = shape
    scale = np.array([width, height], dtype=np.float32)
    return {
        "points": None if points is None else np.asarray(points, dtype=np.float32) * scale,
        "labels": None if labels is None else np.asarray(labels),
        "boxes": None if box is None
        else (np.asarray(box, dtype=np.float32).reshape(2, 2) * scale).reshape(4),
        "multimask_output": multimask,
    }


def batch_boxes(shape: tuple[int, int]) -> np.ndarray:
    """Two boxes for the batched path."""
    h, w = shape
    return np.array([[0.55 * w, 0.42 * h, 0.72 * w, 0.78 * h],
                     [0.10 * w, 0.10 * h, 0.50 * w, 0.60 * h]], dtype=np.float32)


def node_boxes(shape: tuple[int, int]) -> np.ndarray:
    """:data:`NODE_ROWS`' boxes in whole pixels, as a detector's rows would carry them."""
    h, w = shape
    scale = np.array([w, h, w, h], dtype=np.float32)
    return np.floor(np.array([box for box, _, _ in NODE_ROWS], dtype=np.float32) * scale)


def images() -> dict[str, np.ndarray]:
    """The fixture photograph, and the shapes that reach code a landscape photo does not."""
    photo = load_image(str(FIXTURE))
    return {
        "landscape 1920x1281": photo,
        # Portrait: the padding is on the right rather than the bottom.
        "portrait 1281x1920": np.ascontiguousarray(photo.transpose(1, 0, 2)),
        # Odd sizes, where the long-side rounding and the crop both bite; and an upscale.
        "odd 333x517": np.ascontiguousarray(photo[:517, :333]),
        "small upscale 200x150": np.ascontiguousarray(photo[400:550, 900:1100]),
        # 1025 * (1024 / 2048) = 512.5 exactly: upstream's int(x + 0.5) gives 513 where Python's
        # round-half-even would give 512. The only image here where the two rules disagree.
        "half-pixel 2048x1025": np.ascontiguousarray(
            np.concatenate([photo[:1025], photo[:1025, :128]], axis=1)),
    }


def pinned_commit() -> str:
    """The upstream commit ``PROVENANCE.md`` records this package as extracted from."""
    match = re.search(r"Upstream commit\s*\|\s*`([0-9a-f]{40})`", PROVENANCE.read_text())
    if match is None:
        raise SystemExit(f"no upstream commit recorded in {PROVENANCE}")
    return match.group(1)


def published(variant: str) -> Path:
    """Locate a variant's checkpoint the way mozo itself would."""
    from mozo.weights import resolve

    return Path(resolve(FAMILY, variant, "torch-fp32"))


def reference(upstream: Path, variant: str, checkpoint: Path, device: str):
    """Build upstream's own predictor, the way its demos do, on *device*.

    ``eval()`` is called again after the builder, as ``demo/demo_hqsam_light.py`` does: the
    builder calls it before loading, and TinyViT caches its attention biases at that moment, so
    a model used straight from ``build_sam_vit_t`` attends with zeros. See ``tiny_vit.py``.
    """
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    from segment_anything import SamPredictor, sam_model_registry

    with contextlib.redirect_stdout(None):  # the builder prints its load report
        sam = sam_model_registry[UPSTREAM_NAMES[variant]](checkpoint=str(checkpoint))
    return SamPredictor(sam.to(device).eval())


def release() -> None:
    """Hand a dropped network's memory back, so the next one fits on a 12 GB card."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --- recording -----------------------------------------------------------------------------
#
# Each side records every stage into an ordered ``{"<image> | <stage>": tensor}`` on the host.
# The two records are then compared key by key, in order, so the first disagreement is the first
# stage that moved. Recording the sides apart is what lets a side be run twice (determinism), the
# reference be cached across falsifiers, and one network at a time sit on the GPU.

def host(value) -> torch.Tensor:
    """A stage's value as a host tensor, whatever it came as."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return torch.as_tensor(np.asarray(value))


def canonical(tensor: torch.Tensor) -> torch.Tensor:
    """What is compared: float32, contiguous. Lossless for the uint8, bool and fp32 stages here."""
    return tensor.float().contiguous()


def digest(tensor: torch.Tensor) -> tuple:
    """A host tensor's shape and the sha256 of its canonical bytes: what the cache keeps."""
    tensor = canonical(tensor)
    return ("digest", tuple(tensor.shape), hashlib.sha256(tensor.numpy().tobytes()).hexdigest())


def trainable(model: torch.nn.Module) -> torch.Tensor:
    """Which parameters require grad, as the sha256 of their sorted names.

    Recorded because it changes arithmetic: ``torch.matmul`` folds a batched, strided product
    into one matrix multiply or runs it batched depending on whether the weight requires grad,
    and on CUDA those are different sums (see ``network.TRAINABLE``).
    """
    names = sorted(name for name, p in model.named_parameters() if p.requires_grad)
    return torch.tensor(list(hashlib.sha256("\n".join(names).encode()).digest()), dtype=torch.uint8)


def node_key(label: str, row: int) -> str:
    return f"{label} | sam_hq_boxes node: row {row} mask"


def record_reference(theirs, pictures: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """Every stage upstream produces, through its own predictor, its own modules and order."""
    out: dict[str, torch.Tensor] = {}
    device = theirs.device
    for label, image in pictures.items():
        shape = image.shape[:2]

        def put(stage, value):
            out[f"{label} | {stage}"] = host(value)

        # --- preprocessing and the encoder ---
        put("resized image",
            torch.as_tensor(theirs.transform.apply_image(image)).permute(2, 0, 1).contiguous()[None])
        theirs.set_image(image)
        put("image embedding", theirs.features)
        put("hq features (early feature fused)", upstream_hq_features(theirs))

        for spec in PROMPTS:
            name = spec[0]
            prompt = prompt_for(spec, shape)
            # --- the prompt encoder, fed as upstream's predict feeds it ---
            coords = labels = box = None
            if prompt["points"] is not None:
                coords = torch.as_tensor(theirs.transform.apply_coords(prompt["points"], shape),
                                         dtype=torch.float, device=device)[None]
                labels = torch.as_tensor(prompt["labels"], dtype=torch.int, device=device)[None]
            if prompt["boxes"] is not None:
                box = torch.as_tensor(theirs.transform.apply_boxes(prompt["boxes"], shape),
                                      dtype=torch.float, device=device)[None]
            sparse, dense = theirs.model.prompt_encoder(
                points=None if coords is None else (coords, labels), boxes=box, masks=None)
            put(f"{name}: sparse prompt embedding", sparse)
            put(f"{name}: dense prompt embedding", dense)

            # --- the decoder, before selection ---
            raw_masks, raw_iou = raw_decoder(theirs, sparse, dense)
            put(f"{name}: SAM tokens (raw)", raw_masks[:, :4])
            put(f"{name}: hq token (raw)", raw_masks[:, 4:])
            put(f"{name}: iou (raw)", raw_iou)

            # --- upstream's one answer ---
            mask, score, low = theirs.predict(
                point_coords=prompt["points"], point_labels=prompt["labels"], box=prompt["boxes"],
                multimask_output=prompt["multimask_output"])
            put(f"{name}: logits (upstream's answer)", low[None])
            put(f"{name}: iou (upstream's answer)", score[None])
            put(f"{name}: mask (upstream's answer)", mask[None])

            # --- all three candidates, from upstream's raw output and its own resize ---
            if prompt["multimask_output"]:
                summed = raw_masks[:, 1:4] + raw_masks[:, 4:5]
                full = theirs.model.postprocess_masks(summed, theirs.input_size, theirs.original_size)
                put(f"{name}: three candidates, logits", summed)
                put(f"{name}: three candidates, masks", full > theirs.model.mask_threshold)
                put(f"{name}: three candidates, iou", raw_iou[:, 1:4])

        # --- refinement: feed the best low-res logits back in, as upstream documents ---
        prompt = prompt_for(PROMPTS[0], shape)
        _, _, low = theirs.predict(point_coords=prompt["points"], point_labels=prompt["labels"],
                                   multimask_output=True)
        mask, _, low = theirs.predict(point_coords=prompt["points"], point_labels=prompt["labels"],
                                      mask_input=low, multimask_output=False)
        put("refinement with mask_input: logits", low[None])
        put("refinement with mask_input: mask", mask[None])

        # --- a batch of boxes, through upstream's predict_torch ---
        boxes = torch.as_tensor(theirs.transform.apply_boxes(batch_boxes(shape), shape),
                                dtype=torch.float, device=device)
        masks_b, iou_b, low_b = theirs.predict_torch(None, None, boxes=boxes, multimask_output=False)
        put("batch of two boxes: logits", low_b)
        put("batch of two boxes: masks", masks_b)
        put("batch of two boxes: iou", iou_b)

        # --- what the sam_hq_boxes node must put on each row: one mask per box ---
        boxes = torch.as_tensor(theirs.transform.apply_boxes(node_boxes(shape), shape),
                                dtype=torch.float, device=device)
        masks_n, _, _ = theirs.predict_torch(None, None, boxes=boxes, multimask_output=False)
        for row in range(len(NODE_ROWS)):
            out[node_key(label, row)] = host(masks_n[row, 0])
    out["model | parameters that require grad"] = trainable(theirs.model)
    return out


def record_vendor(ours: Segmenter, pictures: dict[str, np.ndarray]):
    """The same stages from the vendor, and the vendor's checks against itself.

    Returns ``(record, internal)``: *internal* is a list of ``(stage, a, b)`` that must be equal
    and involve no reference -- the encoder cache answering a repeated prompt as the first time.
    """
    out: dict[str, torch.Tensor] = {}
    internal = []
    for label, image in pictures.items():
        shape = image.shape[:2]

        def put(stage, value):
            out[f"{label} | {stage}"] = host(value)

        put("resized image", vendor_image.resize_longest_side(image, ours.image_size))
        features = ours.encode(image)
        put("image embedding", features["image_embed"])
        put("hq features (early feature fused)", features["hq_features"])

        for spec in PROMPTS:
            name = spec[0]
            prompt = prompt_for(spec, shape)
            clicks, corners = ours._prompt(prompt["points"], prompt["labels"], prompt["boxes"], shape)
            sparse, dense = ours.network.prompt_encoder(points=clicks, boxes=corners, masks=None)
            put(f"{name}: sparse prompt embedding", sparse)
            put(f"{name}: dense prompt embedding", dense)

            masks_sam, masks_hq, iou = ours.network.mask_decoder(
                image_embeddings=features["image_embed"],
                image_pe=ours.network.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                hq_features=features["hq_features"])
            put(f"{name}: SAM tokens (raw)", masks_sam)
            put(f"{name}: hq token (raw)", masks_hq)
            put(f"{name}: iou (raw)", iou)

            got = ours.predict(image, **prompt)
            best = int(np.argmax(got.scores[0]))
            put(f"{name}: logits (upstream's answer)", got.logits[:, best:best + 1])
            put(f"{name}: iou (upstream's answer)", got.scores[:, best:best + 1])
            put(f"{name}: mask (upstream's answer)", got.masks[:, best:best + 1])
            if prompt["multimask_output"]:
                put(f"{name}: three candidates, logits", got.logits)
                put(f"{name}: three candidates, masks", got.masks)
                put(f"{name}: three candidates, iou", got.scores)

        prompt = prompt_for(PROMPTS[0], shape)
        first = ours.predict(image, **prompt)
        best = int(np.argmax(first.scores[0]))
        got = ours.predict(image, points=prompt["points"], labels=prompt["labels"],
                           mask_input=first.logits[:, best:best + 1], multimask_output=False)
        put("refinement with mask_input: logits", got.logits)
        put("refinement with mask_input: mask", got.masks)

        got = ours.predict(image, boxes=batch_boxes(shape), multimask_output=False)
        put("batch of two boxes: logits", got.logits)
        put("batch of two boxes: masks", got.masks)
        put("batch of two boxes: iou", got.scores)

        # A held image must answer a repeated prompt identically.
        again = ours.predict(image, **prompt)
        internal.append((f"{label} | encoder cache: repeated prompt", again.logits, first.logits))
    out["model | parameters that require grad"] = trainable(ours.network)
    return out, internal


def raw_decoder(theirs, sparse, dense):
    """Upstream's decoder before its selection: ``(B, 5, 256, 256)`` logits, ``(B, 4)`` IoU.

    Calls upstream's own modules in upstream's own order -- the two lines at the top of
    ``MaskDecoderHQ.forward``, then ``predict_masks`` -- and stops before the slicing.
    """
    decoder = theirs.model.mask_decoder
    return decoder.predict_masks(
        image_embeddings=theirs.features,
        image_pe=theirs.model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        hq_features=upstream_hq_features(theirs),
    )


def upstream_hq_features(theirs):
    """The first two lines of upstream's ``MaskDecoderHQ.forward``, on its predictor's features."""
    decoder = theirs.model.mask_decoder
    vit_features = theirs.interm_features[0].permute(0, 3, 1, 2)
    return decoder.embedding_encoder(theirs.features) + decoder.compress_vit_feat(vit_features)


# --- comparing -----------------------------------------------------------------------------

class Comparison:
    """Accumulates stage comparisons and reports how many disagreed."""

    def __init__(self, quiet: bool = False) -> None:
        self.failed: list[str] = []
        self.checked = 0
        self.quiet = quiet

    def __call__(self, name: str, ours, theirs) -> None:
        ours = canonical(host(ours))
        self.checked += 1
        if isinstance(theirs, tuple):  # a cached digest
            mine = digest(ours)
            if mine == theirs:
                self._say(f"  ok    {name:<72} identical (digest)")
            else:
                self._say(f"  FAIL  {name:<72} differs from the cached reference "
                          f"(shape {mine[1]} vs {theirs[1]})")
                self.failed.append(name)
            return
        theirs = canonical(host(theirs))
        if ours.shape != theirs.shape:
            self._say(f"  FAIL  {name:<72} shape {tuple(ours.shape)} != {tuple(theirs.shape)}")
            self.failed.append(name)
        elif torch.equal(ours, theirs):
            self._say(f"  ok    {name:<72} identical")
        else:
            self._say(f"  FAIL  {name:<72} max|d| {(ours - theirs).abs().max().item():.3e}")
            self.failed.append(name)

    def _say(self, line: str) -> None:
        if not self.quiet:
            print(line)


def compare_records(got: dict, want: dict, same: Comparison, skip_node: bool = True) -> None:
    """Every stage of *got* against *want*, in recording order; a stage on one side only fails."""
    for name, value in got.items():
        if name not in want:
            same._say(f"  FAIL  {name:<72} not in the reference record")
            same.checked += 1
            same.failed.append(name)
            continue
        same(name, value, want[name])
    for name in want:
        if name not in got and not (skip_node and "sam_hq_boxes node" in name):
            same._say(f"  FAIL  {name:<72} not recorded by this side")
            same.checked += 1
            same.failed.append(name)


def repeatable(side: str, first: dict, second: dict, same: Comparison) -> None:
    """A side's two records of the same inputs must be byte-identical."""
    before = len(same.failed)
    for name in first:
        same(f"determinism ({side}) {name}", second[name], first[name])
    print(f"  {side}: {len(first)} stages recorded again, "
          f"{'byte-identical' if len(same.failed) == before else 'NOT repeatable'}")


# --- the reference, recorded once ----------------------------------------------------------

def stack(device: str) -> dict:
    """What a cached reference holds for: the stack, the card and the global switches."""
    info = {"torch": torch.__version__, "device": device, "upstream": pinned_commit()}
    if device == "cuda":
        info.update(cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                    gpu=torch.cuda.get_device_name(),
                    tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                    tf32_cudnn=torch.backends.cudnn.allow_tf32,
                    benchmark=torch.backends.cudnn.benchmark)
    return info


def cached_reference(cache: Path | None, variant: str, device: str, pictures) -> dict | None:
    """A variant's reference digests from *cache*, if they were recorded on this exact stack."""
    if cache is None:
        return None
    path = cache / f"{FAMILY}-{variant}-{device}-fp32.pt"
    if not path.is_file():
        return None
    saved = torch.load(path, weights_only=False)
    meta = dict(stack(device), images={k: v.shape for k, v in pictures.items()})
    if saved.get("meta") != meta:
        print(f"  {path.name}: recorded on another stack, recording again")
        return None
    return saved["digests"]


def save_reference(cache: Path | None, variant: str, device: str, pictures, record: dict) -> None:
    if cache is None:
        return
    cache.mkdir(parents=True, exist_ok=True)
    meta = dict(stack(device), images={k: v.shape for k, v in pictures.items()})
    torch.save({"meta": meta, "digests": {k: digest(v) for k, v in record.items()}},
               cache / f"{FAMILY}-{variant}-{device}-fp32.pt")


def reference_run(upstream: Path, variant: str, checkpoint: Path, device: str, pictures,
                  repeat: int, same: Comparison | None):
    """Build upstream, record it *repeat* times (each must match the first), return the record and
    the live predictor (``None`` once released)."""
    theirs = reference(upstream, variant, checkpoint, device)
    first = record_reference(theirs, pictures)
    for _ in range(repeat - 1):
        repeatable("upstream", first, record_reference(theirs, pictures), same)
    return first, theirs


# --- the gate ------------------------------------------------------------------------------

def check_variant(upstream: Path, variant: str, checkpoint: Path, same: Comparison,
                  end_to_end: bool, device: str, repeat: int, swap: bool,
                  cache: Path | None) -> None:
    """Every stage of one variant, on every image and prompt."""
    pictures = images()
    print(f"\n== {variant}  ({checkpoint.name}, upstream {UPSTREAM_NAMES[variant]}, {device} fp32)")
    if repeat > 1:
        print(f"determinism: each side records everything {repeat} times, same process, same settings")
    want, theirs = reference_run(upstream, variant, checkpoint, device, pictures, repeat, same)
    save_reference(cache, variant, device, pictures, want)
    if swap:
        del theirs
        release()

    ours = Segmenter(checkpoint, device=device)
    got, internal = record_vendor(ours, pictures)
    for _ in range(repeat - 1):
        again, _ = record_vendor(ours, pictures)
        repeatable("sam_hq_deploy", got, again, same)
        del again

    print(f"\nsam_hq_deploy against SysCV/sam-hq, {device} fp32:")
    compare_records(got, want, same)
    for name, a, b in internal:
        same(name, a, b)

    if end_to_end:
        landscape = next(iter(pictures))
        prompt = prompt_for(PROMPTS[0], pictures[landscape].shape[:2])
        direct = ours.predict(pictures[landscape], **prompt)
        if swap:
            del ours
            release()
        mozo_path(variant, device, pictures, direct, want, same)


def mozo_path(variant: str, device: str, pictures, direct, want: dict, same: Comparison) -> None:
    """Registry -> weights -> adapter -> PixelFlow, and the ``sam_hq_boxes`` node.

    The adapter is checked against the vendor's answer through the same door. PixelFlow rounds
    confidences to its own precision, so the vendor's raw output is handed to the adapter's own
    ``candidates`` (ranking, then ``pf.detections.from_arrays``) rather than restating that
    rounding here; what is checked is that the adapter adds nothing else.

    The node is checked against upstream directly: each row's mask against upstream's
    ``predict_torch`` on the same boxes, every other field of the row against what went in.
    """
    import pixelflow as pf

    from mozo.adapters._promptable import candidates
    from mozo.manager import ModelManager
    from mozo.workflow.node import Models
    from mozo.workflow.nodes.model import sam_hq_boxes

    print("\nmozo end to end:")
    manager = ModelManager()
    landscape = next(iter(pictures))
    image = pictures[landscape]
    prompt = prompt_for(PROMPTS[0], image.shape[:2])
    model = manager.get_model(FAMILY, variant, device=device)
    found = model.predict(image, points=prompt["points"], labels=prompt["labels"])
    expected = candidates(direct.masks, direct.scores, None, device)

    def rows(detections):
        return (np.array([row.confidence for row in detections], np.float32),
                np.stack([np.asarray(row.masks[0] if isinstance(row.masks, list) else row.masks)
                          for row in detections]),
                np.array([row.bbox for row in detections], np.float32))

    (got_scores, got_masks, got_boxes), (want_scores, want_masks, want_boxes) = rows(found), rows(expected)
    same("adapter: scores, ranked", got_scores, want_scores)
    same("adapter: masks, ranked", got_masks, want_masks)
    same("adapter: boxes, ranked", got_boxes, want_boxes)
    # And the ranking really is the vendor's: best first, so row 0 is upstream's single answer.
    order = np.argsort(-direct.scores[0], kind="stable")
    same("adapter: row masks are the vendor's, best first", got_masks, direct.masks[0][order])

    print("\nsam_hq_boxes node, against upstream:")
    models = Models(lambda family, name, **kw: manager.get_model(family, name, device=device, **kw))
    for label, image in pictures.items():
        boxes = node_boxes(image.shape[:2])
        given = pf.detections.from_arrays(
            boxes=boxes, scores=np.array([s for _, s, _ in NODE_ROWS], np.float32),
            class_ids=np.array([c for _, _, c in NODE_ROWS], np.int64))
        out = sam_hq_boxes(image, given, models, variant=variant)
        same(f"{label} | sam_hq_boxes node: row count", len(out), len(given))
        for row, (before, after) in enumerate(zip(given, out)):
            same(node_key(label, row), np.asarray(after.masks[0]), want[node_key(label, row)])
            same(f"{label} | sam_hq_boxes node: row {row} box, score, class kept",
                 np.array([*after.bbox, after.confidence, after.class_id], np.float64),
                 np.array([*before.bbox, before.confidence, before.class_id], np.float64))
    manager.unload(FAMILY, variant, device=device)
    del model
    release()


# --- falsification -------------------------------------------------------------------------

@contextlib.contextmanager
def perturbed(name: str):
    """Apply one :data:`FALSIFIERS` perturbation to the vendored side, and undo it after."""
    from PIL import Image

    from mozo.vendors.sam_hq_deploy import image as image_module
    from mozo.vendors.sam_hq_deploy import image_encoder, network, predictor, tiny_vit

    undo = []

    def patch(owner, attribute, value):
        undo.append((owner, attribute, getattr(owner, attribute)))
        setattr(owner, attribute, value)

    if name == "resize-filter":
        def bicubic(image, long_side):
            h, w = image_module.preprocess_shape(image.shape[0], image.shape[1], long_side)
            out = np.array(Image.fromarray(image).resize((w, h), Image.BICUBIC))
            return torch.as_tensor(out).permute(2, 0, 1).contiguous()[None]

        patch(image_module, "resize_longest_side", bicubic)
        patch(predictor, "resize_longest_side", bicubic)
    elif name == "resize-rounding":
        def rounded(height, width, long_side):
            scale = long_side * 1.0 / max(height, width)
            return (round(height * scale), round(width * scale))
        patch(image_module, "preprocess_shape", rounded)
    elif name == "layernorm-eps":
        original_factory = network._image_encoder

        def factory(spec):
            encoder, dim = original_factory(spec)
            for module in encoder.modules():
                if isinstance(module, torch.nn.LayerNorm):
                    module.eps = 1e-5
            return encoder, dim
        patch(network, "_image_encoder", factory)
    elif name == "attention-bias":
        def forward(self, x):
            B, N, _ = x.shape
            x = self.norm(x)
            q, k, v = self.qkv(x).view(B, N, self.num_heads, -1).split(
                [self.key_dim, self.key_dim, self.d], dim=3)
            q, k, v = (t.permute(0, 2, 1, 3) for t in (q, k, v))
            attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
            return self.proj((attn @ v).transpose(1, 2).reshape(B, N, self.dh))
        patch(tiny_vit.Attention, "forward", forward)
    elif name == "interm-index":
        original_forward = image_encoder.ImageEncoderViT.forward

        def last_first(self, x):
            features, interm = original_forward(self, x)
            return features, interm[::-1]
        patch(image_encoder.ImageEncoderViT, "forward", last_first)
    elif name == "hq-token-only":
        def decode(self, features, points, boxes, mask_input, multimask_output):
            sparse, dense = self.prompt_encoder(points=points, boxes=boxes, masks=mask_input)
            masks_sam, masks_hq, iou = self.mask_decoder(
                features["image_embed"], self.prompt_encoder.get_dense_pe(), sparse, dense,
                features["hq_features"])
            chosen = slice(1, 4) if multimask_output else slice(0, 1)
            return masks_hq.expand(-1, chosen.stop - chosen.start, -1, -1), iou[:, chosen]
        patch(network.SamHq, "decode", decode)
    elif name == "coords-float32":
        def float32_coords(coords, shape, long_side):
            old_h, old_w = shape
            new_h, new_w = image_module.preprocess_shape(old_h, old_w, long_side)
            scaled = torch.tensor(np.asarray(coords, dtype=np.float32))
            scaled[..., 0] *= np.float32(new_w / old_w)
            scaled[..., 1] *= np.float32(new_h / old_h)
            return scaled
        patch(predictor, "to_model_coords", float32_coords)
    elif name == "no-crop":
        def no_crop(masks, input_size, shape, long_side):
            return torch.nn.functional.interpolate(masks, shape, mode="bilinear", align_corners=False)
        patch(predictor, "to_original", no_crop)
    elif name == "all-trainable":
        patch(network, "TRAINABLE", ("",))
    elif name == "tf32-matmul":
        patch(torch.backends.cuda.matmul, "allow_tf32", True)
    elif name == "cudnn-tf32-off":
        patch(torch.backends.cudnn, "allow_tf32", False)
    elif name == "channels-last":
        original_encoder_forward = {cls: cls.forward for cls in (image_encoder.ImageEncoderViT,
                                                                 tiny_vit.TinyViT)}

        def channels_last(cls):
            def forward(self, x):
                return original_encoder_forward[cls](
                    self, x.contiguous(memory_format=torch.channels_last))
            return forward
        for cls in original_encoder_forward:
            patch(cls, "forward", channels_last(cls))
    elif name == "decode-host":
        import copy

        original_decode = network.SamHq.decode

        def on_host(self, features, points, boxes, mask_input, multimask_output):
            if not hasattr(self, "_host"):
                self._host = copy.deepcopy(self).cpu()
            device = features["image_embed"].device
            move = (lambda t: None if t is None else t.cpu())
            low, iou = original_decode(
                self._host, {k: v.cpu() if torch.is_tensor(v) else v for k, v in features.items()},
                None if points is None else (points[0].cpu(), points[1].cpu()), move(boxes),
                move(mask_input), multimask_output)
            return low.to(device), iou.to(device)
        patch(network.SamHq, "decode", on_host)
    else:
        raise SystemExit(f"unknown falsifier {name!r}; choose from {sorted(FALSIFIERS)}")
    try:
        yield
    finally:
        for owner, attribute, value in reversed(undo):
            setattr(owner, attribute, value)


def falsify(upstream: Path, variants: list[str], checkpoints: dict[str, Path], device: str,
            cache: Path | None) -> int:
    """Run the gate under each perturbation; every applicable one must fail."""
    caught, missed = [], []
    pictures = images()
    for variant in variants:
        want = cached_reference(cache, variant, device, pictures)
        if want is None:
            record, theirs = reference_run(upstream, variant, checkpoints[variant], device,
                                           pictures, 1, None)
            save_reference(cache, variant, device, pictures, record)
            want = {k: digest(v) for k, v in record.items()}
            del record, theirs
            release()
        else:
            print(f"  {variant}: reference digests from {cache}")
        for name, (what, trunk, only) in FALSIFIERS.items():
            if trunk is not None and trunk != ("tiny" if variant == "tiny" else "vit"):
                continue
            if only is not None and only != device:
                continue
            same = Comparison(quiet=True)
            with perturbed(name):
                ours = Segmenter(checkpoints[variant], device=device)
                got, _ = record_vendor(ours, pictures)
            del ours
            release()
            compare_records(got, want, same)
            verdict = (caught if same.failed else missed)
            verdict.append((name, variant))
            first = same.failed[0] if same.failed else "-"
            print(f"  {'caught' if same.failed else 'MISSED':<7} {name:<17} {variant:<6} "
                  f"{len(same.failed):>3}/{same.checked} stages fail; first: {first}")
            print(f"          {what}")
    print()
    if missed:
        print(f"{len(missed)} perturbation(s) passed the gate: {missed}")
        return 1
    print(f"all {len(caught)} perturbations caught ({device}).\nHQ-SAM falsification: PASS")
    return 0


def cuda_environment() -> str:
    """What a GPU result depends on, printed with it: the stack, the card, the global switches."""
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader",
                                 f"--id={torch.cuda.current_device()}"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        driver = "unknown"
    backends = torch.backends
    return (f"CUDA {torch.version.cuda}, cuDNN {backends.cudnn.version()}, "
            f"{torch.cuda.get_device_name()}, driver {driver}\n"
            f"           TF32 matmul {backends.cuda.matmul.allow_tf32}, "
            f"TF32 cuDNN {backends.cudnn.allow_tf32}, cudnn.benchmark {backends.cudnn.benchmark}, "
            f"cudnn.deterministic {backends.cudnn.deterministic}, "
            f"deterministic algorithms {torch.are_deterministic_algorithms_enabled()}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("variants", nargs="*", help="variants to check (default: all published)")
    parser.add_argument("--upstream", type=Path, required=True, help="checkout of SysCV/sam-hq")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="a checkpoint to check instead of the published one (one variant)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                        help="where both sides run, in one process (default: cpu)")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32",
                        help="the precision both sides run in; HQ-SAM's torch runtime is fp32 only")
    parser.add_argument("--repeat", type=int, default=None,
                        help="records per side, all byte-identical; default 2 on cuda, 1 on cpu")
    parser.add_argument("--swap-models", action="store_true",
                        help="keep one network on the device at a time")
    parser.add_argument("--reference-cache", type=Path, default=None,
                        help="directory for the reference's digests, reused by --falsify")
    parser.add_argument("--falsify", action="store_true",
                        help="run each deliberate perturbation; every one must fail")
    parser.add_argument("--allow-commit-drift", action="store_true",
                        help="run against an upstream other than the extracted commit")
    arguments = parser.parse_args()
    if arguments.dtype != "fp32":
        raise SystemExit(f"HQ-SAM has no {arguments.dtype} torch runtime to gate: mozo publishes "
                         "torch-fp32 only and derives no half-precision torch runtime from it. "
                         "TensorRT fp16 was measured (PROVENANCE) but is not published, and "
                         "TensorRT is held to parity and accuracy, not bits.")
    device = arguments.device
    repeat = arguments.repeat or (2 if device == "cuda" else 1)

    from mozo.registry import get_model_info

    published_variants = get_model_info(FAMILY)["variants"]
    variants = arguments.variants or published_variants
    for variant in variants:
        if variant not in published_variants:
            raise SystemExit(f"{variant!r} is not published; choose from {published_variants}")
    if arguments.checkpoint is not None and len(variants) != 1:
        raise SystemExit("--checkpoint checks one variant; name it")

    upstream = arguments.upstream.resolve()
    if not (upstream / "segment_anything" / "build_sam.py").is_file():
        raise SystemExit(f"no sam-hq checkout at {upstream}. Clone it:\n"
                         f"  git clone https://github.com/SysCV/sam-hq {upstream}")
    head = subprocess.run(["git", "-C", str(upstream), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    pinned = pinned_commit()
    if head != pinned and not arguments.allow_commit_drift:
        raise SystemExit(f"upstream is at {head[:12]}, PROVENANCE.md records {pinned[:12]}.\n"
                         "Check it out, or pass --allow-commit-drift.")

    import PIL
    import torchvision

    torch.set_grad_enabled(False)
    print(f"upstream   {upstream} @ {head[:12]}")
    print(f"versions   torch {torch.__version__}, torchvision {torchvision.__version__}, "
          f"Pillow {PIL.__version__}, numpy {np.__version__}; device {device}, fp32")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda: no CUDA device visible")
        print(f"cuda       {cuda_environment()}")
    checkpoints = {v: (arguments.checkpoint or published(v)).resolve() for v in variants}

    if arguments.falsify:
        return falsify(upstream, variants, checkpoints, device, arguments.reference_cache)

    same = Comparison()
    for variant in variants:
        check_variant(upstream, variant, checkpoints[variant], same,
                      end_to_end=arguments.checkpoint is None, device=device, repeat=repeat,
                      swap=arguments.swap_models, cache=arguments.reference_cache)
    print()
    if same.failed:
        print(f"{len(same.failed)} of {same.checked} comparisons differ from the published model "
              f"({device} fp32). First: {same.failed[0]}")
        return 1
    print(f"  {same.checked} comparisons, every one identical to the published model "
          f"({device} fp32{', determinism included' if repeat > 1 else ''}).")
    print("\nHQ-SAM: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

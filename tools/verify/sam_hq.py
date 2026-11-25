#!/usr/bin/env python3
"""Check that mozo's HQ-SAM returns exactly what the published model does.

Three paths reach the same weights. ``SysCV/sam-hq`` itself, driven through its own
``SamPredictor``; ``mozo/vendors/sam_hq_deploy`` -- build a ``Segmenter``, hand it an image and a
prompt; and mozo end to end -- registry, weights, adapter, PixelFlow. Between upstream and the
vendor sit a TinyViT with timm removed, a resize that calls Pillow instead of torchvision, a
decoder whose selection moved out of the module, and a predictor rewritten around a cache -- and
any of those could quietly change a number.

The comparison is exact (``torch.equal``), at every stage: the resized image, the image
embedding, the early feature the HQ decoder fuses, the decoder's raw tokens, and the final
logits, IoU and masks for each prompt.

**One thing mozo returns that upstream does not, and how it is checked.** With
``multimask_output=True`` upstream's ``MaskDecoderHQ.forward`` keeps only the SAM candidate with
the highest predicted IoU and adds the HQ token's mask to it -- one mask. mozo's promptable
contract returns three candidates, so the vendor adds the HQ mask to *each* of the three. The
best of mozo's three is compared against upstream's one answer; all three are compared against
upstream's own raw decoder output (``predict_masks``) put through the same addition and upstream's
``postprocess_masks``. Both must be bit-identical.

Run from the repository root, in an environment with torch, torchvision and Pillow::

    git clone https://github.com/SysCV/sam-hq ~/src/sam-hq
    git -C ~/src/sam-hq checkout <commit in PROVENANCE.md>
    python tools/verify/sam_hq.py --upstream ~/src/sam-hq            # every published variant
    python tools/verify/sam_hq.py --upstream ~/src/sam-hq tiny base
    python tools/verify/sam_hq.py --upstream ~/src/sam-hq --falsify  # each perturbation must FAIL

``--falsify`` runs the gate once per deliberate perturbation of the vendored side and passes only
if every one of them is caught -- a gate that has never failed has not been shown to work.
"""

from __future__ import annotations

import argparse
import contextlib
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

#: Deliberate perturbations of the vendored side, each of which must make the gate fail. Each
#: names the stage it should first fail at, so a falsifier caught somewhere unexpected is visible.
#: The second field is which trunk a perturbation applies to: ``"vit"``, ``"tiny"`` or both.
FALSIFIERS = {
    "resize-filter": ("PIL BILINEAR -> BICUBIC in the resize (fails at: resized image)", None),
    "resize-rounding": ("round-half-even instead of upstream's int(x + 0.5) "
                        "(fails at: resized image, half-pixel size)", None),
    "layernorm-eps": ("ViT LayerNorm eps 1e-6 -> 1e-5 (fails at: image embedding)", "vit"),
    "attention-bias": ("TinyViT attention biases left at zero, upstream's stale-buffer case "
                       "(fails at: image embedding)", "tiny"),
    "interm-index": ("fuse the last global block's feature instead of the first "
                     "(fails at: hq features)", "vit"),
    "hq-token-only": ("HQ token alone instead of SAM + HQ (fails at: final logits)", None),
    "coords-float32": ("scale prompts in float32 instead of float64 (fails at: prompt embedding)",
                       None),
    "no-crop": ("skip cropping the letterbox padding before the final resize "
                "(fails at: final masks)", None),
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


def reference(upstream: Path, variant: str, checkpoint: Path):
    """Build upstream's own predictor, the way its demos do.

    ``eval()`` is called again after the builder, as ``demo/demo_hqsam_light.py`` does: the
    builder calls it before loading, and TinyViT caches its attention biases at that moment, so
    a model used straight from ``build_sam_vit_t`` attends with zeros. See ``tiny_vit.py``.
    """
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    from segment_anything import SamPredictor, sam_model_registry

    with contextlib.redirect_stdout(None):  # the builder prints its load report
        sam = sam_model_registry[UPSTREAM_NAMES[variant]](checkpoint=str(checkpoint))
    return SamPredictor(sam.to("cpu").eval())


class Comparison:
    """Accumulates stage comparisons and reports how many disagreed."""

    def __init__(self, quiet: bool = False) -> None:
        self.failed: list[str] = []
        self.checked = 0
        self.quiet = quiet

    def __call__(self, name: str, ours, theirs) -> None:
        ours = torch.as_tensor(np.asarray(ours)).float()
        theirs = torch.as_tensor(np.asarray(theirs)).float()
        self.checked += 1
        if ours.shape != theirs.shape:
            self._say(f"  FAIL  {name:<58} shape {tuple(ours.shape)} != {tuple(theirs.shape)}")
            self.failed.append(name)
        elif torch.equal(ours, theirs):
            self._say(f"  ok    {name:<58} identical")
        else:
            self._say(f"  FAIL  {name:<58} max|d| {(ours - theirs).abs().max().item():.3e}")
            self.failed.append(name)

    def _say(self, line: str) -> None:
        if not self.quiet:
            print(line)


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


def check_variant(upstream: Path, variant: str, checkpoint: Path, same: Comparison,
                  end_to_end: bool) -> None:
    """Every stage of one variant, on every image and prompt."""
    ours = Segmenter(checkpoint, device="cpu")
    theirs = reference(upstream, variant, checkpoint)
    print(f"\n== {variant}  ({checkpoint.name}, upstream {UPSTREAM_NAMES[variant]})")

    for label, image in images().items():
        shape = image.shape[:2]
        print(f"\n{label}:")
        # --- preprocessing and the encoder ---
        resized = vendor_image.resize_longest_side(image, ours.image_size)
        upstream_resized = torch.as_tensor(theirs.transform.apply_image(image))
        same("resized image", resized, upstream_resized.permute(2, 0, 1).contiguous()[None])
        theirs.set_image(image)
        features = ours.encode(image)
        same("image embedding", features["image_embed"], theirs.features)
        same("hq features (early feature fused)", features["hq_features"], upstream_hq_features(theirs))

        for spec in PROMPTS:
            name = spec[0]
            prompt = prompt_for(spec, shape)
            # --- the prompt encoder ---
            clicks, corners = ours._prompt(prompt["points"], prompt["labels"], prompt["boxes"], shape)
            sparse, dense = ours.network.prompt_encoder(points=clicks, boxes=corners, masks=None)
            coords = labels = box = None
            if prompt["points"] is not None:
                coords = torch.as_tensor(theirs.transform.apply_coords(prompt["points"], shape),
                                         dtype=torch.float)[None]
                labels = torch.as_tensor(prompt["labels"], dtype=torch.int)[None]
            if prompt["boxes"] is not None:
                box = torch.as_tensor(theirs.transform.apply_boxes(prompt["boxes"], shape),
                                      dtype=torch.float)[None]
            their_sparse, their_dense = theirs.model.prompt_encoder(
                points=None if coords is None else (coords, labels), boxes=box, masks=None)
            same(f"{name}: sparse prompt embedding", sparse, their_sparse)
            same(f"{name}: dense prompt embedding", dense, their_dense)

            # --- the decoder, before selection ---
            masks_sam, masks_hq, iou = ours.network.mask_decoder(
                image_embeddings=features["image_embed"],
                image_pe=ours.network.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                hq_features=features["hq_features"])
            their_masks, their_iou = raw_decoder(theirs, their_sparse, their_dense)
            same(f"{name}: SAM tokens (raw)", masks_sam, their_masks[:, :4])
            same(f"{name}: hq token (raw)", masks_hq, their_masks[:, 4:])
            same(f"{name}: iou (raw)", iou, their_iou)

            # --- the answer: upstream's one mask against mozo's best ---
            got = ours.predict(image, **prompt)
            their_mask, their_score, their_low = theirs.predict(
                point_coords=prompt["points"], point_labels=prompt["labels"], box=prompt["boxes"],
                multimask_output=prompt["multimask_output"])
            best = int(np.argmax(got.scores[0]))
            same(f"{name}: logits (upstream's answer)", got.logits[:, best:best + 1], their_low[None])
            same(f"{name}: iou (upstream's answer)", got.scores[:, best:best + 1], their_score[None])
            same(f"{name}: mask (upstream's answer)", got.masks[:, best:best + 1], their_mask[None])

            # --- all three candidates, against upstream's raw output and its own resize ---
            if prompt["multimask_output"]:
                summed = their_masks[:, 1:4] + their_masks[:, 4:5]
                full = theirs.model.postprocess_masks(summed, theirs.input_size, theirs.original_size)
                same(f"{name}: three candidates, logits", got.logits, summed)
                same(f"{name}: three candidates, masks", got.masks, full > theirs.model.mask_threshold)
                same(f"{name}: three candidates, iou", got.scores, their_iou[:, 1:4])

        # --- refinement: feed the best low-res logits back in, as upstream documents ---
        prompt = prompt_for(PROMPTS[0], shape)
        first = ours.predict(image, **prompt)
        best = int(np.argmax(first.scores[0]))
        refine = first.logits[:, best:best + 1]
        got = ours.predict(image, points=prompt["points"], labels=prompt["labels"],
                           mask_input=refine, multimask_output=False)
        their_mask, their_score, their_low = theirs.predict(
            point_coords=prompt["points"], point_labels=prompt["labels"],
            mask_input=refine[0], multimask_output=False)
        same("refinement with mask_input: logits", got.logits, their_low[None])
        same("refinement with mask_input: mask", got.masks, their_mask[None])

        # --- a batch of boxes, through upstream's predict_torch ---
        h, w = shape
        boxes = np.array([[0.55 * w, 0.42 * h, 0.72 * w, 0.78 * h],
                          [0.10 * w, 0.10 * h, 0.50 * w, 0.60 * h]], dtype=np.float32)
        got = ours.predict(image, boxes=boxes, multimask_output=False)
        box_torch = torch.as_tensor(theirs.transform.apply_boxes(boxes, shape), dtype=torch.float)
        their_masks_b, their_iou_b, their_low_b = theirs.predict_torch(
            None, None, boxes=box_torch, multimask_output=False)
        same("batch of two boxes: logits", got.logits, their_low_b)
        same("batch of two boxes: masks", got.masks, their_masks_b)

        # A held image must answer a repeated prompt identically.
        again = ours.predict(image, **prompt)
        same("encoder cache: repeated prompt", again.logits, first.logits)

    if end_to_end:
        mozo_path(variant, ours, same)


def mozo_path(variant: str, ours: Segmenter, same: Comparison) -> None:
    """Registry -> weights -> adapter -> PixelFlow, against the vendor's answer through the same door.

    PixelFlow rounds confidences to its own precision, so the vendor's raw output is handed to the
    adapter's own ``candidates`` (ranking, then ``pf.detections.from_arrays``) rather than
    restating that rounding here; what is checked is that the adapter adds nothing else.
    """
    from mozo.adapters._promptable import candidates
    from mozo.manager import ModelManager

    print("\nmozo end to end:")
    image = load_image(str(FIXTURE))
    prompt = prompt_for(PROMPTS[0], image.shape[:2])
    model = ModelManager().get_model(FAMILY, variant, device="cpu")
    found = model.predict(image, points=prompt["points"], labels=prompt["labels"])
    direct = ours.predict(image, **prompt)
    expected = candidates(direct.masks, direct.scores, None, "cpu")

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
    else:
        raise SystemExit(f"unknown falsifier {name!r}; choose from {sorted(FALSIFIERS)}")
    try:
        yield
    finally:
        for owner, attribute, value in reversed(undo):
            setattr(owner, attribute, value)


def falsify(upstream: Path, variants: list[str], checkpoints: dict[str, Path]) -> int:
    """Run the gate under each perturbation; every applicable one must fail."""
    caught, missed = [], []
    for name, (what, trunk) in FALSIFIERS.items():
        for variant in variants:
            if trunk is not None and trunk != ("tiny" if variant == "tiny" else "vit"):
                continue
            same = Comparison(quiet=True)
            with perturbed(name), contextlib.redirect_stdout(None):
                check_variant(upstream, variant, checkpoints[variant], same, end_to_end=False)
            verdict = (caught if same.failed else missed)
            verdict.append((name, variant))
            first = same.failed[0] if same.failed else "-"
            print(f"  {'caught' if same.failed else 'MISSED':<7} {name:<16} {variant:<6} "
                  f"{len(same.failed):>3}/{same.checked} stages fail; first: {first}")
            print(f"          {what}")
    print()
    if missed:
        print(f"{len(missed)} perturbation(s) passed the gate: {missed}")
        return 1
    print(f"all {len(caught)} perturbations caught.\nHQ-SAM falsification: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("variants", nargs="*", help="variants to check (default: all published)")
    parser.add_argument("--upstream", type=Path, required=True, help="checkout of SysCV/sam-hq")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="a checkpoint to check instead of the published one (one variant)")
    parser.add_argument("--falsify", action="store_true",
                        help="run each deliberate perturbation; every one must fail")
    parser.add_argument("--allow-commit-drift", action="store_true",
                        help="run against an upstream other than the extracted commit")
    arguments = parser.parse_args()

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
          f"Pillow {PIL.__version__}, numpy {np.__version__}; device cpu")
    checkpoints = {v: (arguments.checkpoint or published(v)).resolve() for v in variants}

    if arguments.falsify:
        return falsify(upstream, variants, checkpoints)

    same = Comparison()
    for variant in variants:
        check_variant(upstream, variant, checkpoints[variant], same,
                      end_to_end=arguments.checkpoint is None)
    print()
    if same.failed:
        print(f"{len(same.failed)} of {same.checked} stages differ from the published model.")
        return 1
    print(f"  {same.checked} stages, every one identical to the published model.")
    print("\nHQ-SAM: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

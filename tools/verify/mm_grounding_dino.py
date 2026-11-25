#!/usr/bin/env python3
"""Check that mozo's MM-Grounding-DINO returns exactly what the published model does.

Three paths reach the same weights and this compares all three. ``mozo/vendors/
mm_grounding_dino_deploy`` is one. ``transformers``' ``MMGroundingDinoForObjectDetection`` with its
``AutoProcessor`` is the second -- the reference, because the checkpoints are the HF-format files
the LLMDet authors publish and tell users to run with it. mozo itself is the third: registry,
weights, adapter, PixelFlow.

**The comparison is exact**, ``torch.equal`` at every stage: pixels, token ids, BERT, projected
text, every Swin level handed to the neck, encoder image and text memory, each decoder layer's
output and reference boxes, the final logits and boxes -- and the detections after
post-processing, against the processor's own ``post_process_grounded_object_detection``.

**The reference must be transformers 4.x.** 5.x (measured on 5.9.0 and 5.12.1) loads these
checkpoints with the six decoder box heads tied to the first and every score bias set to the
first's, so it runs a different model; this script refuses to compare against it. 4.55.0 -- the
version the LLMDet README names -- and 4.57.3 both match this package exactly::

    python -m venv --system-site-packages refenv && ./refenv/bin/pip install 'transformers==4.57.3'
    MKL_CBWR=AVX2,STRICT ./refenv/bin/python tools/verify/mm_grounding_dino.py --reference

``MKL_CBWR`` pins MKL's kernel choice, which otherwise depends on buffer alignment and can move
the last bit between two processes. Without ``--reference`` only the mozo-vs-vendor half runs.
``--falsify NAME`` perturbs one thing in the vendor and expects the gate to fail at that stage.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))

os.environ.setdefault("MOZO_BASE_URL", f"file://{ROOT / 'weights'}")

from common import fixtures  # noqa: E402
from conftest import as_pixelflow_reports  # noqa: E402
from mozo.image import load_image  # noqa: E402
from mozo.vendors.mm_grounding_dino_deploy import SEPARATORS, SPECS, Predictor  # noqa: E402
from mozo.vendors.mm_grounding_dino_deploy.checkpoint import build  # noqa: E402
from mozo.vendors.mm_grounding_dino_deploy.image import preprocess  # noqa: E402
from mozo.vendors.mm_grounding_dino_deploy.network import phrase_masks  # noqa: E402
from mozo.vendors.mm_grounding_dino_deploy.predictor import caption_for  # noqa: E402
from mozo.vendors.mm_grounding_dino_deploy.text.tokenizer import Tokenizer  # noqa: E402
from mozo.weights import WeightsError, resolve  # noqa: E402

FAMILY = "mm_grounding_dino"

#: Where the reference loads each variant from: the authors' repositories, at the commits
#: ``tools/fetch/mm_grounding_dino.py`` publishes.
REPOS = {
    "llmdet-tiny": ("iSEE-Laboratory/llmdet_tiny", "d05199165a19320a9236396e20fca0a5e065189c"),
    "llmdet-base": ("iSEE-Laboratory/llmdet_base", "3537869413e7dca01cfd86d6ab10553b72e4fe29"),
    "llmdet-large": ("iSEE-Laboratory/llmdet_large", "bec37f296f05b22f6c6b39bc05a6c611239f4e31"),
}

#: Prompt sets: one phrase, several, multi-word phrases whose tokens must stay grouped, mixed case
#: and inner punctuation (which the caption lowercases and BERT splits), and one matching nothing.
PROMPTS = [
    ["person"],
    ["person", "laptop", "cup"],
    ["a yellow school bus", "a person holding a mug"],
    ["Coffee-Mug", "USB cable", "the man's hand"],
    ["dinosaur"],
]

#: Phrases for the tokenizer fuzz, beyond the random ones it generates.
_TOKENIZER_CASES = [
    "Café", "naïve façade", "東京タワー", "Ελληνικά", "русский", "عربى", "emoji 🙂 face",
    "tab\there", "ﬁ ligature", "ǅ titlecase", "İstanbul", "ß straße", "a/b-c_d", "x" * 120,
    "́accent", "[MASK] token", "[unused1]", "  spaced   out  ", "Ⅻ roman", "½ half",
]

#: Perturbations for ``--falsify``: what each breaks, and so the first stage that must fail.
#: Each touches the vendor only -- the reference shares torch, so patching a torch function would
#: move both sides and prove nothing.
FALSIFIERS = {
    "swin-norm-eps": "Swin LayerNorm eps 1e-5 -> 1e-6 (swin onwards; bert and text pass)",
    "temperature": "image position temperature 20 -> 21 (memory onwards; swin, bert, text pass)",
    "no-score-scale": "drop the contrastive 1/sqrt(256) (decoder onwards; memory passes)",
    "shared-box-head": "every decoder layer uses layer 0's box head, as transformers 5 loads it "
                       "(references onwards)",
    "bert-eager": "BERT softmax by hand instead of SDPA (bert onwards)",
    "resize-rounding": "Grounding DINO's 167x1336 resize rounding (pixels, on the panorama only)",
}


@contextlib.contextmanager
def falsified(name: str | None):
    """Patch the vendor's modules for one perturbation; yields a function to apply to a model."""
    import math

    from mozo.vendors.mm_grounding_dino_deploy import image, network, transformer
    from mozo.vendors.mm_grounding_dino_deploy.text import bert

    undo = []

    def patch(owner, attribute, value):
        undo.append((owner, attribute, getattr(owner, attribute)))
        setattr(owner, attribute, value)

    def mutate(model):
        return model

    if name is None:
        pass
    elif name == "swin-norm-eps":
        def mutate(model):  # noqa: F811
            for module in model.backbone.modules():
                if isinstance(module, torch.nn.LayerNorm):
                    module.eps = 1e-6
            return model
    elif name == "temperature":
        position = network.image_position
        patch(network, "image_position", lambda valid, feats, temperature:
              position(valid, feats, temperature + 1))
    elif name == "no-score-scale":
        def forward(self, vision, text, text_valid):
            scores = vision @ text.transpose(-1, -2) + self.bias
            scores = scores.masked_fill(~text_valid[:, None, :], float("-inf"))
            padded = torch.full((*scores.shape[:-1], self.max_text_len), float("-inf"))
            padded[..., : scores.shape[-1]] = scores
            return padded
        patch(transformer.ContrastiveEmbedding, "forward", forward)
    elif name == "shared-box-head":
        def mutate(model):  # noqa: F811
            for head in model.bbox_embed[1:]:
                head.load_state_dict(model.bbox_embed[0].state_dict())
            return model
    elif name == "bert-eager":
        def forward(self, hidden, mask):
            batch, tokens, _ = hidden.shape
            q, k, v = (self._heads(f(hidden)) for f in (self.query, self.key, self.value))
            scores = q @ k.transpose(-1, -2) / math.sqrt(self.attention_head_size) + mask
            context = (scores.softmax(-1) @ v).transpose(1, 2)
            return context.reshape(batch, tokens, -1)
        patch(bert._SelfAttention, "forward", forward)
    elif name == "resize-rounding":
        def rounded_first(height, width, short_side, max_side):
            small, large = float(min(height, width)), float(max(height, width))
            if large / small * short_side > max_side:
                short_side = int(round(max_side * small / large))
            if (width <= height and width == short_side) or (
                    height <= width and height == short_side):
                return height, width
            if width < height:
                return int(short_side * height / width), short_side
            return short_side, int(short_side * width / height)
        patch(image, "resized_size", rounded_first)
    else:
        raise SystemExit(f"unknown falsifier {name!r}. Known: {sorted(FALSIFIERS)}")
    try:
        yield mutate
    finally:
        for owner, attribute, value in reversed(undo):
            setattr(owner, attribute, value)


def images() -> dict:
    """Every fixture photograph, plus crops that reach the awkward geometry.

    A crop wider than 1333/800 makes the long-side cap bind (where this processor's rounding
    differs from Grounding DINO's), a narrow one makes the image portrait, and a tiny one is
    upscaled and padded inside Swin at every stage.
    """
    out = {}
    for path in fixtures():
        image = load_image(str(path))
        height, width = image.shape[:2]
        out[path.name] = image
        out[f"{path.name}[panorama]"] = image[: max(1, width // 8), :]
        out[f"{path.name}[portrait]"] = image[:, : max(1, width // 9)]
        out[f"{path.name}[tiny]"] = image[:37, :45]
    return out


def require_reference():
    """Import ``transformers`` and refuse a version that runs a different model."""
    import transformers

    major = int(transformers.__version__.split(".")[0])
    if major >= 5:
        raise SystemExit(
            f"transformers {transformers.__version__} ties these checkpoints' six decoder heads to "
            "the first on load, so it is not the reference. Use 4.55-4.57 (see the docstring).")
    return transformers


def reference(variant: str):
    """The upstream model and processor at the pinned commit, parameters re-allocated."""
    transformers = require_reference()
    repo, commit = REPOS[variant]
    model = transformers.MMGroundingDinoForObjectDetection.from_pretrained(
        repo, revision=commit).eval()
    # ``from_pretrained`` leaves tensors at the checkpoint's offsets; BLAS picks vector paths by
    # alignment, so re-allocate before comparing arithmetic (VENDORING section 4).
    for parameter in model.parameters():
        parameter.data = parameter.data.clone()
    processor = transformers.AutoProcessor.from_pretrained(repo, revision=commit)
    if type(processor.image_processor).__name__ != "GroundingDinoImageProcessor":
        raise SystemExit(f"expected the slow image processor, got {type(processor.image_processor)}")
    return model, processor


def tap(module, store: dict, name: str, read=lambda output: output):
    """Record one module's output under *name* every time it runs."""
    module.register_forward_hook(lambda _m, _i, output: store.__setitem__(name, read(output)))


def taps(ours, theirs, mine: dict, upstream: dict) -> None:
    """Hook the same stages on both sides, from one function so they cannot drift apart."""
    tap(ours.text_backbone, mine, "bert")
    tap(theirs.model.text_backbone, upstream, "bert", lambda o: o.last_hidden_state)
    tap(ours.text_projection, mine, "text")
    tap(theirs.model.text_projection, upstream, "text")
    tap(ours.backbone, mine, "swin")
    tap(theirs.model.backbone.conv_encoder.model, upstream, "swin", lambda o: list(o.feature_maps))
    tap(ours.encoder, mine, "memory", lambda o: o[0])
    tap(theirs.model.encoder, upstream, "memory", lambda o: o[0])
    tap(ours.encoder, mine, "memory_text", lambda o: o[1])
    tap(theirs.model.encoder, upstream, "memory_text", lambda o: o[1])
    tap(ours.decoder, mine, "decoder", lambda o: o[0])
    tap(theirs.model.decoder, upstream, "decoder",
        lambda o: list(o.intermediate_hidden_states.unbind(1)))
    tap(ours.decoder, mine, "references", lambda o: o[1][1:])
    tap(theirs.model.decoder, upstream, "references",
        lambda o: list(o.intermediate_reference_points.unbind(1)))


def _differs(got, want) -> str | None:
    if got.shape != want.shape:
        return f"shape {tuple(got.shape)} vs {tuple(want.shape)}"
    if torch.equal(got, want):
        return None
    finite = want.isfinite() & got.isfinite()
    if not torch.equal(got.isfinite(), want.isfinite()):
        return "different non-finite entries"
    return f"{(got[finite].double() - want[finite].double()).abs().max().item():.3e}"


def against_tokenizer(processor, seed: int = 0, count: int = 20000) -> tuple[int, list[str]]:
    """The vendor's caption and WordPiece against the processor's, on hand-picked and random text.

    The processor lowercases with Python's ``str.lower`` and then its Rust normaliser lowercases,
    strips accents and drops control characters by its own Unicode tables; the vendor reads those
    tables from an asset derived from it. Fuzzed because a table built one codepoint at a time
    could still miss what happens between neighbours.
    """
    tokenizer = Tokenizer()
    rng = random.Random(seed)
    # Half the draws from the scripts people write prompts in, half from anywhere at all. Never
    # ``.`` or ``?``: a prompt carrying one is refused before it is tokenized.
    common = [chr(c) for c in range(32, 0x3000)]
    anywhere = [chr(c) for c in range(32, 0x110000) if not 0xD800 <= c <= 0xDFFF]

    def draw(alphabet):
        return "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 12))).replace(
            ".", "").replace("?", "") or "x"

    cases = [[case] for case in _TOKENIZER_CASES]
    for index in range(count):
        alphabet = common if index % 2 else anywhere
        cases.append([draw(alphabet) for _ in range(rng.randint(1, 3))])
    failures = []
    for prompts in cases:
        if any(not p.strip() for p in prompts):
            continue
        want = processor.tokenizer(processor._preprocess_input_text([prompts])[0])["input_ids"]
        got, _, _ = tokenizer.encode(caption_for(prompts))
        if got != want:
            failures.append(f"tokenizer {prompts!r}: {got[:12]} vs {want[:12]}")
    return len(cases), failures


def against_reference(variant: str, checkpoint: Path, falsify: str | None) -> tuple[int, list[str]]:
    """Compare every stage of the vendored model, and its detections, against the reference."""
    from PIL import Image

    spec = SPECS[variant]
    theirs, processor = reference(variant)
    checked, failures = against_tokenizer(processor)

    with falsified(falsify) as mutate:
        ours = mutate(build(spec, checkpoint))
        vendor = Predictor(checkpoint, spec)
        vendor.model = mutate(vendor.model)
        vendor._forward = vendor.model
        tokenizer = Tokenizer()
        separators = torch.tensor(tokenizer.convert_tokens_to_ids(list(SEPARATORS)))
        mine: dict = {}
        upstream: dict = {}
        taps(ours, theirs, mine, upstream)

        for name, image in images().items():
            for prompts in PROMPTS:
                label = f"{name} {prompts}"
                inputs = processor(images=Image.fromarray(image), text=[prompts],
                                   return_tensors="pt")
                pixels = preprocess(image, spec.short_side, spec.max_side)
                ids, types, mask = tokenizer.encode(caption_for(prompts))
                checked += 2
                if (problem := _differs(pixels, inputs["pixel_values"][0])) is not None:
                    failures.append(f"{label}: pixels {problem}")
                    continue
                if ids != inputs["input_ids"][0].tolist():
                    failures.append(f"{label}: token ids differ")
                    continue

                batch_ids = torch.tensor([ids])
                attention, positions, _ = phrase_masks(batch_ids, separators)
                with torch.no_grad():
                    logits, boxes = ours(pixels[None], batch_ids, torch.tensor([types]),
                                         torch.tensor([mask], dtype=torch.bool), attention,
                                         positions)
                    want = theirs(**inputs)

                for stage in mine:
                    got, expected = mine[stage], upstream[stage]
                    pairs = (list(zip(got, expected)) if isinstance(got, list)
                             else [(got, expected)])
                    for level, (a, b) in enumerate(pairs):
                        checked += 1
                        if (problem := _differs(a, b)) is not None:
                            where = f"{stage}[{level}]" if isinstance(got, list) else stage
                            failures.append(f"{label}: {where} {problem}")
                checked += 2
                if (problem := _differs(logits, want.logits)) is not None:
                    failures.append(f"{label}: logits {problem}")
                if (problem := _differs(boxes, want.pred_boxes)) is not None:
                    failures.append(f"{label}: boxes {problem}")

                # The detections, against upstream's own post-processing at the same threshold.
                height, width = image.shape[:2]
                post = processor.post_process_grounded_object_detection(
                    want, inputs["input_ids"], threshold=spec.box_threshold,
                    target_sizes=[(height, width)])[0]
                kept = vendor(image, prompts)
                checked += 1
                if (len(kept) != len(post["scores"])
                        or [d.score for d in kept] != post["scores"].tolist()
                        or [list(d.box) for d in kept] != post["boxes"].tolist()):
                    # The vendor drops a query whose peak token is a separator, so a count
                    # mismatch alone is not a failure: every vendor detection must be upstream's.
                    upstream_rows = list(zip(post["scores"].tolist(), post["boxes"].tolist()))
                    if not all((d.score, list(d.box)) in upstream_rows for d in kept):
                        failures.append(f"{label}: detections differ from post-processing")
    return checked, failures


def against_mozo(variant: str, checkpoint: Path) -> tuple[int, list[str]]:
    """The adapter sorts, converts to PixelFlow and rounds. It must change nothing else."""
    from mozo.adapters.mm_grounding_dino import MMGroundingDinoPredictor

    vendor = Predictor(checkpoint, SPECS[variant], device="cpu")
    adapter = MMGroundingDinoPredictor(variant, device="cpu", checkpoint_path=checkpoint)

    checked, failures = 0, []
    for name, image in images().items():
        for prompts in PROMPTS:
            raw = sorted(vendor(image, prompts), key=lambda d: -d.score)
            got = adapter.predict(image, prompts).to_dict()
            checked += 1
            if len(raw) != len(got):
                failures.append(f"{name} {prompts}: {len(raw)} from vendor, {len(got)} from mozo")
                continue
            want_boxes, want_scores = as_pixelflow_reports(
                [d.box for d in raw], [d.score for d in raw], [d.prompt_index for d in raw])
            if [list(b) for b in want_boxes] != [row["bbox"] for row in got]:
                failures.append(f"{name} {prompts}: boxes differ after rounding")
            elif list(want_scores) != [row["confidence"] for row in got]:
                failures.append(f"{name} {prompts}: scores differ after rounding")
            elif [prompts[d.prompt_index] for d in raw] != [row["class_name"] for row in got]:
                failures.append(f"{name} {prompts}: names differ")
    return checked, failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variants", nargs="*", default=["llmdet-tiny"],
                        help=f"variants to check (default: llmdet-tiny; known: {list(SPECS)})")
    parser.add_argument("--reference", action="store_true",
                        help="also compare against transformers 4.x (see the docstring)")
    parser.add_argument("--falsify", choices=sorted(FALSIFIERS),
                        help="perturb the vendor and expect the reference comparison to fail")
    args = parser.parse_args()
    if args.falsify and not args.reference:
        parser.error("--falsify needs --reference")

    print(f"torch {torch.__version__}")
    if args.reference:
        print(f"transformers {require_reference().__version__}")
    total, problems = 0, []
    for variant in args.variants:
        try:
            checkpoint = resolve(FAMILY, variant, "torch-fp32")
        except WeightsError as error:
            print(f"{variant}: {error}")
            return 1

        if not args.falsify:
            checked, failures = against_mozo(variant, checkpoint)
            total += checked
            problems.extend(f"[mozo/{variant}] {f}" for f in failures)
            print(f"{variant}: {checked} mozo-vs-vendor comparisons, {len(failures)} failed")

        if args.reference:
            checked, failures = against_reference(variant, checkpoint, args.falsify)
            total += checked
            problems.extend(f"[reference/{variant}] {f}" for f in failures)
            print(f"{variant}: {checked} reference comparisons, {len(failures)} failed")

    if args.falsify:
        print(f"\nfalsified with {args.falsify}: {FALSIFIERS[args.falsify]}")
        first = problems[0] if problems else "nothing"
        print(f"{len(problems)} failures; first: {first}")
        return 0 if problems else 1

    if problems:
        print(f"\n{len(problems)} FAILURES:")
        for problem in problems[:40]:
            print(f"  {problem}")
        return 1

    print(f"\n{total} comparisons, all identical.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

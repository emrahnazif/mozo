#!/usr/bin/env python3
"""Fetch LLMDet's published checkpoints into the local ``weights/`` tree.

Bootstrap tooling. It runs on a machine you control, never ships, and is the only place the
upstream addresses live.

    python tools/fetch/mm_grounding_dino.py                 # all three
    python tools/fetch/mm_grounding_dino.py llmdet-tiny

**Three checkpoints, one per size**, from the authors' own Hugging Face organisation
(``iSEE-Laboratory/llmdet_{tiny,base,large}``) at pinned revisions. They are the MM-Grounding-DINO
architecture in ``transformers``' format; the LLMDet README names them as the way to run the model.
MM-Grounding-DINO's own checkpoints run on the same code and are not fetched -- see the vendor's
PROVENANCE for why.

**Safetensors only, so mozo repacks**, as ``tools/fetch/siglip2.py`` does: the file is verified
against the sha256 the Hub records, read, and the same tensors are written back out under the same
names as an ordinary ``.pth``. No tensor is altered, renamed, cast or dropped -- the vendor maps
names at load time. ``safetensors`` is a dependency of this script, not of mozo.

**On the licence.** Each repository's model card carries ``license: apache-2.0``, ungated, and the
LLMDet code repository is Apache-2.0. The LICENSE placed beside the weights is the authors' own
file from that repository.

The output is ``weights/mm_grounding_dino/<variant>/<revision>/torch-fp32.pth``, a NOTICE and the
LICENSE.

``--derive-tokenizer`` instead rebuilds the vendored tokenizer's character table from the reference
tokenizer (``tokenizers``, the Rust library ``BertTokenizerFast`` runs on): what its normaliser does
to every codepoint, and which codepoints its pre-tokenizer treats as whitespace or punctuation. It
is committed to the repository, as SigLIP 2's tokenizer asset is.
"""

from __future__ import annotations

import gzip
import json
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from common import digest, download_verified, require_licence, variant_parser  # noqa: E402

_HF = "https://huggingface.co"

#: The revision this family is published under.
REVISION = "2026-11-25"

#: The authors' LICENSE, which the weights repositories point back to.
LICENCE_SOURCE = "https://raw.githubusercontent.com/iSEE-Laboratory/LLMDet/main/LICENSE"


@dataclass(frozen=True)
class Checkpoint:
    """One published checkpoint: where it lives, at which commit, and what it is."""

    variant: str
    repo: str
    #: The Hub commit the file is fetched at, so a later push upstream cannot change what a
    #: revision of mozo means.
    commit: str
    backbone: str
    #: The training data as the model card lists it.
    trained_on: str
    #: LVIS minival AP, zero-shot, as the LLMDet README reports it.
    lvis_minival: str


CHECKPOINTS: dict[str, Checkpoint] = {
    entry.variant: entry
    for entry in (
        Checkpoint("llmdet-tiny", "iSEE-Laboratory/llmdet_tiny",
                   "d05199165a19320a9236396e20fca0a5e065189c", "Swin-T",
                   "Objects365 v1, GoldG, GRIT, V3Det and GroundingCap-1M", "44.7"),
        Checkpoint("llmdet-base", "iSEE-Laboratory/llmdet_base",
                   "3537869413e7dca01cfd86d6ab10553b72e4fe29", "Swin-B",
                   "Objects365 v1, GoldG, V3Det and GroundingCap-1M", "48.3"),
        Checkpoint("llmdet-large", "iSEE-Laboratory/llmdet_large",
                   "bec37f296f05b22f6c6b39bc05a6c611239f4e31", "Swin-L",
                   "Objects365 v2, OpenImages v6, GoldG and GroundingCap-1M", "51.1"),
    )
}


def recorded_sha256(entry: Checkpoint) -> str:
    """The sha256 the Hub records for ``model.safetensors`` at the pinned commit.

    Read from the API rather than typed here: a digest transcribed by hand verifies the
    transcription, not the file.
    """
    url = f"{_HF}/api/models/{entry.repo}/revision/{entry.commit}?blobs=true"
    with urllib.request.urlopen(url, timeout=60) as response:
        listing = json.load(response)
    for sibling in listing["siblings"]:
        if sibling["rfilename"] == "model.safetensors":
            return sibling["lfs"]["sha256"]
    raise SystemExit(f"{entry.repo}@{entry.commit} has no model.safetensors")


def _notice(entry: Checkpoint, sha256: str) -> str:
    """Attribution that travels with the weights, as Apache-2.0 asks."""
    return (
        f"LLMDet -- {entry.variant} ({entry.backbone})\n"
        "open-vocabulary object detection on the MM-Grounding-DINO architecture\n"
        f"trained on {entry.trained_on}\n"
        f"{entry.lvis_minival} AP on LVIS minival, zero-shot, as reported by the authors\n\n"
        "Copyright the LLMDet authors, iSEE Laboratory, Sun Yat-sen University.\n"
        "Shenghao Fu, Qize Yang, Qijie Mo, Junkai Yan, Xihan Wei, Jingke Meng, Xiaohua Xie,\n"
        "Wei-Shi Zheng.\n\n"
        f"Source:  {_HF}/{entry.repo}/blob/{entry.commit}/model.safetensors\n"
        f"         sha256 {sha256}\n"
        "Project: https://github.com/iSEE-Laboratory/LLMDet\n"
        "Paper:   LLMDet: Learning Strong Open-Vocabulary Object Detectors under the Supervision\n"
        "         of Large Language Models (CVPR 2025), arXiv:2501.18954\n"
        "Builds on: MM-Grounding-DINO (OpenMMLab, arXiv:2401.02361, Apache-2.0) and\n"
        "         Grounding DINO (IDEA Research, Apache-2.0).\n"
        "Licence: Apache-2.0 (full text in the LICENSE file beside this one)\n\n"
        "Where that licence is claimed. The model card of the repository above states\n"
        "`license: apache-2.0`, and the LLMDet code repository is Apache-2.0; the LICENSE beside\n"
        "this file is that repository's own.\n\n"
        "This file is not the byte stream the authors serve: tools/fetch/mm_grounding_dino.py\n"
        "verifies the safetensors against the sha256 above, then writes the same tensors, under\n"
        "the same names, as an ordinary PyTorch checkpoint. No tensor is altered, renamed, cast\n"
        "or dropped.\n\n"
        "Training data. The weights carry Apache-2.0; the datasets they were trained on carry\n"
        "their own terms, which are not restated here. GoldG includes Flickr30k Entities, whose\n"
        "images are under Flickr's terms of use, and GroundingCap-1M is built partly on COCO\n"
        "train2017 -- so COCO validation scores are not zero-shot for these weights.\n\n"
        "The checkpoint carries its own fine-tuned BERT-base text encoder. The vocabulary that\n"
        "tokenizes prompts for it is Google's `bert-base-uncased` vocab.txt, Apache-2.0,\n"
        "byte-identical to the vocab.txt in the repository above, and ships inside mozo.\n"
    )


def fetch(variant: str, revision: str, weights_dir: Path) -> None:
    """Download one variant's checkpoint, verify it, repack it, and place it with its terms."""
    entry = CHECKPOINTS[variant]
    target = weights_dir / "mm_grounding_dino" / variant / revision / "torch-fp32.pth"
    target.parent.mkdir(parents=True, exist_ok=True)

    sha256 = recorded_sha256(entry)
    part = target.with_name("model.safetensors")
    if not target.is_file():
        download_verified(f"{_HF}/{entry.repo}/resolve/{entry.commit}/model.safetensors", part,
                          sha256, label=variant, width=12, detail="Apache-2.0")
        from safetensors.torch import load_file

        print(f"  {variant:12s} repacking safetensors -> state dict")
        torch.save(load_file(part), target)
    part.unlink(missing_ok=True)

    (target.parent / "NOTICE").write_text(_notice(entry, sha256))
    require_licence(target.parent, "Apache-2.0", LICENCE_SOURCE)


#: Where the tokenizer's character table lands.
CHARS_ASSET = ROOT / "mozo" / "vendors" / "mm_grounding_dino_deploy" / "assets" / "bert_chars.json.gz"


def _ranges(codes: list[int]) -> list[list[int]]:
    """Sorted codepoints -> inclusive ``[start, end]`` runs."""
    runs: list[list[int]] = []
    for code in codes:
        if runs and code == runs[-1][1] + 1:
            runs[-1][1] = code
        else:
            runs.append([code, code])
    return runs


def derive_tokenizer() -> None:
    """Probe the reference tokenizer one codepoint at a time and write what it does.

    ``BertNormalizer`` (clean text, CJK spacing, strip accents, lowercase -- the configuration
    every LLMDet ``tokenizer.json`` states) is asked for each codepoint alone; ``BertPreTokenizer``
    is asked to split ``"a" + c + "b"``, which comes back as three pieces for punctuation, two for
    whitespace and one otherwise. Needs ``tokenizers``, bootstrap-only.
    """
    import tokenizers
    from tokenizers import normalizers, pre_tokenizers

    normaliser = normalizers.BertNormalizer(clean_text=True, handle_chinese_chars=True,
                                           strip_accents=None, lowercase=True)
    splitter = pre_tokenizers.BertPreTokenizer()
    mapped: dict[str, str] = {}
    classes: dict[str, list[int]] = {name: [] for name in
                                     ("drop", "space", "spaced", "whitespace", "punctuation")}
    for code in range(0x110000):
        if 0xD800 <= code <= 0xDFFF:            # lone surrogates are not encodable text
            continue
        char = chr(code)
        text = normaliser.normalize_str(char)
        if text == "":
            classes["drop"].append(code)
        elif text == " ":
            classes["space"].append(code)
        elif text == f" {char} ":
            classes["spaced"].append(code)
        elif text != char:
            mapped[str(code)] = text
        pieces = [piece for piece, _ in splitter.pre_tokenize_str(f"a{char}b")]
        if pieces == ["a", "b"]:
            classes["whitespace"].append(code)
        elif pieces == ["a", char, "b"]:
            classes["punctuation"].append(code)
        elif pieces != [f"a{char}b"]:
            raise SystemExit(f"U+{code:04X} pre-tokenizes as {pieces}; the table assumes not")

    payload = {"tokenizers": tokenizers.__version__, "mapped": mapped,
               **{name: _ranges(codes) for name, codes in classes.items()}}
    CHARS_ASSET.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0, so a rebuild from the same reference is byte-identical and its sha256 meaningful.
    CHARS_ASSET.write_bytes(gzip.compress(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode(), 9, mtime=0))
    print(f"{CHARS_ASSET.relative_to(ROOT)}  {CHARS_ASSET.stat().st_size:,} bytes, "
          f"tokenizers {tokenizers.__version__}")
    print(f"  {len(mapped):,} mapped, " + ", ".join(
        f"{len(codes):,} {name}" for name, codes in classes.items()))
    print(f"  sha256 {digest(CHARS_ASSET)}")


def main() -> int:
    if "--derive-tokenizer" in sys.argv[1:]:
        derive_tokenizer()
        return 0
    args = variant_parser(__doc__, ROOT / "weights", revision=REVISION).parse_args()

    wanted = args.variants or list(CHECKPOINTS)
    unknown = [v for v in wanted if v not in CHECKPOINTS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}. Known: {list(CHECKPOINTS)}")

    for variant in wanted:
        fetch(variant, args.revision, args.weights_dir)

    print(f"\n{len(wanted)} checkpoints in {args.weights_dir}, all Apache-2.0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

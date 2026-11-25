#!/usr/bin/env python3
"""Fetch Microsoft's Florence-2 checkpoints into the local ``weights/`` tree.

Bootstrap tooling: runs on a machine you control, never ships, and is the only place the upstream
URLs live.

    python tools/fetch/florence2_detect.py                  # every variant
    python tools/fetch/florence2_detect.py base large
    python tools/fetch/florence2_detect.py --derive-vocab   # rebuild the vocabulary asset

**Pinned to a commit, verified against the Hub's own hash.** Files are read from
``resolve/<commit>/`` and each weight file's sha256 is read from the Hub's API at fetch time.

**Microsoft stores the weights in float16, and mozo publishes them that way, as ``torch-fp16``.**
``torch-fp32`` is built from that file at load (``mozo.registry.DERIVES``): every float16 value is a
float32 value, so the upcast is exact, and it is what ``from_pretrained(torch_dtype=float32)`` does.
The file is re-saved with ``torch.save`` (mozo does not depend on ``safetensors``); no tensor is
renamed, cast or dropped. ``safetensors`` is a dependency of *this script* only.

The output is ``weights/florence2_detect/<variant>/<revision>/torch-fp16.pth``, Microsoft's
``LICENSE`` (MIT) as served by the same repository and commit, and a NOTICE.
"""

from __future__ import annotations

import gzip
import json
import sys
import urllib.request
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from common import digest, download_verified, hub_lfs_sha256s, variant_parser  # noqa: E402

from mozo.vendors.florence2_deploy.config import SPECS  # noqa: E402
from mozo.vendors.florence2_deploy.tokenizer import VOCAB_PATH  # noqa: E402

_HF = "https://huggingface.co"
FAMILY = "florence2_detect"

#: The revision this family is published under.
REVISION = "2026-11-25"

#: Variant -> (repository, commit, weight file).
SOURCES = {
    "base": ("microsoft/Florence-2-base", "5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac",
             "model.safetensors"),
    "large": ("microsoft/Florence-2-large", "21a599d414c4d928c9032694c424fb94458e3594",
              "model.safetensors"),
    "base-ft": ("microsoft/Florence-2-base-ft", "f6c1a25888ffc1d945ee8a1a77ac833c7303d46e",
                "model.safetensors"),
    "large-ft": ("microsoft/Florence-2-large-ft", "4a12a2b54b7016a48a22037fbd62da90cd566f2a",
                 "model.safetensors"),
}

#: The tokens ``Florence2Processor.__init__`` adds to BART's vocabulary, in its order.
ADDED = (["<od>", "</od>", "<ocr>", "</ocr>"] + [f"<loc_{x}>" for x in range(1000)]
         + ["<cap>", "</cap>", "<ncap>", "</ncap>", "<dcap>", "</dcap>", "<grounding>",
            "</grounding>", "<seg>", "</seg>", "<sep>", "<region_cap>", "</region_cap>",
            "<region_to_desciption>", "</region_to_desciption>", "<proposal>", "</proposal>",
            "<poly>", "</poly>", "<and>"])


def published(variant: str, revision: str = REVISION, weights_dir: Path = ROOT / "weights") -> Path:
    """Where this tool places *variant*'s checkpoint: what the verify, bench and export tools read."""
    return weights_dir / FAMILY / variant / revision / "torch-fp16.pth"


def resolve(repo: str, commit: str, name: str) -> str:
    return f"{_HF}/{repo}/resolve/{commit}/{name}"


def _notice(variant: str, sha: str) -> str:
    repo, commit, name = SOURCES[variant]
    return (
        f"Florence-2 -- {variant} ({repo})\n"
        "a vision foundation model; mozo serves its <OD> task as florence2_detect\n\n"
        "Copyright (c) Microsoft Corporation.\n\n"
        f"Source:  {_HF}/{repo} at commit {commit}\n"
        "Paper:   Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks,\n"
        "         arXiv:2311.06242\n"
        "Licence: MIT (full text in the LICENSE file beside this one, which is the LICENSE file\n"
        "         of the same repository at the same commit). The model card's metadata says\n"
        "         `license: mit` and links that file. The repository is not gated.\n\n"
        "This file is not the byte stream Microsoft serves. tools/fetch/florence2_detect.py\n"
        f"verifies {name} against the sha256 the Hub records,\n"
        f"           {sha}\n"
        "then writes the same tensors out as one ordinary torch checkpoint, still in float16.\n"
        "Nothing is renamed, cast or dropped.\n\n"
        "The vocabulary shipped inside mozo derives from tokenizer.json in the Florence-2-base\n"
        "repository, under the same licence.\n"
    )


def fetch(variant: str, revision: str, weights_dir: Path) -> None:
    """Download one variant's checkpoint, verify it, and place it with its licence and notice."""
    repo, commit, name = SOURCES[variant]
    target = published(variant, revision, weights_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    sha = hub_lfs_sha256s(repo, commit)[name]
    if not target.is_file():
        staged = target.with_name(name)
        download_verified(resolve(repo, commit, name), staged, sha, label=variant, width=8,
                          detail="MIT")
        from safetensors.torch import load_file

        state = load_file(staged)
        dtypes = {str(t.dtype) for t in state.values()}
        if dtypes != {"torch.float16"}:
            raise SystemExit(f"expected every tensor in float16, found {sorted(dtypes)}")
        print(f"  {variant:8s} writing {len(state)} tensors, float16 as published")
        torch.save({key: tensor.contiguous() for key, tensor in state.items()}, target)
        staged.unlink()
    with urllib.request.urlopen(resolve(repo, commit, "LICENSE"), timeout=60) as response:
        licence = response.read()
    if not licence.lstrip().startswith(b"MIT License"):
        raise SystemExit(f"{repo}@{commit} LICENSE is not the MIT licence; stop and look")
    (target.parent / "LICENSE").write_bytes(licence)
    (target.parent / "NOTICE").write_text(_notice(variant, sha))


def derive_vocab() -> None:
    """Rebuild the vocabulary asset: BART's 50,265 pieces plus the processor's 1,024 tokens.

    Committed rather than fetched, because it has to be in the wheel before a user runs anything.
    Every repository's ``tokenizer.json`` is checked to be the same file.
    """
    texts = set()
    for repo, commit, _ in SOURCES.values():
        with urllib.request.urlopen(resolve(repo, commit, "tokenizer.json"), timeout=300) as r:
            texts.add(r.read().decode("utf-8"))
    if len(texts) != 1:
        raise SystemExit("the repositories' tokenizer.json files differ; one asset cannot serve all")
    spec = json.loads(texts.pop())
    # An early tokenizers file: the BPE model carries no "type", but its merges say what it is.
    if "merges" not in spec["model"] or spec["decoder"]["type"] != "ByteLevel":
        raise SystemExit(f"unexpected tokenizer: {sorted(spec['model'])}, {spec['decoder']}")
    vocab = spec["model"]["vocab"]
    pieces = [piece for piece, _ in sorted(vocab.items(), key=lambda item: item[1])]
    if [vocab[piece] for piece in pieces] != list(range(len(pieces))) or len(pieces) != 50265:
        raise SystemExit("BART's vocabulary is not 50,265 contiguous ids")
    special = [token["id"] for token in spec["added_tokens"] if token["special"]]
    if sorted(pieces[i] for i in special) != sorted(["<s>", "<pad>", "</s>", "<unk>", "<mask>"]):
        raise SystemExit(f"unexpected special tokens: {[pieces[i] for i in special]}")
    special += list(range(len(pieces), len(pieces) + len(ADDED)))
    pieces += ADDED
    payload = json.dumps({"pieces": pieces, "special": sorted(special)},
                         ensure_ascii=False).encode()
    VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
    VOCAB_PATH.write_bytes(gzip.compress(payload, 9, mtime=0))
    print(f"{VOCAB_PATH.relative_to(ROOT)}  {VOCAB_PATH.stat().st_size:,} bytes, "
          f"{len(pieces):,} pieces, {len(special):,} special")
    print(f"  sha256 {digest(VOCAB_PATH)}")


def main() -> int:
    parser = variant_parser(__doc__, ROOT / "weights", revision=REVISION)
    parser.add_argument("--derive-vocab", action="store_true",
                        help="rebuild the vocabulary asset and exit")
    args = parser.parse_args()
    if args.derive_vocab:
        if args.variants:
            raise SystemExit("--derive-vocab takes no variants")
        derive_vocab()
        return 0
    wanted = args.variants or list(SPECS)
    unknown = [name for name in wanted if name not in SOURCES]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}. Known: {list(SOURCES)}")
    for variant in wanted:
        fetch(variant, args.revision, args.weights_dir)
    print(f"\n{len(wanted)} checkpoint(s) in {args.weights_dir}, all MIT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

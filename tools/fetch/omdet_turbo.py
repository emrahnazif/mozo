#!/usr/bin/env python3
"""Fetch OmDet-Turbo's published checkpoint into the local ``weights/`` tree.

Bootstrap tooling. It runs on a machine you control, never ships, and is the only place the
upstream URL lives.

    python tools/fetch/omdet_turbo.py                     # the one variant
    python tools/fetch/omdet_turbo.py --derive-lowercase  # rebuild the tokenizer's case table

**The download hash comes from Hugging Face's API**, read at fetch time from the commit pinned
below, so a mismatch means the bytes changed under us.

**Safetensors only, so mozo repacks.** Om AI Lab publishes ``model.safetensors`` and nothing torch
can read without the ``safetensors`` package. This reads it and writes the same tensors back out as
an ordinary ``.pth`` -- no tensor renamed, cast or dropped -- as ``tools/fetch/siglip2.py`` does.
``safetensors`` is a dependency of this script, not of mozo.

**One checkpoint, in two formats upstream.** ``omlab/omdet-turbo-swin-tiny-hf`` (transformers'
format, 1.01M all-time downloads on 2026-11-25) is the one taken. ``omlab/OmDet-Turbo_tiny_SWIN_T``
holds the same model in the authors' original format, plus CLIP's ``ViT-B-16.pt`` it was built
from, and reports no downloads.

The output is ``weights/omdet_turbo/tiny/<revision>/torch-fp32.pth`` and a NOTICE. The LICENSE
beside them is not written by this script: a licence is part of what is published, so it lives in
the weights tree like any other artifact.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))

sys.path.insert(0, str(ROOT))

from common import download_verified, hub_lfs_sha256s, require_licence, variant_parser  # noqa: E402

from mozo.vendors.omdet_turbo_deploy.config import VARIANTS  # noqa: E402

REPO = "omlab/omdet-turbo-swin-tiny-hf"

#: The Hub commit fetched, so a later upload to the repository cannot change what is published.
COMMIT = "7fe93cecfb770c4d76cf71163956221249cab566"

#: The revision this family was published under.
REVISION = "2026-11-25"

#: Om AI Lab ships no LICENSE file in the repository; its model card says ``apache-2.0``.
LICENCE_SOURCE = "https://www.apache.org/licenses/LICENSE-2.0.txt"

#: The tokenizer's lowercase table, committed to the repository because it has to be in the wheel.
LOWERCASE_ASSET = ROOT / "mozo" / "vendors" / "omdet_turbo_deploy" / "assets" / "lowercase.json.gz"



def _notice(sha256: str) -> str:
    """Attribution that travels with the weights, as Apache-2.0 asks."""
    return (
        "OmDet-Turbo -- tiny (Swin-T)\n"
        "real-time open-vocabulary object detection\n\n"
        "Copyright 2024 Om AI Lab.\n"
        "Tiancheng Zhao, Peng Liu, Xuan He, Lu Zhang, Kyusong Lee.\n\n"
        f"Source:  https://huggingface.co/{REPO}\n"
        f"         commit {COMMIT}\n"
        "Project: https://github.com/om-ai-lab/OmDet\n"
        "Paper:   Real-time Transformer-based Open-Vocabulary Detection with Efficient Fusion\n"
        "         Head, arXiv:2403.06892\n"
        "Licence: Apache-2.0, as stated on the model card (full text in the LICENSE file beside\n"
        "         this one)\n\n"
        "The checkpoint holds a Swin-T image backbone (Microsoft, MIT) and CLIP ViT-B/16's text\n"
        "tower (OpenAI, MIT), both fine-tuned into this detector by Om AI Lab and released by\n"
        "them under Apache-2.0.\n\n"
        "This file is not the byte stream the Hub serves: tools/fetch/omdet_turbo.py verifies\n"
        "model.safetensors against the sha256 the Hub records,\n"
        f"    {sha256}\n"
        "then writes the same tensors back out as an ordinary checkpoint. No tensor is altered,\n"
        "renamed, cast or dropped.\n"
    )


def fetch(revision: str, weights_dir: Path) -> None:
    """Download the checkpoint, verify it, repack it, and place it with its notice."""
    target = weights_dir / "omdet_turbo" / "tiny" / revision / "torch-fp32.pth"
    target.parent.mkdir(parents=True, exist_ok=True)
    source = target.with_name("model.safetensors")
    sha256 = download_verified(
        f"https://huggingface.co/{REPO}/resolve/{COMMIT}/model.safetensors", source,
        hub_lfs_sha256s(REPO, COMMIT)["model.safetensors"], label="tiny", width=6,
        detail="Apache-2.0")

    if not target.is_file():
        from safetensors.torch import load_file

        print("  tiny   repacking safetensors -> state dict")
        torch.save(load_file(source), target)
    source.unlink(missing_ok=True)

    (target.parent / "NOTICE").write_text(_notice(sha256))
    require_licence(target.parent, "Apache-2.0", LICENCE_SOURCE)


def derive_lowercase() -> None:
    """Rebuild the tokenizer's lowercase table from ``tokenizers``' own ``Lowercase`` normaliser.

    The reference lowercases in Rust, codepoint by codepoint, with that crate's Unicode tables.
    Python's ``str.lower()`` differs from it in two ways, both measured: it applies the Greek final
    sigma rule (``"ΑΣ".lower()`` is ``"ας"``, the reference gives ``"ασ"``), and its tables are the
    interpreter's Unicode version -- 55 single codepoints lowercase differently on Python 3.12, more
    on 3.10. Carrying the table makes a prompt's ids a property of the package, not of the Python
    running it. Needs ``tokenizers``, which nothing under ``mozo/`` imports.
    """
    from tokenizers import normalizers

    lowercase = normalizers.Lowercase()
    table = {}
    for codepoint in range(0x110000):
        if 0xD800 <= codepoint <= 0xDFFF:            # lone surrogates are not encodable text
            continue
        folded = lowercase.normalize_str(chr(codepoint))
        if folded != chr(codepoint):
            table[str(codepoint)] = folded
    payload = json.dumps(table, ensure_ascii=True, sort_keys=True).encode()
    # mtime=0 so rebuilding from the same tokenizers gives the same bytes.
    LOWERCASE_ASSET.write_bytes(gzip.compress(payload, mtime=0))
    print(f"{len(table)} codepoints -> {LOWERCASE_ASSET.relative_to(ROOT)}")


def main() -> int:
    parser = variant_parser(__doc__, ROOT / "weights", revision=REVISION)
    parser.add_argument("--derive-lowercase", action="store_true",
                        help="rebuild the tokenizer's lowercase table instead of fetching")
    args = parser.parse_args()
    if args.derive_lowercase:
        derive_lowercase()
        return 0
    unknown = [v for v in args.variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}. Known: {list(VARIANTS)}")
    fetch(args.revision, args.weights_dir)
    print(f"\n1 checkpoint in {args.weights_dir}, Apache-2.0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

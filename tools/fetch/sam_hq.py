#!/usr/bin/env python3
"""Fetch HQ-SAM's published checkpoints into the local ``weights/`` tree.

Bootstrap tooling. It runs on a machine you control, never ships, and is the only place the
upstream URLs live.

    python tools/fetch/sam_hq.py            # every variant
    python tools/fetch/sam_hq.py tiny base

**The files are placed unchanged.** HQ-SAM's authors publish plain PyTorch state dicts on Hugging
Face (``lkeab/hq-sam``, the mirror their README links for every size), and the vendored package
loads them as they are, so the published ``torch-fp32.pth`` is byte for byte the authors' file:
its sha256 is the one the Hub records. The hash is read from the Hub's API at a pinned commit, not
transcribed, so a mismatch means the bytes changed rather than that a digest was copied wrongly.

**Four of the five files are carried.** The fifth, ``sam2.1_hq_hiera_large.pt``, is HQ-SAM 2 --
the same idea on SAM 2.1 -- which needs a SAM 2 trunk this package does not build; see the
family's ``PROVENANCE.md`` for why it was left out.

Licensing: Apache-2.0 on the code (``SysCV/sam-hq``'s LICENSE) and on the checkpoints (the
``lkeab/hq-sam`` model card). The frozen parts inside them are Meta's SAM (Apache-2.0) and, for
``tiny``, MobileSAM's TinyViT (Apache-2.0). The authors' LICENSE text is placed beside each
revision; it is the Apache-2.0 text byte for byte.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from common import download_verified, hub_lfs_sha256s, require_licence, variant_parser  # noqa: E402

_HF = "https://huggingface.co"
REPOSITORY = "lkeab/hq-sam"

#: The Hub commit the hashes are read at. Pinned, so a re-upload is a decision rather than a drift.
HUB_COMMIT = "09b02a333b37772133eff3997bdba997867374b7"

#: The revision this family is published under.
REVISION = "2026-11-25"

#: The authors' licence file, at the code commit ``PROVENANCE.md`` records.
LICENCE_SOURCE = ("https://raw.githubusercontent.com/SysCV/sam-hq/"
                  "e696978d60352dc9a26b12631cd91781502c6546/LICENSE")

#: mozo's variant name -> the file in the repository. Upstream calls them by trunk.
FILES: dict[str, str] = {
    "tiny": "sam_hq_vit_tiny.pth",
    "base": "sam_hq_vit_b.pth",
    "large": "sam_hq_vit_l.pth",
    "huge": "sam_hq_vit_h.pth",
}

_TRUNKS = {
    "tiny": "MobileSAM's TinyViT image encoder (Apache-2.0, Kyung Hee University,\n"
            "         https://github.com/ChaoningZhang/MobileSAM), itself derived from\n"
            "         Microsoft's TinyViT (MIT/Apache-2.0)",
    "base": "Meta's SAM ViT-B (Apache-2.0, https://github.com/facebookresearch/segment-anything)",
    "large": "Meta's SAM ViT-L (Apache-2.0, https://github.com/facebookresearch/segment-anything)",
    "huge": "Meta's SAM ViT-H (Apache-2.0, https://github.com/facebookresearch/segment-anything)",
}


def _notice(variant: str, filename: str, sha256: str) -> str:
    """Attribution that travels with the weights, as Apache-2.0 asks."""
    return (
        f"HQ-SAM -- {variant}\n"
        "promptable segmentation in high quality: a click or a box in, a fine mask out\n\n"
        "Copyright (c) the HQ-SAM authors (ETH Zurich and HKUST).\n"
        "Lei Ke, Mingqiao Ye, Martin Danelljan, Yifan Liu, Yu-Wing Tai, Chi-Keung Tang, Fisher Yu.\n\n"
        f"Source:  {_HF}/{REPOSITORY}/blob/{HUB_COMMIT}/{filename}\n"
        "Project: https://github.com/SysCV/sam-hq\n"
        "Paper:   Segment Anything in High Quality (NeurIPS 2023), arXiv:2306.01567\n"
        "Licence: Apache-2.0 (full text in the LICENSE file beside this one)\n\n"
        "The checkpoint holds, frozen and unchanged, the image encoder, prompt encoder and mask\n"
        f"decoder of {_TRUNKS[variant]};\n"
        "and HQ-SAM's own trained parameters on top (the HQ output token, its MLP and the\n"
        "feature-fusion layers).\n\n"
        "This file is the authors' published checkpoint, unmodified:\n"
        f"         {filename}  sha256 {sha256}\n"
    )


def fetch(variant: str, revision: str, weights_dir: Path, hashes: dict[str, str]) -> None:
    """Download one variant's checkpoint, verify it, and place it with its notice."""
    filename = FILES[variant]
    target = weights_dir / "sam_hq" / variant / revision / "torch-fp32.pth"
    sha256 = download_verified(f"{_HF}/{REPOSITORY}/resolve/{HUB_COMMIT}/{filename}", target,
                               hashes[filename], label=variant, width=6, timeout=1800,
                               detail="Apache-2.0")
    (target.parent / "NOTICE").write_text(_notice(variant, filename, sha256))
    require_licence(target.parent, "Apache-2.0", LICENCE_SOURCE)


def main() -> int:
    args = variant_parser(__doc__, ROOT / "weights", revision=REVISION).parse_args()
    wanted = args.variants or list(FILES)
    unknown = [v for v in wanted if v not in FILES]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}. Known: {list(FILES)}")

    hashes = hub_lfs_sha256s(REPOSITORY, HUB_COMMIT)
    for variant in wanted:
        fetch(variant, args.revision, args.weights_dir, hashes)
    print(f"\n{len(wanted)} checkpoints in {args.weights_dir}, all Apache-2.0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

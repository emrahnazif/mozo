#!/usr/bin/env python3
"""Fetch Mayaku's street checkpoints into the local ``weights/`` tree, with their terms and labels.

    python tools/fetch/mayaku_street.py                    # all four
    python tools/fetch/mayaku_street.py nano seg-nano

Bootstrap tooling; never ships. Each checkpoint is the publisher's ``.pth``, byte for byte, taken
from its Hugging Face repository at a pinned commit and checked against the sha256 the Hub records
for that LFS object -- read from the Hub's API at run time, not transcribed. The same bytes are what
Mayaku's own manifest (``dtmfiles.com/mayaku/v1/models/manifest.json``, revision ``2026-11-23``)
serves; that was checked once by hand on 2026-11-24 and is recorded in the vendor's PROVENANCE.

Beside each checkpoint it writes:

- ``LICENSE``: Mayaku's Apache-2.0 text at the ``v3.1.1`` tag. The weights are Apache-2.0 by the
  publisher's statement on every model card ("The weights are Apache 2.0, like the library").
- ``NOTICE``: attribution, provenance, and what is known about the training labels.
- ``labels.json``: the 22 class names, read out of the checkpoint's own record, for the graph
  runtimes, which carry no vocabulary.

The output is ``weights/mayaku_street/<variant>/<revision>/``. Run ``tools/export/mayaku_street.py``
next for the graph artifacts, then ``tools/generate_manifest.py``.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from common import download_verified, hub_lfs_sha256s, variant_parser  # noqa: E402

FAMILY = "mayaku_street"
REVISION = "2026-11-24"

#: mozo variant -> (Hugging Face repo, the commit pinned, the file in it). Pinned so a later push
#: to ``main`` cannot change what this publishes without someone changing this table.
CHECKPOINTS: dict[str, tuple[str, str, str]] = {
    "nano": ("enazif/mayaku-n-det-street", "98b3641999644e038ab75c5192f46391d1527652",
             "mayaku-n-det-street.pth"),
    "medium": ("enazif/mayaku-m-det-street", "735857563f708beaa0c5c9a73b2a3daad4137218",
               "mayaku-m-det-street.pth"),
    "seg-nano": ("enazif/mayaku-n-seg-street", "771d212792d3bbb529cf430675db6cd3b32d610b",
                 "mayaku-n-seg-street.pth"),
    "seg-medium": ("enazif/mayaku-m-seg-street", "2c6105f29d4eac9ebe414fccb584631e855dc5fb",
                   "mayaku-m-seg-street.pth"),
}

#: The licence text, from the release the vendor was extracted from.
LICENCE_URL = "https://raw.githubusercontent.com/emrahnazif/mayaku/v3.1.1/LICENSE"


def notice(variant: str, repo: str, commit: str, filename: str) -> str:
    return (
        f"Mayaku street -- {variant}\n\n"
        "Copyright 2026 the mayaku authors.\n\n"
        f"Source:  https://huggingface.co/{repo}/blob/{commit}/{filename}\n"
        "Project: https://github.com/emrahnazif/mayaku (pip install mayaku)\n"
        "Licence: Apache-2.0 (full text in the LICENSE file beside this one)\n\n"
        "torch-fp32.pth is the publisher's checkpoint, byte for byte. The weights are Apache-2.0\n"
        "by the publisher's own statement on the model card: \"The weights are Apache 2.0, like\n"
        "the library.\" Any graph in this directory (ONNX or otherwise) is exported from these\n"
        "weights and is covered by the same licence.\n\n"
        "TRAINING DATA\n"
        "-------------\n"
        "Images from Objects365, COCO, Wikimedia Commons (under their individual free licences),\n"
        "the Songdo Vision dataset (Fonod, Cho, Yeo and Geroliminis, 2025), roundabout aerial\n"
        "images by De-Las-Heras, Sanchez-Soriano and Puertas, and aerial photos from\n"
        "Beeldmateriaal Nederland / PDOK, as the publisher lists them.\n\n"
        "The training labels were made with Meta's SAM 3. SAM 3 itself is under the SAM License,\n"
        "which covers the \"SAM Materials\" Meta distributes (models, code, weights,\n"
        "documentation). Its text mentions outputs only in its warranty disclaimer and its\n"
        "litigation clause, and states no term covering models trained on those outputs. This\n"
        "is recorded as found, not as legal advice.\n"
    )


def fetch(variant: str, revision: str, weights_dir: Path, licence: bytes) -> None:
    """Download one variant, verify it against the Hub, and place its terms and labels."""
    repo, commit, filename = CHECKPOINTS[variant]
    expected = hub_lfs_sha256s(repo, commit)[filename]
    target = weights_dir / FAMILY / variant / revision / "torch-fp32.pth"
    download_verified(f"https://huggingface.co/{repo}/resolve/{commit}/{filename}", target,
                      expected, label=variant, width=10, timeout=120)

    (target.parent / "LICENSE").write_bytes(licence)
    (target.parent / "NOTICE").write_text(notice(variant, repo, commit, filename))

    # Read through the vendor, which refuses a checkpoint that is not the variant it claims to be,
    # so the names published are those of a model that loaded as this variant.
    from mozo.vendors.mayaku_street_deploy import Detector

    names = Detector(target, variant).names
    (target.parent / "labels.json").write_text(
        json.dumps([{"id": i, "name": name} for i, name in enumerate(names)]))
    print(f"  {variant:<10} {len(names)} classes, LICENSE, NOTICE")


def main() -> int:
    args = variant_parser(__doc__, ROOT / "weights", revision=REVISION).parse_args()
    wanted = args.variants or list(CHECKPOINTS)
    unknown = [v for v in wanted if v not in CHECKPOINTS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}. Known: {list(CHECKPOINTS)}")

    with urllib.request.urlopen(LICENCE_URL, timeout=60) as response:
        licence = response.read()
    if b"Apache License" not in licence[:200] or b"Version 2.0" not in licence[:200]:
        raise SystemExit(f"{LICENCE_URL} is not the Apache-2.0 text; stop and look")

    for variant in wanted:
        fetch(variant, args.revision, args.weights_dir, licence)
    print(f"\n{len(wanted)} checkpoints in {args.weights_dir / FAMILY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

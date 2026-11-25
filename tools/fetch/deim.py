#!/usr/bin/env python3
"""Fetch DEIM's published checkpoints into the local ``weights/`` tree, with their terms and labels.

    python tools/fetch/deim.py                    # all six
    python tools/fetch/deim.py nano xlarge-obj365

Bootstrap tooling; never ships. Each checkpoint is the authors' ``.pth``, byte for byte, from the
Google Drive file their README links (``github.com/ShihuaHuang95/DEIM`` at the pinned commit
below). Google Drive records no hash a script can read without an API key, and the authors
publish none, so each file is checked against the sha256 mozo measured when it first downloaded
them on 2026-11-25 (twice, through two different download paths, with identical bytes). A mismatch
means the file behind the link changed; do not publish through it until that is understood.

Beside each checkpoint it writes:

- ``LICENSE`` is placed by hand, as for every family (``common.require_licence`` says how): the
  repository's Apache-2.0 text at the pinned commit. The checkpoints carry no terms of their own;
  ``NOTICE`` says that the licence is inferred from the repository's, and says it as an inference.
- ``NOTICE``: attribution, provenance, training data.
- ``labels.json``: COCO's 80 names, in the contiguous order the model's class ids use.

The output is ``weights/deim/<variant>/<revision>/``. Run ``tools/export/deim.py`` next for the
graph artifacts, then ``tools/generate_manifest.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from common import download_verified, require_licence, variant_parser  # noqa: E402

FAMILY = "deim"
REVISION = "2026-11-25"

#: The upstream commit the vendor was extracted from and the README links were read at.
COMMIT = "09d35d53d39ee3145a1e61e3a989b28b9468d1dd"

#: mozo variant -> (Google Drive file id, upstream's file name, sha256 measured on 2026-11-25).
CHECKPOINTS: dict[str, tuple[str, str, str]] = {
    "nano": ("1ZPEhiU9nhW4M5jLnYOFwTSLQC1Ugf62e", "deim_dfine_hgnetv2_n_coco_160e.pth",
             "8536a97a57bad2951cb4679e8578be225e4a19db678ab568a44e538f4f0fc917"),
    "small": ("1tB8gVJNrfb6dhFvoHJECKOF5VpkthhfC", "deim_dfine_hgnetv2_s_coco_120e.pth",
              "a089c9eb798f8a0322e05f757fe8e8188eee34aa578788a35ec253bcec8a6b38"),
    "medium": ("18Lj2a6UN6k_n_UzqnJyiaiLGpDzQQit8", "deim_dfine_hgnetv2_m_coco_90e.pth",
               "2b6cd0582a4aa711f583982057b7fb0f3daebdd98e4dc168824714014c3219bc"),
    "large": ("1PIRf02XkrA2xAD3wEiKE2FaamZgSGTAr", "deim_dfine_hgnetv2_l_coco_50e.pth",
              "a9ac1f9bf51a41a7ec02c373c85e0109f84908dde11d80759b43248dcb638baa"),
    "xlarge": ("1dPtbgtGgq1Oa7k_LgH1GXPelg1IVeu0j", "deim_dfine_hgnetv2_x_coco_50e.pth",
               "e729436f96d95e78dbf67452bf3fb654cd5bed37de43067f5cf0aafb24d0fc7a"),
    "xlarge-obj365": ("1RMNrHh3bYN0FfT5ZlWhXtQxkG23xb2xj", "deim_dfine_hgnetv2_x_obj2coco_24e.pth",
                      "f240a1128280eb6ce0fff911e8811a8dff669e18be25f5160db7b4263452277d"),
}

#: The licence text, from the commit the vendor was extracted from.
LICENCE_URL = f"https://raw.githubusercontent.com/ShihuaHuang95/DEIM/{COMMIT}/LICENSE"

#: COCO's 80 names in the order of upstream's ``mscoco_category2name`` (``engine/data/dataset/
#: coco_dataset.py``), which is the order of the model's contiguous class ids: id ``i`` is the
#: ``i``-th category there.
COCO = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
)


def notice(variant: str, drive_id: str, filename: str) -> str:
    obj365 = variant == "xlarge-obj365"
    data = ("Pretrained on Objects365, then fine-tuned for 24 epochs on COCO 2017 train." if obj365
            else "Trained on COCO 2017 train.")
    return (
        f"DEIM (DEIM-D-FINE) -- {variant}\n\n"
        "Copyright (C) INTELLINDUST INFORMATION TECHNOLOGY (SHENZHEN) CO., LTD. and all its\n"
        "affiliates. DEIM builds on D-FINE (Copyright (c) 2024 D-FINE Authors) and RT-DETR\n"
        "(Copyright (c) 2023 lyuwenyu), both Apache-2.0.\n\n"
        f"Source:  https://drive.google.com/file/d/{drive_id} ({filename}), linked from\n"
        f"         https://github.com/ShihuaHuang95/DEIM/blob/{COMMIT}/README.md\n"
        "Paper:   DEIM: DETR with Improved Matching for Fast Convergence (CVPR 2025),\n"
        "         https://arxiv.org/abs/2412.04234\n"
        "Licence: Apache-2.0 (full text in the LICENSE file beside this one)\n\n"
        "torch-fp32.pth is the authors' checkpoint, byte for byte. The repository is licensed\n"
        "Apache-2.0 and publishes these checkpoints from its README; neither the repository nor\n"
        "the checkpoints state separate terms for the weights. Apache-2.0 for the weights is\n"
        "therefore an inference from the repository's licence, not a statement the authors made\n"
        "about the weights. Any graph in this directory (ONNX or otherwise) is exported from these\n"
        "weights and is covered by the same terms.\n\n"
        f"TRAINING DATA\n-------------\n{data}\n"
    )


def fetch(variant: str, revision: str, weights_dir: Path) -> None:
    """Download one variant, verify it, and place its terms and labels."""
    drive_id, filename, expected = CHECKPOINTS[variant]
    target = weights_dir / FAMILY / variant / revision / "torch-fp32.pth"
    url = f"https://drive.usercontent.google.com/download?id={drive_id}&export=download&confirm=t"
    download_verified(url, target, expected, label=variant, width=14, timeout=120)

    require_licence(target.parent, "Apache-2.0", LICENCE_URL)
    (target.parent / "NOTICE").write_text(notice(variant, drive_id, filename))
    (target.parent / "labels.json").write_text(
        json.dumps([{"id": i, "name": name} for i, name in enumerate(COCO)]))

    # Loaded through the vendor, which refuses a checkpoint whose keys, shapes or dtypes are not
    # this variant's, so what is published is known to build as the variant it is filed under.
    from mozo.vendors.deim_deploy import load

    load(variant, target)
    print(f"  {variant:<14} loads strictly, LICENSE, NOTICE, labels")


def main() -> int:
    args = variant_parser(__doc__, ROOT / "weights", revision=REVISION).parse_args()
    wanted = args.variants or list(CHECKPOINTS)
    unknown = [v for v in wanted if v not in CHECKPOINTS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}. Known: {list(CHECKPOINTS)}")

    for variant in wanted:
        fetch(variant, args.revision, args.weights_dir)
    print(f"\n{len(wanted)} checkpoints in {args.weights_dir / FAMILY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

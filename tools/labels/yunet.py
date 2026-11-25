#!/usr/bin/env python3
"""Write the vocabulary for each YuNet variant into the local ``weights/`` tree.

Bootstrap tooling; never ships.

    python tools/labels/yunet.py

One class, ``face`` at id 0 -- the head scores a single class (upstream's checkpoints call it
``FG``), and 0 is the index it emits -- and the five landmarks in the order the landmark channels
come out in: right eye, left eye, nose tip, right mouth corner, left mouth corner. "Right" is the
subject's, so on a frontal face the right eye is the one on the left of the picture. The order and
the names are OpenCV's (``FaceDetectorYN``'s documentation names them ``re, le, nt, rcm, lcm``),
and the SCRFD/RetinaFace WIDER FACE annotations YuNet was trained on put them in the same order.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from common import variant_parser  # noqa: E402

#: The landmark channels' order, as named in the published labels.
KEYPOINT_NAMES = ["right_eye", "left_eye", "nose_tip", "right_mouth_corner", "left_mouth_corner"]


def keypoint_labels() -> list[dict]:
    """The one category the head predicts, with its landmarks, in PixelFlow's rich label format."""
    return [{"id": 0, "name": "face", "keypoints": [{"name": name} for name in KEYPOINT_NAMES]}]


def write(variant: str, labels: list[dict], weights_dir: Path) -> None:
    """Write ``labels.json`` into every fetched revision of *variant*."""
    variant_dir = weights_dir / "yunet" / variant
    revisions = sorted(d for d in variant_dir.iterdir() if d.is_dir()) if variant_dir.is_dir() else []
    if not revisions:
        print(f"  {variant:<5} skipped, not fetched")
        return
    for directory in revisions:
        (directory / "labels.json").write_text(json.dumps(labels, indent=2) + "\n")
    print(f"  {variant:<5} face + 5 landmarks in {', '.join(d.name for d in revisions)}")


def main() -> int:
    from tools.fetch.yunet import CHECKPOINTS

    args = variant_parser(__doc__, ROOT / "weights").parse_args()
    labels = keypoint_labels()
    for variant in args.variants or list(CHECKPOINTS):
        write(variant, labels, args.weights_dir)
    print("\nRun tools/generate_manifest.py to pick these up.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

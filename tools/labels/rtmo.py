#!/usr/bin/env python3
"""Write the vocabulary for each RTMO variant into the local ``weights/`` tree.

Bootstrap tooling; never ships.

    python tools/labels/rtmo.py

One class, ``person`` at id 0 -- the head scores a single class, and 0 is the index it emits -- and
COCO's seventeen joint names, in COCO's order, which is the order the keypoint channels come out
in. The same names ``tools/labels/vitpose.py`` and ``tools/labels/rfdetr.py`` publish, so a
skeleton is labelled the same whichever family drew it.

The order is not assumed: ``tools/fetch/rtmo.py`` holds this list against each checkpoint's own
``dataset_meta.keypoint_id2name`` when it repacks it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from common import variant_parser  # noqa: E402

# COCO's 17 person keypoints, in the dataset's order: the one list ``tools/labels/vitpose.py``
# keeps, so the families that name COCO's joints cannot drift apart.
from tools.labels.vitpose import KEYPOINT_NAMES  # noqa: E402


def keypoint_labels() -> list[dict]:
    """The one category the head predicts, with its joints, in PixelFlow's rich label format."""
    return [{"id": 0, "name": "person",
             "keypoints": [{"name": name} for name in KEYPOINT_NAMES]}]


def write(variant: str, labels: list[dict], weights_dir: Path) -> None:
    """Write ``labels.json`` into every fetched revision of *variant*."""
    variant_dir = weights_dir / "rtmo" / variant
    revisions = sorted(d for d in variant_dir.iterdir() if d.is_dir()) if variant_dir.is_dir() else []
    if not revisions:
        print(f"  {variant:<6} skipped, not fetched")
        return
    for directory in revisions:
        (directory / "labels.json").write_text(json.dumps(labels, indent=2) + "\n")
    print(f"  {variant:<6} person + 17 joints in {', '.join(d.name for d in revisions)}")


def main() -> int:
    from tools.fetch.rtmo import CHECKPOINTS

    args = variant_parser(__doc__, ROOT / "weights").parse_args()
    labels = keypoint_labels()
    for variant in args.variants or list(CHECKPOINTS):
        write(variant, labels, args.weights_dir)
    print("\nRun tools/generate_manifest.py to pick these up.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

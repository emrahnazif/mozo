#!/usr/bin/env python3
"""Fetch RTMO's published checkpoints into the local ``weights/`` tree.

Bootstrap tooling. It runs on a machine you control, never ships, and is the only place the
upstream URLs live.

    python tools/fetch/rtmo.py              # every variant
    python tools/fetch/rtmo.py small large

**The hash.** OpenMMLab names each checkpoint with the first eight hex digits of its sha256 --
that prefix is the only digest the publisher states. The full sha256 below was measured on the
files as served on 2026-11-25, and each one starts with the publisher's prefix; this script checks
both, so a re-released file fails even where its new hash happened to share the prefix.

**mozo repacks.** The upstream file is ``{"meta", "state_dict"}``, and ``meta`` pickles numpy
arrays and the training config, so it only loads with ``weights_only=False`` -- arbitrary code on
load. It is unpickled here, once, after its hash has been verified, and only the ``state_dict`` is
written back out: no tensor is altered, renamed, cast or dropped, and the result loads with
``weights_only=True``. The joint names in ``meta`` are checked against
``tools/labels/rtmo.py``'s list on the way.

**Four checkpoints of eleven.** The body7 tiny/small/medium/large. The COCO-only s/m/l are the same
architecture and 0.9-1.7 AP worse at every size; the CrowdPose ones predict 14 joints. See
``mozo/vendors/rtmo_deploy/PROVENANCE.md``.

Licensing: MMPose is Apache-2.0 and states no separate terms for its weights; mozo infers the
weights' licence from the repository's and the NOTICE says it is an inference.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from common import download_verified, require_licence, variant_parser  # noqa: E402

_BASE = "https://download.openmmlab.com/mmpose/v1/projects/rtmo"

#: The revision this family was published under.
REVISION = "2026-11-25"

#: MMPose's own licence text, at the release the vendor was extracted from.
LICENCE_SOURCE = "https://raw.githubusercontent.com/open-mmlab/mmpose/v1.3.2/LICENSE"

#: mozo's variant -> (upstream file, its full sha256, COCO val2017 AP the model zoo states).
CHECKPOINTS: dict[str, tuple[str, str, str]] = {
    "tiny": ("rtmo-t_8xb32-600e_body7-416x416-f48f75cb_20231219.pth",
             "f48f75cbd59d7026b1d18378ef3e408c80f25b50144ba07deeebc4dc5e54714c", "57.4"),
    "small": ("rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.pth",
              "dac2bf749bbfb51e69ca577ca0327dff4433e3be9a56b782f0b7ef94fb45247e", "68.6"),
    "medium": ("rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.pth",
               "39e78cc4afed7d1dd31ea73ee04842f7d8f8e695d900d41b90746e94728a01c3", "72.6"),
    "large": ("rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.pth",
              "b37118cee39b43531c23d9e443fbb4f0b8987f590d9c6ed61978d12ecfce84df", "74.8"),
}


def publisher_prefix(filename: str) -> str:
    """The sha256 prefix OpenMMLab writes into the filename: ``...-<8 hex>_<date>.pth``."""
    return filename.rsplit("_", 1)[0].rsplit("-", 1)[1]


def _notice(variant: str, filename: str, sha256: str, ap: str) -> str:
    """Attribution that travels with the weights, as Apache-2.0 asks."""
    return (
        f"RTMO -- {variant}\n"
        "one-stage multi-person pose estimation: a frame in, people with boxes and\n"
        "COCO's 17 joints out\n\n"
        "Copyright (c) OpenMMLab. All rights reserved.\n"
        "Peng Lu, Tao Jiang, Yining Li, Xiangtai Li, Kai Chen, Wenming Yang.\n\n"
        f"Source:  {_BASE}/{filename}\n"
        "Project: https://github.com/open-mmlab/mmpose (projects/rtmo, v1.3.2)\n"
        "Paper:   RTMO: Towards High-Performance One-Stage Real-Time Multi-Person Pose\n"
        "         Estimation (CVPR 2024), arXiv:2312.07526\n"
        f"COCO val2017 AP stated by the model zoo: {ap}\n"
        "Licence: Apache-2.0 (full text in the LICENSE file beside this one)\n\n"
        "MMPose is released under the Apache License 2.0. Its model zoo states no separate\n"
        "terms for the checkpoints, so mozo takes them to be under the repository's licence.\n"
        "That is an inference from the absence of other terms, not a statement the authors make\n"
        "about the weights.\n\n"
        "Trained on body7: COCO, AI Challenger, CrowdPose, MPII, sub-JHMDB, Halpe and PoseTrack18.\n"
        "Those datasets carry terms of their own, which govern the datasets.\n\n"
        "This file is not the byte stream OpenMMLab serves. tools/fetch/rtmo.py verifies that\n"
        "file's sha256, unpickles it, and writes its state_dict back out alone -- dropping the\n"
        "pickled training metadata so the result loads with weights_only=True. No tensor is\n"
        "altered, renamed, cast or dropped. The source file is:\n"
        f"           {filename}  sha256 {sha256}\n"
    )


def _check_joints(meta: dict) -> None:
    """Hold the checkpoint's own joint names against the published labels, position by position."""
    from tools.labels.rtmo import KEYPOINT_NAMES

    declared = meta["dataset_meta"]["keypoint_id2name"]
    theirs = [declared[index] for index in range(len(declared))]
    if theirs != KEYPOINT_NAMES:
        raise SystemExit(f"checkpoint names its joints {theirs}, labels say {KEYPOINT_NAMES}")


def fetch(variant: str, revision: str, weights_dir: Path) -> None:
    """Download one variant's checkpoint, verify it, repack it, and place it with its notice."""
    filename, sha256, ap = CHECKPOINTS[variant]
    if not sha256.startswith(publisher_prefix(filename)):
        raise SystemExit(f"{variant}: pinned sha256 does not start with the publisher's prefix")
    target = weights_dir / "rtmo" / variant / revision / "torch-fp32.pth"
    target.parent.mkdir(parents=True, exist_ok=True)

    part = target.with_name(filename)
    download_verified(f"{_BASE}/{filename}", part, sha256, label=variant, width=6,
                      detail="Apache-2.0")
    if not target.is_file():
        # Safe only because the bytes were just verified against the pinned digest.
        checkpoint = torch.load(part, map_location="cpu", weights_only=False)
        _check_joints(checkpoint["meta"])
        torch.save(dict(checkpoint["state_dict"]), target)
        print(f"  {variant:6s} repacked: state_dict only, {len(checkpoint['state_dict'])} tensors")
    part.unlink(missing_ok=True)

    (target.parent / "NOTICE").write_text(_notice(variant, filename, sha256, ap))
    require_licence(target.parent, "Apache-2.0", LICENCE_SOURCE)


def main() -> int:
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

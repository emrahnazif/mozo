#!/usr/bin/env python3
"""Fetch YuNet's published checkpoints into the local ``weights/`` tree.

Bootstrap tooling. It runs on a machine you control, never ships, and is the only place the
upstream URLs live.

    python tools/fetch/yunet.py              # both variants
    python tools/fetch/yunet.py small

**The source.** The authors' training repository, ShiqiYu/libfacedetection.train, at the commit
the vendor was extracted from. Its ``weights/`` folder holds exactly two checkpoints; ``yunet_s``
is the network OpenCV Zoo distributes as ``face_detection_yunet_2023mar.onnx`` (that file is
byte-identical to this repository's ``onnx/yunet_s_640_640.onnx``, committed with it).

**The hash.** The files are plain git blobs, not LFS, so the digest the publisher records is the
git blob id, which this script reads from GitHub's contents API at the pinned commit and checks
against the bytes. The full sha256 measured on 2026-11-25 is pinned too and checked as they
download, so a force-pushed history fails even where the API agrees with itself.

**mozo repacks.** The upstream file is ``{"meta", "state_dict", "optimizer"}`` -- the SGD momentum
buffers and the mmdetection training config travel with the weights. Only ``state_dict`` is
written back out: every tensor, ``num_batches_tracked`` included, no tensor altered, renamed,
cast or dropped.

Licensing: the repository's LICENSE is BSD-3-Clause at the pinned commit and states no separate
terms for the weights, which are files of the repository. See the NOTICE this writes.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from common import download_verified, hub_json, require_licence, variant_parser  # noqa: E402

REPO = "ShiqiYu/libfacedetection.train"
COMMIT = "dca340aa082c71081a68d17db8e58b33a58a914b"

#: The revision this family was published under.
REVISION = "2026-11-25"

#: The repository's own licence text, at the pinned commit.
LICENCE_SOURCE = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/LICENSE"

#: mozo's variant -> (upstream file, its git blob id, its full sha256, upstream's experiment name).
CHECKPOINTS: dict[str, tuple[str, str, str, str]] = {
    "small": ("yunet_s.pth", "2c709a076b64ae79dccf468fad6f59c61bb5ab6a",
              "f2688fecd5aef7d204d04ca12ff377b724978abbe5955740bc67280d99ffe4d6",
              "yunet_yolox_shared_stack_1_step_nophoto_diou_nonorm_layerch32"),
    "base": ("yunet_n.pth", "441b9440a911fd79593e1194ba2528e1791523f6",
             "9c7c9e14a4d60e57491d77e4d8b0d451bd834f300e05c0dea7fd1f6533817b06",
             "normal_0.5_1.5"),
}


def git_blob_id(path: Path) -> str:
    """The id git gives *path*'s bytes: sha1 over ``blob <size>\\0`` and the content."""
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def published_blob_ids() -> dict[str, str]:
    """File name -> git blob id, for ``weights/`` at the pinned commit, as GitHub states them."""
    listing = hub_json(f"https://api.github.com/repos/{REPO}/contents/weights?ref={COMMIT}")
    return {entry["name"]: entry["sha"] for entry in listing}


def _notice(variant: str, filename: str, sha256: str, experiment: str) -> str:
    """Attribution that travels with the weights, as BSD-3-Clause asks."""
    return (
        f"YuNet -- {variant}\n"
        "a tiny face detector: an image in, faces with five landmarks out\n\n"
        "Copyright (c) 2022-2026, Shiqi Yu <shiqi.yu@gmail.com>. All rights reserved.\n"
        "Wei Wu, Hanyang Peng, Shiqi Yu.\n\n"
        f"Source:  https://github.com/{REPO}/blob/{COMMIT}/weights/{filename}\n"
        f"Project: https://github.com/{REPO} (commit {COMMIT[:8]})\n"
        "Paper:   YuNet: A Tiny Millisecond-level Face Detector, Machine Intelligence Research\n"
        "         20(5), 2023, doi:10.1007/s11633-023-1423-y\n"
        f"Upstream name: {filename[:-4]} (training run {experiment})\n"
        "Licence: BSD-3-Clause (full text in the LICENSE file beside this one)\n\n"
        "The repository states no separate terms for its checkpoints, which are files in it:\n"
        "they are under its LICENSE, BSD-3-Clause at the commit above. When they were committed\n"
        "(2022-12 and 2023-01) the repository's LICENSE was Apache-2.0. OpenCV Zoo also\n"
        "distributes the yunet_s network, as face_detection_yunet_2023mar.onnx, under the MIT\n"
        "License (github.com/opencv/opencv_zoo, models/face_detection_yunet/LICENSE).\n\n"
        "Trained on WIDER FACE (with the SCRFD landmark annotations). The dataset carries terms\n"
        "of its own (non-commercial research, CC BY-NC-ND 4.0 on its Hugging Face card), which\n"
        "govern the dataset.\n\n"
        "This file is not the byte stream the repository serves. tools/fetch/yunet.py verifies\n"
        "that file's git blob id and sha256, and writes its state_dict back out alone -- dropping\n"
        "the optimizer state and the training metadata. No tensor is altered, renamed, cast or\n"
        "dropped. The source file is:\n"
        f"           {filename}  sha256 {sha256}\n"
    )


def fetch(variant: str, revision: str, weights_dir: Path, blob_ids: dict[str, str]) -> None:
    """Download one variant's checkpoint, verify it, repack it, and place it with its notice."""
    filename, blob, sha256, experiment = CHECKPOINTS[variant]
    if blob_ids.get(filename) != blob:
        raise SystemExit(f"{variant}: GitHub states blob {blob_ids.get(filename)} for {filename} "
                         f"at {COMMIT[:8]}, pinned {blob}")
    target = weights_dir / "yunet" / variant / revision / "torch-fp32.pth"
    target.parent.mkdir(parents=True, exist_ok=True)

    part = target.with_name(filename)
    url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/weights/{filename}"
    download_verified(url, part, sha256, label=variant, width=5, detail="BSD-3-Clause")
    if git_blob_id(part) != blob:
        raise SystemExit(f"{variant}: the downloaded bytes are not git blob {blob}")
    if not target.is_file():
        checkpoint = torch.load(part, map_location="cpu", weights_only=True)
        torch.save(dict(checkpoint["state_dict"]), target)
        print(f"  {variant:5s} repacked: state_dict only, {len(checkpoint['state_dict'])} tensors")
    part.unlink(missing_ok=True)

    (target.parent / "NOTICE").write_text(_notice(variant, filename, sha256, experiment))
    require_licence(target.parent, "BSD-3-Clause", LICENCE_SOURCE)


def main() -> int:
    args = variant_parser(__doc__, ROOT / "weights", revision=REVISION).parse_args()
    wanted = args.variants or list(CHECKPOINTS)
    unknown = [v for v in wanted if v not in CHECKPOINTS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}. Known: {list(CHECKPOINTS)}")
    blob_ids = published_blob_ids()
    for variant in wanted:
        fetch(variant, args.revision, args.weights_dir, blob_ids)
    print(f"\n{len(wanted)} checkpoints in {args.weights_dir}, all BSD-3-Clause")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

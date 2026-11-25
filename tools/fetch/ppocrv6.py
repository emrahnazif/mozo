#!/usr/bin/env python3
"""Fetch PaddlePaddle's PP-OCRv6 checkpoints and fuse each variant into one mozo checkpoint.

Bootstrap tooling. It runs on a machine you control, never ships, and is the only place the
upstream URLs live.

    python tools/fetch/ppocrv6.py                  # medium, small, tiny
    python tools/fetch/ppocrv6.py medium

**The PyTorch checkpoints, not the Paddle ones.** PaddlePaddle publishes every PP-OCRv6 network
three ways: a Paddle inference graph (``PP-OCRv6_<size>_<det|rec>``), an ONNX export of it
(``..._onnx``), and a ``model.safetensors`` for its own PyTorch implementation in transformers
(``..._safetensors``), which PaddleOCR runs with ``engine="transformers"``. mozo is torch, so it
takes the last: those tensors load into the vendored modules with no conversion at all. For the
tiny recogniser the safetensors checkpoint is a different training snapshot from the Paddle
graph; see the vendor's PROVENANCE.md.

**Two repositories in, one file out.** Each ``torch-fp32.pth`` is
``{"detector": state_dict, "recogniser": state_dict}``, tensors exactly as published.

**Verified against the publisher's hash.** The Hub records an sha256 for every LFS file; the
commits below pin which ones.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from common import download_verified, hub_lfs_sha256s, require_licence, variant_parser  # noqa: E402

from mozo.vendors.ppocrv6_deploy.checkpoint import DETECTOR, RECOGNISER, load  # noqa: E402
from mozo.vendors.ppocrv6_deploy.config import SPECS, VARIANTS  # noqa: E402

#: The revision this family is published under.
REVISION = "2026-11-25"

#: PaddleOCR's own LICENSE: Apache-2.0, the terms PaddlePaddle states on every checkpoint's card.
LICENCE_SOURCE = "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/LICENSE"

#: Hub commit of each repository, checked 2026-11-25.
COMMITS = {
    ("medium", "det"): "4236c2b61741a259c091fd879dcc4edc339e916c",
    ("medium", "rec"): "024cad6a831de75c2c3c26e711ba8c4a82ccd24b",
    ("small", "det"): "eae2ee920a39fb3087637d3dbb58df1896ec1f24",
    ("small", "rec"): "fe049fb103f57443fe8840c54ed06b702f3c1de5",
    ("tiny", "det"): "07595f982703daf0d4e120a12a01da8073542f3a",
    ("tiny", "rec"): "6f2d2d51b4b4226d7a2329a02f416f4994106f3a",
}

NOTICE = """\
PP-OCRv6 -- {variant}
{rule}

Text detection and recognition, published by PaddlePaddle (Baidu).

Code:    https://github.com/PaddlePaddle/PaddleOCR  (Apache License 2.0)
Weights: https://huggingface.co/PaddlePaddle, each model card "license: apache-2.0"
Paper:   PP-OCRv6: From 1.5M to 34.5M Parameters, Surpassing Billion-Scale VLMs on OCR Tasks,
         arXiv:2606.13108

This artifact fuses two of PaddlePaddle's published checkpoints into one file:

  detector    {det_repo}@{det_commit}  model.safetensors
  recogniser  {rec_repo}@{rec_commit}  model.safetensors

Neither network is modified. The tensors are PaddlePaddle's, byte for byte; only the two state
dictionaries' arrangement into a single file is mozo's.
"""


def _repo(variant: str, kind: str) -> str:
    return f"PaddlePaddle/PP-OCRv6_{variant}_{kind}_safetensors"


def _download(variant: str, kind: str, staging: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    repo, commit = _repo(variant, kind), COMMITS[(variant, kind)]
    expected = hub_lfs_sha256s(repo, commit)["model.safetensors"]
    target = staging / f"{variant}_{kind}.safetensors"
    download_verified(f"https://huggingface.co/{repo}/resolve/{commit}/model.safetensors",
                      target, expected, label=f"{variant} {kind}", width=10)
    return load_file(target)


def main() -> int:
    parser = variant_parser(__doc__, ROOT / "weights", revision=REVISION)
    args = parser.parse_args()
    wanted = args.variants or VARIANTS
    unknown = [v for v in wanted if v not in SPECS]
    if unknown:
        raise SystemExit(f"unknown variant(s) {unknown}. Known: {VARIANTS}")

    staging = ROOT / ".fetch-cache" / "ppocrv6"
    for variant in (v for v in VARIANTS if v in wanted):
        print(variant)
        state = {DETECTOR: _download(variant, "det", staging),
                 RECOGNISER: _download(variant, "rec", staging)}
        revision_dir = args.weights_dir / "ppocrv6" / variant / args.revision
        revision_dir.mkdir(parents=True, exist_ok=True)
        target = revision_dir / "torch-fp32.pth"
        torch.save(state, target)
        load(target, SPECS[variant])   # the strict load is the check that the geometry matches
        (revision_dir / "NOTICE").write_text(NOTICE.format(
            variant=variant, rule="=" * (11 + len(variant)),
            det_repo=_repo(variant, "det"), det_commit=COMMITS[(variant, "det")],
            rec_repo=_repo(variant, "rec"), rec_commit=COMMITS[(variant, "rec")]))
        print(f"  {'fused':<10} {target}  {target.stat().st_size / 1e6:.1f} MB")
        require_licence(revision_dir, "Apache-2.0", LICENCE_SOURCE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

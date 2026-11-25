#!/usr/bin/env python3
"""Fetch BiRefNet's published checkpoints into the local ``weights/`` tree.

Bootstrap tooling. It runs on a machine you control, never ships, and is the only place the
upstream URLs live.

    python tools/fetch/birefnet.py                 # all four
    python tools/fetch/birefnet.py general lite

**Which checkpoints.** The authors publish sixteen repositories under ``ZhengPeng7/``, all
``license: mit`` on their cards. mozo takes four, between them every distinct code path:

    general   ZhengPeng7/BiRefNet           Swin-L, 1024   general use (987k downloads/month)
    lite      ZhengPeng7/BiRefNet_lite      Swin-T, 1024   the fast one
    matting   ZhengPeng7/BiRefNet-matting   Swin-L, 1024   soft alpha, trimap-free matting
    hr        ZhengPeng7/BiRefNet_HR        Swin-L, 2048   general use at high resolution

Not taken, and why, is in ``mozo/vendors/birefnet_deploy/PROVENANCE.md``: ``BiRefNet-portrait``
(88k, the largest omission -- ``matting`` is trained on its data and more), ``BiRefNet_HR-matting``,
``BiRefNet_dynamic`` and the academic DIS/HRSOD/COD checkpoints.

**The hash comes from Hugging Face's API**, read for the pinned commit at fetch time.

**What is published is not the byte stream upstream serves.** The safetensors is read and written
back as an ordinary ``.pth`` (mozo has no safetensors dependency; this script does), **in the
dtype it was published in** -- ``general`` and ``hr`` are float16 upstream and stay float16, which
the vendored loader copies into float32 parameters exactly. Twelve tensors are left out: the
multi-scale supervision heads (``decoder.conv_ms_spvn_*``) and the gradient-prediction heads
(``decoder.gdt_convs_pred_*``), which upstream runs only in training. Every other tensor is
written as the safetensors holds it, and the result must load strictly into the vendored network.

The output is ``weights/birefnet/<variant>/<revision>/torch-fp32.pth`` and a NOTICE. The LICENSE
beside them is not written by this script; if it is missing this script says how to get it.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from common import download_verified, hub_lfs_sha256s, require_licence, variant_parser  # noqa: E402

_HF = "https://huggingface.co"

#: The revision this family is published under.
REVISION = "2026-11-25"

#: Upstream's own MIT text, at the GitHub commit PROVENANCE.md records.
LICENCE_SOURCE = ("https://raw.githubusercontent.com/ZhengPeng7/BiRefNet/"
                  "f7079632d681ecf12161fcea8905682934e59e51/LICENSE")

#: Tensors only training reads, by key prefix.
TRAINING_ONLY = ("decoder.conv_ms_spvn_", "decoder.gdt_convs_pred_")


@dataclass(frozen=True)
class Checkpoint:
    repo: str
    #: The Hugging Face commit pinned in PROVENANCE.md.
    commit: str
    kind: str


CHECKPOINTS: dict[str, Checkpoint] = {
    "general": Checkpoint("ZhengPeng7/BiRefNet", "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4",
                          "Swin-L at 1024x1024, general use: background removal"),
    "lite": Checkpoint("ZhengPeng7/BiRefNet_lite", "aa62cd87eafb9cc43056d08ef3615a14628b831d",
                       "Swin-T at 1024x1024, general use: background removal"),
    "matting": Checkpoint("ZhengPeng7/BiRefNet-matting",
                          "eccde0a8cbdce7ac5fecfeb06340fe7b949e85d9",
                          "Swin-L at 1024x1024, general trimap-free matting"),
    "hr": Checkpoint("ZhengPeng7/BiRefNet_HR", "a7a562f6fd16021180f2f4348f4de003a2d3d1e1",
                     "Swin-L at 2048x2048, general use at high resolution"),
}


def convert(source: Path, variant: str) -> tuple[dict[str, torch.Tensor], int]:
    """The state dict mozo publishes: upstream's tensors minus :data:`TRAINING_ONLY`.

    Checked against the vendored network: the result must load strictly into it.
    """
    from safetensors.torch import load_file

    from mozo.vendors.birefnet_deploy import MODEL_SPECS, BiRefNet

    state = load_file(source)
    kept = {key: tensor for key, tensor in state.items() if not key.startswith(TRAINING_ONLY)}

    BiRefNet(MODEL_SPECS[variant]).load_state_dict(kept, strict=True)
    dtypes = sorted({str(t.dtype).removeprefix("torch.") for t in kept.values()})
    print(f"  {variant:<8} {len(kept)} of {len(state)} tensors kept ({', '.join(dtypes)}), "
          f"{len(state) - len(kept)} training-only dropped")
    return kept, len(state)


def _notice(variant: str, entry: Checkpoint, sha256: str, kept: int, total: int) -> str:
    return (
        f"BiRefNet -- {variant} ({entry.repo.split('/')[1]})\n"
        f"{entry.kind}\n\n"
        "Copyright (c) 2024 ZhengPeng.\n"
        "Peng Zheng, Dehong Gao, Deng-Ping Fan, Li Liu, Jorma Laaksonen, Wanli Ouyang, Nicu Sebe.\n"
        "\"Bilateral Reference for High-Resolution Dichotomous Image Segmentation\",\n"
        "CAAI Artificial Intelligence Research 3, 9150038 (2024). arXiv:2401.03407.\n\n"
        f"Source:  {_HF}/{entry.repo} at commit {entry.commit}\n"
        "Project: https://github.com/ZhengPeng7/BiRefNet\n"
        "Licence: MIT, stated on the ungated model card (license: mit), and the licence of the\n"
        "         project repository (LICENSE, Copyright (c) 2024 ZhengPeng). The full text is in\n"
        "         the LICENSE file beside this one.\n\n"
        "The training data behind these weights includes datasets published under their own,\n"
        "in some cases research-only, terms (for example P3M-10k and AM-2k). The MIT grant above\n"
        "is the publisher's statement about these weights; mozo takes it as stated and does not\n"
        "track the terms of the data behind them.\n\n"
        "This is not the byte stream that repository serves: tools/fetch/birefnet.py verifies\n"
        "model.safetensors against the sha256 the Hub records, drops the training-only\n"
        "supervision heads (decoder.conv_ms_spvn_*, decoder.gdt_convs_pred_*), and writes the\n"
        f"remaining {kept} of {total} tensors, unaltered and in their published dtype, as an\n"
        "ordinary checkpoint. The source file is:\n"
        f"           model.safetensors  sha256 {sha256}\n"
    )


def fetch(variant: str, revision: str, weights_dir: Path) -> None:
    """Download one checkpoint, verify it, convert it, and place it with its notice."""
    entry = CHECKPOINTS[variant]
    target = weights_dir / "birefnet" / variant / revision / "torch-fp32.pth"
    target.parent.mkdir(parents=True, exist_ok=True)

    source = target.with_name("model.safetensors")
    sha256 = download_verified(
        f"{_HF}/{entry.repo}/resolve/{entry.commit}/model.safetensors", source,
        hub_lfs_sha256s(entry.repo, entry.commit)["model.safetensors"],
        label=variant, width=8, detail="MIT")

    kept, total = convert(source, variant)
    torch.save(kept, target)
    source.unlink()

    (target.parent / "NOTICE").write_text(_notice(variant, entry, sha256, len(kept), total))
    require_licence(target.parent, "MIT", LICENCE_SOURCE)


def main() -> int:
    args = variant_parser(__doc__, ROOT / "weights", revision=REVISION).parse_args()
    wanted = args.variants or list(CHECKPOINTS)
    unknown = [v for v in wanted if v not in CHECKPOINTS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}. Known: {list(CHECKPOINTS)}")
    for variant in wanted:
        fetch(variant, args.revision, args.weights_dir)
    print(f"\n{len(wanted)} checkpoints in {args.weights_dir}, MIT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fetch DINOv2's published backbones into the local ``weights/`` tree.

Bootstrap tooling. It runs on a machine you control, never ships, and is the only place the
upstream URLs live -- the vendored package carries none of them.

    python tools/fetch/dinov2.py                  # every variant
    python tools/fetch/dinov2.py small base-reg

**Read from Meta's Hugging Face repositories, pinned to a commit, and verified against the Hub's
own hash.** The authors' download links (``dl.fbaipublicfiles.com/dinov2/...pth``) carry no
content hash -- only a multipart ETag -- while the Hub records a sha256 for every LFS file. The two
hold the same tensors: every one of 175 to 567 per variant was ``torch.equal`` when this family was
built, and ``tools/verify/dinov2.py`` holds mozo's artifact to the authors' ``.pth`` on every run.

**Renamed back to the authors' layout.** The Hub's files are in ``transformers``' layout, with each
block's attention projection split into ``query``, ``key`` and ``value``. They are concatenated
back into upstream's ``qkv`` (a copy, so exact) and every other key renamed, so the published file
is the state dict ``torch.hub.load('facebookresearch/dinov2', ...)`` loads. Nothing is dropped:
the published repositories are the bare backbone, ``mask_token`` included.

``pytorch_model.bin`` is not used: a pickle is not something to load from the internet. That
makes ``safetensors`` a dependency of *this script*; nothing in ``mozo/`` imports it.

The output is ``weights/dinov2/<variant>/<revision>/torch-fp32.pth`` and a NOTICE. The LICENSE
beside them is upstream's, placed by hand (``require_licence`` says how).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from common import download_verified, hub_lfs_sha256s, require_licence, variant_parser  # noqa: E402

from mozo.vendors.dinov2_deploy.config import SPECS  # noqa: E402

_HF = "https://huggingface.co"

#: The revision this family is published under.
REVISION = "2026-11-25"

#: The authors' code, at the commit the vendored package was extracted from.
UPSTREAM = "https://github.com/facebookresearch/dinov2"
COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"

#: Where the authors serve the original checkpoints. The reference the gate loads.
AUTHORS = "https://dl.fbaipublicfiles.com/dinov2"

#: Meta's Hugging Face repository for each variant, and the commit every file is read from.
REPOSITORIES = {
    "small": ("facebook/dinov2-small", "ed25f3a31f01632728cabb09d1542f84ab7b0056"),
    "base": ("facebook/dinov2-base", "f9e44c814b77203eaa57a6bdbbd535f21ede1415"),
    "large": ("facebook/dinov2-large", "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c"),
    "giant": ("facebook/dinov2-giant", "611a9d42f2335e0f921f1e313ad3c1b7178d206d"),
    "small-reg": ("facebook/dinov2-with-registers-small", "0d9846e56b43a21fa46d7f3f5070f0506a5795a9"),
    "base-reg": ("facebook/dinov2-with-registers-base", "a1d738ccfa7ae170945f210395d99dde8adb1805"),
}

LICENCE_SOURCE = f"https://raw.githubusercontent.com/facebookresearch/dinov2/{COMMIT}/LICENSE"


def authors_url(variant: str) -> str:
    """The authors' own checkpoint for *variant*: ``dinov2_vits14/dinov2_vits14_reg4_pretrain.pth``."""
    spec = SPECS[variant]
    base = spec.upstream.removesuffix("_reg")
    return f"{AUTHORS}/{base}/{base}{'_reg4' if spec.registers else ''}_pretrain.pth"


#: ``transformers`` name -> upstream name, for everything outside the blocks.
_RENAMES = {
    "embeddings.cls_token": "cls_token",
    "embeddings.mask_token": "mask_token",
    "embeddings.position_embeddings": "pos_embed",
    "embeddings.register_tokens": "register_tokens",
    "embeddings.patch_embeddings.projection.weight": "patch_embed.proj.weight",
    "embeddings.patch_embeddings.projection.bias": "patch_embed.proj.bias",
    "layernorm.weight": "norm.weight",
    "layernorm.bias": "norm.bias",
}

#: Within block *i*: ``transformers`` suffix -> upstream suffix.
_BLOCK = {
    "norm1": "norm1", "norm2": "norm2",
    "attention.output.dense": "attn.proj",
    "mlp.fc1": "mlp.fc1", "mlp.fc2": "mlp.fc2",            # GELU MLP
    "mlp.weights_in": "mlp.w12", "mlp.weights_out": "mlp.w3",  # giant's SwiGLU
    "layer_scale1.lambda1": "ls1.gamma", "layer_scale2.lambda1": "ls2.gamma",
}


def to_upstream(hub: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The Hub's state dict in the authors' layout. Raises if a key is left over either way."""
    out, used = {}, set()
    for old, new in _RENAMES.items():
        if old in hub:
            out[new], _ = hub[old], used.add(old)
    i = 0
    while f"encoder.layer.{i}.norm1.weight" in hub:
        src, dst = f"encoder.layer.{i}.", f"blocks.{i}."
        for kind in ("weight", "bias"):
            parts = [src + f"attention.attention.{n}.{kind}" for n in ("query", "key", "value")]
            out[dst + f"attn.qkv.{kind}"] = torch.cat([hub[p] for p in parts])
            used.update(parts)
        for old, new in _BLOCK.items():
            for kind in (("",) if "lambda1" in old else (".weight", ".bias")):
                if src + old + kind in hub:
                    out[dst + new + kind] = hub[src + old + kind]
                    used.add(src + old + kind)
        i += 1
    left = sorted(set(hub) - used)
    if left:
        raise SystemExit(f"keys the rename does not account for: {left[:5]}")
    return out


def _notice(variant: str, sha256: str) -> str:
    """Attribution that travels with the weights, as Apache-2.0 asks."""
    spec = SPECS[variant]
    repo, commit = REPOSITORIES[variant]
    return (
        f"DINOv2 -- {variant} ({spec.upstream})\n"
        f"a self-supervised Vision Transformer: an image in, one {spec.width}-dimensional vector out\n\n"
        "Copyright (c) Meta Platforms, Inc. and affiliates.\n\n"
        f"Project: {UPSTREAM} (code at commit {COMMIT})\n"
        f"Weights: {authors_url(variant)}\n"
        f"         as mirrored by Meta at {_HF}/{repo}, commit {commit}\n"
        "Papers:  DINOv2: Learning Robust Visual Features without Supervision, arXiv:2304.07193\n"
        "         Vision Transformers Need Registers, arXiv:2309.16588\n"
        "Licence: Apache-2.0 (full text in the LICENSE file beside this one). The project's README\n"
        "         states \"DINOv2 code and model weights are released under the Apache License 2.0\";\n"
        "         its MODEL_CARD.md and the Hugging Face model card say the same.\n\n"
        "This file is not the byte stream Meta serves. tools/fetch/dinov2.py verifies\n"
        "model.safetensors against the sha256 the Hub records, then writes the same tensors in\n"
        "the authors' own key layout: each block's query, key and value are concatenated back\n"
        "into one qkv projection. Nothing is cast, dropped or added. The source file is:\n"
        f"           model.safetensors  sha256 {sha256}\n"
    )


def fetch(variant: str, revision: str, weights_dir: Path) -> None:
    """Download one variant, verify it, rename it, and place it with its notice."""
    repo, commit = REPOSITORIES[variant]
    target = weights_dir / "dinov2" / variant / revision / "torch-fp32.pth"
    staged = target.with_name("model.safetensors")

    sha256 = hub_lfs_sha256s(repo, commit)["model.safetensors"]
    if not target.is_file():
        download_verified(f"{_HF}/{repo}/resolve/{commit}/model.safetensors", staged, sha256,
                          label=variant, width=9, detail="Apache-2.0", timeout=600)
        from safetensors.torch import load_file

        state = to_upstream(load_file(staged))
        dtypes = {str(t.dtype) for t in state.values()}
        if dtypes != {"torch.float32"}:
            raise SystemExit(f"expected float32 throughout, found {sorted(dtypes)}")
        print(f"  {variant:9s} writing {len(state)} tensors in the authors' layout")
        torch.save(state, target)
        staged.unlink()

    (target.parent / "NOTICE").write_text(_notice(variant, sha256))
    require_licence(target.parent, "Apache-2.0", LICENCE_SOURCE)


def main() -> int:
    parser = variant_parser(__doc__, ROOT / "weights", revision=REVISION)
    args = parser.parse_args()
    for variant in args.variants or list(SPECS):
        if variant not in SPECS:
            raise SystemExit(f"unknown variant {variant!r}; known: {list(SPECS)}")
        fetch(variant, args.revision, args.weights_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fetch EoMT's published checkpoints into the local ``weights/`` tree.

Bootstrap tooling. It runs on a machine you control, never ships, and is the only place the
upstream URLs live -- the vendored package carries none of them.

    python tools/fetch/eomt.py                          # every variant
    python tools/fetch/eomt.py coco-panoptic-small

**Read from the authors' Hugging Face repositories, pinned to a commit, verified against the Hub's
own hash.** Each repository holds the authors' ``pytorch_model.bin`` (what upstream's notebook
loads) and a ``model.safetensors`` in ``transformers``' layout. The safetensors is taken: a pickle
is not something to load from the internet, and the two hold the same tensors --
``tools/verify/eomt.py`` holds mozo's artifact to the authors' ``.bin`` tensor by tensor on every
run.

**Renamed back to upstream's layout.** ``transformers`` splits each block's ``qkv`` into
``q_proj``/``k_proj``/``v_proj``, keeps the position embedding as an ``nn.Embedding`` and renames
every head; all of it is put back (a concatenation and renames, so exact), giving the state dict
of upstream's ``EoMT`` network module. Two of upstream's buffers are not in the Hub's file and are
added: ``encoder.pixel_mean`` and ``encoder.pixel_std``, ImageNet's statistics exactly as
``models/vit.py`` constructs them. ``criterion.empty_weight`` (the loss's no-object weight) is
dropped: it is not part of the network.

The class names are the same repository's ``config.json`` ``id2label``, at the same commit; upstream
itself names no classes. The output is ``weights/eomt/<variant>/<revision>/torch-fp32.pth``,
``labels.json`` and a NOTICE. The LICENSE beside them is upstream's, placed by hand
(``require_licence`` says how).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from common import download_verified, hub_json, hub_lfs_sha256s, require_licence, variant_parser  # noqa: E402

from mozo.vendors.eomt_deploy.config import MEAN, SPECS, STD  # noqa: E402

_HF = "https://huggingface.co"

#: The revision this family is published under.
REVISION = "2026-11-25"

#: The authors' code, at the commit the vendored package was extracted from.
UPSTREAM = "https://github.com/tue-mps/eomt"
COMMIT = "aa017c34afbe023de9462b17ae6067b52fef9867"

#: The commit of each repository every file is read from.
COMMITS = {
    "coco-panoptic-small": "10f5326794a8ce7e0e975791033ae1c13aec3767",
    "coco-panoptic-base": "1ac497947a2410fdb05279c5907c5209af37b9f1",
    "coco-panoptic-large": "dcd130bed9b1ebda7041fd660fddb16f905b9c3b",
    "coco-panoptic-large-1280": "d1bb11420d0d054623ffb2b6899ff0430e4cc655",
    "ade20k-semantic-large": "aeb262f5a0bc609236d27f1277c3173ce57aeb27",
}

LICENCE_SOURCE = f"https://raw.githubusercontent.com/tue-mps/eomt/{COMMIT}/LICENSE"

#: What each variant was trained on, for the NOTICE. Neither dataset's terms are a licence on
#: the weights; they are stated so a user can judge their own use.
DATA = {
    "panoptic": (
        "COCO 2017 panoptic (https://cocodataset.org). The annotations are CC-BY-4.0; the images\n"
        "are Flickr photographs under their owners' own licences."),
    "semantic": (
        "ADE20K (MIT Scene Parsing, https://ade20k.csail.mit.edu).\n"
        "The annotations are BSD-3-Clause; MIT CSAIL's terms make the *images*\n"
        "available for non-commercial research and educational use only, and MIT does not own\n"
        "their copyright. Those terms are not a licence on these weights, which the authors\n"
        "release under MIT, but judge your own use with them in mind."),
}

#: ``transformers`` name -> upstream name, outside the blocks.
_RENAMES = {
    "attn_mask_probs": "attn_mask_probs",
    "query.weight": "q.weight",
    "class_predictor.weight": "class_head.weight",
    "class_predictor.bias": "class_head.bias",
    "embeddings.cls_token": "encoder.backbone.cls_token",
    "embeddings.register_tokens": "encoder.backbone.reg_token",
    "embeddings.patch_embeddings.projection.weight": "encoder.backbone.patch_embed.proj.weight",
    "embeddings.patch_embeddings.projection.bias": "encoder.backbone.patch_embed.proj.bias",
    "layernorm.weight": "encoder.backbone.norm.weight",
    "layernorm.bias": "encoder.backbone.norm.bias",
    **{f"mask_head.fc{i + 1}.{kind}": f"mask_head.{2 * i}.{kind}"
       for i in range(3) for kind in ("weight", "bias")},
}

#: Within upscale block *i*: ``transformers`` suffix -> upstream suffix.
_UPSCALE = {"conv1.weight": "conv1.weight", "conv1.bias": "conv1.bias",
            "conv2.weight": "conv2.weight",
            "layernorm2d.weight": "norm.weight", "layernorm2d.bias": "norm.bias"}

#: Within block *i*: ``transformers`` suffix -> upstream suffix.
_BLOCK = {
    "norm1.weight": "norm1.weight", "norm1.bias": "norm1.bias",
    "norm2.weight": "norm2.weight", "norm2.bias": "norm2.bias",
    "attention.out_proj.weight": "attn.proj.weight", "attention.out_proj.bias": "attn.proj.bias",
    "mlp.fc1.weight": "mlp.fc1.weight", "mlp.fc1.bias": "mlp.fc1.bias",
    "mlp.fc2.weight": "mlp.fc2.weight", "mlp.fc2.bias": "mlp.fc2.bias",
    "layer_scale1.lambda1": "ls1.gamma", "layer_scale2.lambda1": "ls2.gamma",
}

#: In the Hub's file and not part of the network.
_DROPPED = {"criterion.empty_weight"}


def to_upstream(hub: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The Hub's state dict in upstream's network layout. Raises if a key is left over."""
    out, used = {}, set(_DROPPED & set(hub))
    for old, new in _RENAMES.items():
        out[new] = hub[old]
        used.add(old)
    # An ``nn.Embedding`` of (grid², width) there; timm's (1, grid², width) parameter here.
    out["encoder.backbone.pos_embed"] = hub["embeddings.position_embeddings.weight"][None].clone()
    used.add("embeddings.position_embeddings.weight")
    i = 0
    while f"layers.{i}.norm1.weight" in hub:
        src, dst = f"layers.{i}.", f"encoder.backbone.blocks.{i}."
        for kind in ("weight", "bias"):
            names = [src + f"attention.{n}_proj.{kind}" for n in ("q", "k", "v")]
            out[dst + f"attn.qkv.{kind}"] = torch.cat([hub[n] for n in names])
            used.update(names)
        for old, new in _BLOCK.items():
            out[dst + new] = hub[src + old]
            used.add(src + old)
        i += 1
    j = 0
    while f"upscale_block.block.{j}.conv1.weight" in hub:
        for old, new in _UPSCALE.items():
            out[f"upscale.{j}.{new}"] = hub[f"upscale_block.block.{j}.{old}"]
            used.add(f"upscale_block.block.{j}.{old}")
        j += 1
    left = sorted(set(hub) - used)
    if left:
        raise SystemExit(f"keys the rename does not account for: {left[:5]}")
    # ``models/vit.py``: registered buffers the Hub's file does not carry.
    out["encoder.pixel_mean"] = torch.tensor(MEAN).reshape(1, -1, 1, 1)
    out["encoder.pixel_std"] = torch.tensor(STD).reshape(1, -1, 1, 1)
    return out


def labels(variant: str) -> list[dict]:
    """The repository's own ``id2label``, as ``[{"id", "name"}]`` in id order."""
    spec = SPECS[variant]
    config = hub_json(f"{_HF}/{spec.repository}/raw/{COMMITS[variant]}/config.json")
    names = config["id2label"]
    if len(names) != spec.classes:
        raise SystemExit(f"{variant}: {len(names)} names for {spec.classes} classes")
    return [{"id": i, "name": names[str(i)]} for i in range(spec.classes)]


def _notice(variant: str, sha256: str) -> str:
    """Attribution that travels with the weights, as MIT asks."""
    spec = SPECS[variant]
    return (
        f"EoMT -- {variant} ({spec.repository})\n"
        f"Encoder-only Mask Transformer: {spec.task} segmentation, a DINOv2 ViT with "
        f"{spec.queries} queries\n\n"
        "Copyright (c) 2025 Mobile Perception Systems Lab at TU/e\n\n"
        f"Project: {UPSTREAM} (code at commit {COMMIT}, config configs/dinov2/{spec.config})\n"
        f"Weights: {_HF}/{spec.repository}, commit {COMMITS[variant]}\n"
        "Paper:   Your ViT is Secretly an Image Segmentation Model, CVPR 2025, arXiv:2503.19108\n"
        "Licence: MIT (full text in the LICENSE file beside this one), stated on the repository\n"
        "         and as `license: mit` on the Hugging Face model card. The backbone was\n"
        "         initialised from Meta's DINOv2 with registers, released under Apache-2.0.\n\n"
        f"Training data: {DATA[spec.task]}\n\n"
        "This file is not the byte stream the authors serve. tools/fetch/eomt.py verifies\n"
        "model.safetensors against the sha256 the Hub records, then writes the same tensors in\n"
        "upstream's own network layout: each block's q, k and v projections are concatenated back\n"
        "into one qkv, the loss's no-object weight is dropped, and the network's two constant\n"
        "buffers (ImageNet mean and std) are added. Nothing is cast. The source file is:\n"
        f"           model.safetensors  sha256 {sha256}\n"
        "labels.json is the same repository's config.json id2label.\n"
    )


def fetch(variant: str, revision: str, weights_dir: Path) -> None:
    """Download one variant, verify it, rename it, and place it with its labels and notice."""
    spec = SPECS[variant]
    commit = COMMITS[variant]
    target = weights_dir / "eomt" / variant / revision / "torch-fp32.pth"
    staged = target.with_name("model.safetensors")

    sha256 = hub_lfs_sha256s(spec.repository, commit)["model.safetensors"]
    if not target.is_file():
        download_verified(f"{_HF}/{spec.repository}/resolve/{commit}/model.safetensors", staged,
                          sha256, label=variant, width=24, detail="MIT", timeout=900)
        from safetensors.torch import load_file

        state = to_upstream(load_file(staged))
        dtypes = {str(t.dtype) for t in state.values()}
        if dtypes != {"torch.float32"}:
            raise SystemExit(f"expected float32 throughout, found {sorted(dtypes)}")
        print(f"  {variant:24s} writing {len(state)} tensors in upstream's layout")
        torch.save(state, target)
        staged.unlink()

    (target.parent / "labels.json").write_text(json.dumps(labels(variant), indent=2) + "\n")
    (target.parent / "NOTICE").write_text(_notice(variant, sha256))
    require_licence(target.parent, "MIT", LICENCE_SOURCE)


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

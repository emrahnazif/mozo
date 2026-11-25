#!/usr/bin/env python3
"""Fetch Depth Anything 3's Apache-2.0 depth checkpoints into the local ``weights/`` tree.

Bootstrap tooling. It runs on a machine you control, never ships, and is the only place the
upstream URLs live.

    python tools/fetch/depth_anything_v3.py                 # all three
    python tools/fetch/depth_anything_v3.py small

**Which checkpoints, and which not.** Upstream publishes eleven. mozo takes the three that are
Apache-2.0 on their model card *and* in the authors' own model table, and that predict depth
from one image without a camera:

    small        depth-anything/DA3-SMALL        Apache-2.0
    base         depth-anything/DA3-BASE         Apache-2.0
    mono-large   depth-anything/DA3MONO-LARGE    Apache-2.0

Not taken: DA3-LARGE, DA3-GIANT, the NESTED models and their ``-1.1`` retrains (CC-BY-NC-4.0);
DA3-LARGE-1.1, whose card says apache-2.0 while the authors' table says CC BY-NC 4.0 (the
stricter claim wins); DA3METRIC-LARGE, Apache-2.0, whose output is metres only once multiplied
by the camera's focal length, which mozo's depth API cannot be told (see the family's page).

**The hash comes from Hugging Face's API**, read for the pinned commit at fetch time.

**What is published is not the byte stream upstream serves.** The safetensors is read and
written back as an ordinary ``.pth`` (mozo has no safetensors dependency; this script does),
with three groups of tensors left out because nothing in single-image depth reads them: the
camera encoder (``cam_enc``, pose-conditioned input), the camera decoder (``cam_dec``, pose
output) and the ray branch of the any-view head (``head.scratch.*_aux``). The ``model.`` prefix
of upstream's API wrapper is dropped. Every tensor kept is checked ``torch.equal`` to the
safetensors one, and the result must load strictly into the vendored network.

The output is ``weights/depth_anything_v3/<variant>/<revision>/torch-fp32.pth``, ``pos_embed.pth``
(the checkpoint's learned position embedding alone, which the graph runtimes compute their shape
inputs from, so they need not download the checkpoint) and a NOTICE.
The LICENSE beside them is not written by this script; if it is missing this script says so.
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

#: Upstream's own Apache-2.0 text, at the commit PROVENANCE.md records.
LICENCE_SOURCE = ("https://raw.githubusercontent.com/ByteDance-Seed/Depth-Anything-3/"
                  "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/LICENSE")

#: Tensors no single-image depth computation reads, by prefix of upstream's key (after the
#: ``model.`` wrapper prefix is removed).
UNUSED = ("cam_enc.", "cam_dec.", "head.scratch.refinenet1_aux.", "head.scratch.refinenet2_aux.",
          "head.scratch.refinenet3_aux.", "head.scratch.refinenet4_aux.",
          "head.scratch.output_conv1_aux.", "head.scratch.output_conv2_aux.")


#: The learned position embedding, published on its own as ``pos_embed`` for the graph runtimes.
POS_EMBED = "backbone.pretrained.pos_embed"


@dataclass(frozen=True)
class Checkpoint:
    repo: str
    #: The Hugging Face commit pinned in PROVENANCE.md.
    commit: str
    params: str
    kind: str


CHECKPOINTS: dict[str, Checkpoint] = {
    "small": Checkpoint("depth-anything/DA3-SMALL", "e08cab65ca0ec38e7826075418411ab90cab4da3",
                        "0.08B", "any-view model, run on one view: relative depth"),
    "base": Checkpoint("depth-anything/DA3-BASE", "f4a6c9b3c95e41c82048423d3493a81ec3fa810e",
                       "0.12B", "any-view model, run on one view: relative depth"),
    "mono-large": Checkpoint("depth-anything/DA3MONO-LARGE",
                             "f465978e618db8cc79c83b8bbf24964857db1875",
                             "0.35B", "monocular model: relative depth, with a sky head"),
}


def convert(source: Path, variant: str) -> dict[str, torch.Tensor]:
    """The state dict mozo publishes: upstream's tensors minus :data:`UNUSED`, prefix dropped.

    Checked against the vendored network: the result loads strictly, and every tensor equals
    the source's.
    """
    from safetensors.torch import load_file

    from mozo.vendors.depth_anything_v3_deploy import MODEL_SPECS, DepthAnything3Net

    upstream = load_file(source)
    if not all(key.startswith("model.") for key in upstream):
        raise SystemExit(f"{source}: expected every key under 'model.'")
    state = {key[len("model."):]: tensor for key, tensor in upstream.items()}
    kept = {key: tensor for key, tensor in state.items() if not key.startswith(UNUSED)}

    net = DepthAnything3Net(MODEL_SPECS[variant])
    net.load_state_dict(kept, strict=True)
    for key, tensor in net.state_dict().items():
        if tensor.dtype != state[key].dtype or not torch.equal(tensor, state[key]):
            raise SystemExit(f"{variant}: {key} did not survive the conversion")
    print(f"  {variant:<10} {len(kept)} of {len(state)} tensors kept, "
          f"{len(state) - len(kept)} unused dropped")
    return kept


def _notice(variant: str, entry: Checkpoint, sha256: str, kept: int, total: int) -> str:
    dropped = (
        "verifies model.safetensors against the sha256 the Hub records, drops the camera\n"
        "encoder, camera decoder and ray-head tensors that single-image depth never reads, and\n"
        f"writes the remaining {kept} of {total} tensors, unaltered, as an ordinary checkpoint.\n"
        if kept < total else
        "verifies model.safetensors against the sha256 the Hub records and writes all of its\n"
        f"{total} tensors, unaltered, as an ordinary checkpoint.\n")
    return (
        f"Depth Anything 3 -- {variant} ({entry.repo.split('/')[1]}, {entry.params} parameters)\n"
        f"{entry.kind}\n\n"
        "Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.\n"
        "Haotong Lin, Sili Chen, Jun Hao Liew, Donny Y. Chen, Zhenyu Li, Guang Shi, Jiashi Feng,\n"
        "Bingyi Kang. \"Depth Anything 3: Recovering the Visual Space from Any Views\",\n"
        "arXiv:2511.10647.\n\n"
        f"Source:  {_HF}/{entry.repo} at commit {entry.commit}\n"
        "Project: https://github.com/ByteDance-Seed/Depth-Anything-3\n"
        "Licence: Apache-2.0, stated on the model card (license: apache-2.0) and in the\n"
        "         authors' model table in the project README. The full text is in the LICENSE\n"
        "         file beside this one.\n\n"
        "This is not the byte stream that repository serves: tools/fetch/depth_anything_v3.py\n"
        f"{dropped}"
        "The source file is:\n"
        f"           model.safetensors  sha256 {sha256}\n"
    )


def fetch(variant: str, revision: str, weights_dir: Path) -> None:
    """Download one checkpoint, verify it, convert it, and place it with its notice."""
    entry = CHECKPOINTS[variant]
    target = weights_dir / "depth_anything_v3" / variant / revision / "torch-fp32.pth"
    target.parent.mkdir(parents=True, exist_ok=True)

    source = target.with_name("model.safetensors")
    sha256 = download_verified(
        f"{_HF}/{entry.repo}/resolve/{entry.commit}/model.safetensors", source,
        hub_lfs_sha256s(entry.repo, entry.commit)["model.safetensors"],
        label=variant, width=10, detail="Apache-2.0")

    from safetensors import safe_open

    with safe_open(source, "pt") as handle:
        total = len(list(handle.keys()))
    kept = convert(source, variant)
    torch.save(kept, target)
    # The one tensor a graph runtime needs beside its graph: the shape inputs are computed from
    # it, and publishing it alone spares a graph runtime the whole checkpoint.
    torch.save(kept[POS_EMBED].clone(), target.with_name("pos_embed.pth"))
    source.unlink()

    (target.parent / "NOTICE").write_text(_notice(variant, entry, sha256, len(kept), total))
    require_licence(target.parent, "Apache-2.0", LICENCE_SOURCE)


def main() -> int:
    args = variant_parser(__doc__, ROOT / "weights", revision=REVISION).parse_args()
    wanted = args.variants or list(CHECKPOINTS)
    unknown = [v for v in wanted if v not in CHECKPOINTS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}. Known: {list(CHECKPOINTS)}")
    for variant in wanted:
        fetch(variant, args.revision, args.weights_dir)
    print(f"\n{len(wanted)} checkpoints in {args.weights_dir}, Apache-2.0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

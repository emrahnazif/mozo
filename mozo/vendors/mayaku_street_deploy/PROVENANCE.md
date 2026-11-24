# Provenance

A deployment-only extraction of the street models' inference path from
[Mayaku](https://github.com/emrahnazif/mayaku).

| | |
|---|---|
| Reference | `mayaku==3.1.1` from PyPI (tag `v3.1.1`, commit `5c9ecf16da85`) |
| Extracted | 2026-11-24 |
| Verified against | `mayaku` 3.1.1, `torch` 2.13.0+cu130, `torchvision` 0.28.0, Python 3.12, CPU |
| Gate | `tools/verify/mayaku_street.py` |

**Why 3.1.1.** The checkpoints were written by Mayaku 3.0.0 and, by the 3.1.1 changelog, "download
by name in every 3.x version". 3.1.0 is the first release with the prediction decode (a score
threshold, NMS at IoU 0.5, one class per location) that the checkpoints' `predict` blocks
configure, and 3.1.1 is the release that documents the street models. It is also the latest.

## The weights

The publisher's checkpoints, byte for byte, from the Hugging Face repositories the model cards
link to. The same bytes (same sha256) are served by Mayaku's own manifest,
`dtmfiles.com/mayaku/v1/models/manifest.json`, revision `2026-11-23`; checked 2026-11-24.

| mozo variant | upstream name | bytes | sha256 |
|---|---|---|---|
| `nano` | `mayaku-n-det-street` | 22,405,191 | `87349e38e049be8117c2e854e3a1352948b1f6a679d8fc88adc37eef16f6f289` |
| `medium` | `mayaku-m-det-street` | 137,716,005 | `d8154930fee7dc3fa675a60430d6065f64ea757a6d66e2e8bec8d888dbf14cd6` |
| `seg-nano` | `mayaku-n-seg-street` | 22,847,277 | `e79fedd195d1213a7ded66616afd3c370f0f3c3be8445733812e262cf3df447c` |
| `seg-medium` | `mayaku-m-seg-street` | 138,813,259 | `700860e6a818a71de1636869f6ad411f3c60b88b4a671df3b0a089bd783ffac3` |

Each holds `{"model": state_dict, "mayaku": sidecar}`; the sidecar is JSON, so the file loads with
`weights_only=True`. A det checkpoint is its seg checkpoint without the 64 `aux.*` keys: every
shared tensor is equal (measured), and both name the same source training checkpoint.

**Present and unused:** every convolution and residual add carries `act_fq.mn` / `act_fq.mx`, the
int8 activation ranges of quantisation-aware training. The fp32 deploy graph never reads them
(upstream's `strip_fake_quant`), so the loader drops exactly those keys by name and loads the rest
strictly.

## Licences

- **Code:** Apache-2.0 (GitHub licence `apache-2.0`; PyPI classifier; `LICENSE` and `NOTICE` in the
  3.1.1 wheel). Mayaku's NOTICE: "Copyright 2026 the mayaku authors ... bundles no third-party
  source code."
- **Weights, per checkpoint:** Apache-2.0, stated by the publisher on each of the four model cards
  (`license: apache-2.0`, ungated, and "The weights are Apache 2.0, like the library") and on the
  docs page `docs/ready-to-use/street.md`. A direct statement on the artifact, not an inference.
- **Training labels were made with SAM 3** (model cards and docs). The SAM License defines the
  "SAM Materials" as what Meta distributes -- models, code, weights, documentation -- and mentions
  outputs only in its warranty disclaimer and its litigation clause; it states no term that
  reaches a model trained on SAM 3's outputs. Recorded as found, not as legal advice. The training
  images are from Objects365, COCO, Wikimedia Commons, Songdo Vision, a roundabout aerial set and
  PDOK aerial photos, under their own terms, as the publisher lists them.

## What was taken

From `mayaku` 3.1.1, reduced to the street models (tiers `n` and `m`, masks at stride 4 from the
box towers):

- `model/blocks.py` -> `blocks.py`: the trained-form blocks and the fusion, verbatim in arithmetic.
- `model/{tiers,backbone,neck,head,aux,detector}.py` -> `network.py`.
- `inference/preprocess.py`, `data/geometry.py` -> `image.py` (letterbox, planar input, inverse).
- `inference/decode.py`, `model/box.py` -> `decode.py` (the prediction decode only).
- `model/mask.py` and `decode._decode_masks` -> `mask.py`, for one image.

Written here: `model.py` (`Spec` per variant, the sidecar check, `detect`, `Detector`).

## What was deliberately left behind

Training, quantisation-aware training and the int8 graph, the evaluation decode (multi-label,
conf 0.001), keypoints, oriented boxes, classification, the other tiers, the stride-8 mask path,
the `tower` aux arm, export, the CLI, the download client, and the dependencies on `pydantic`,
`pyyaml`, `typer`, `onnx`, `pycocotools` and `faster-coco-eval`.

## Where this diverges, and why

- **RGB in.** Upstream takes BGR (OpenCV's order) and reverses the channels while laying out the
  planar input. mozo hands over RGB, so the vendor letterboxes RGB and does not reverse. The
  resize works per channel and the padding is one grey in all three, so the planes are the same;
  the gate compares the canvas and the input tensor directly.
- **One image at a time.** Upstream decodes a batch; the vendor decodes one image, with the same
  operations on the same shapes (the mask chunking is upstream's).
- **The decode runs on the CPU** whatever ran the graph. Upstream decodes on the GPU when its model
  is on CUDA.
- **A threshold outside [0, 1) is refused.** Upstream raises `ZeroDivisionError` at 1 and a math
  domain error at 0; the vendor accepts 0 (keeps everything up to top-k) and refuses the rest with
  a message.
- **The geometry is written down** (`Spec`, `TIERS`) and the checkpoint's sidecar is checked
  against it, rather than the network being built from the sidecar's config as upstream does.
  The seg checkpoints carry no `predict` block; upstream then uses its defaults (0.4 / 0.5 / one
  class per location), the values the det checkpoints record, and the check holds an absent block
  to those defaults.

## Parity

(filled in from the gate run below)

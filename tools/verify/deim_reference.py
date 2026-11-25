#!/usr/bin/env python3
"""Run upstream DEIM on a set of images and record every stage, for ``tools/verify/deim.py``.

    python tools/verify/deim_reference.py --upstream DEIM/ --config <yml> --checkpoint <pth> \\
        --inputs inputs.npz --out reference.pt

Not meant to be run by hand: ``tools/verify/deim.py`` calls it once per variant, in a fresh
process, and that is not a convenience. Upstream's ``YAMLConfig`` merges each config into a
module-level registry that outlives it, so a second config built in the same process inherits the
first one's keys -- building ``nano`` and then ``large`` gives ``large`` nano's ``use_lab: True``
and decoder width. One config per process is the only way to get what each config really says.

It does what upstream's ``tools/inference/torch_inf.py`` does, at the pinned commit: build the
model from the config with ``HGNetv2.pretrained`` off, ``load_state_dict`` the checkpoint's
``model`` (strict, upstream's default), ``deploy()`` the model and the post-processor, and
preprocess with ``T.Compose([T.Resize((640, 640)), T.ToTensor()])`` on a PIL image. The image is
the RGB array the driver hands over, so both sides see the same decoded pixels.

Recorded per image: the network input, the encoder's outputs, the raw logits and boxes, the
post-processed labels, boxes and scores, and a digest of every module's output in call order
(``digests``), which is what locates a divergence to one module.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def digest(value) -> object:
    """A hashable record of a module output: sha256 of each tensor's bytes, dtype and shape."""
    if isinstance(value, torch.Tensor):
        t = value.detach().contiguous().cpu()
        return (str(t.dtype), tuple(t.shape), hashlib.sha256(t.numpy().tobytes()).hexdigest())
    if isinstance(value, (list, tuple)):
        return tuple(digest(v) for v in value)
    if isinstance(value, dict):
        return tuple((k, digest(v)) for k, v in sorted(value.items()))
    return repr(value)


def record_digests(model: torch.nn.Module):
    """Hook every named module. Returns the dict the hooks fill (name -> [digest per call]) and
    the hook handles, to remove once the call being recorded is done."""
    store: dict[str, list] = defaultdict(list)
    handles = [module.register_forward_hook(
                   lambda _m, _i, out, name=name: store[name].append(digest(out)))
               for name, module in model.named_modules() if name]
    return store, handles


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.upstream.resolve()))
    import torchvision.transforms as T
    from PIL import Image

    from engine.core import YAMLConfig

    cfg = YAMLConfig(str(args.config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state)
    model = cfg.model.deploy()
    postprocessor = cfg.postprocessor.deploy()
    transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])

    store, _ = record_digests(model)
    encoder_out: list = []
    model.encoder.register_forward_hook(
        lambda _m, _i, out: encoder_out.append([t.detach().clone() for t in out]))

    inputs = np.load(args.inputs)
    results = {}
    for name in inputs.files:
        rgb = inputs[name]
        store.clear()
        encoder_out.clear()
        im_pil = Image.fromarray(rgb)
        w, h = im_pil.size
        orig_size = torch.tensor([[w, h]])
        im_data = transforms(im_pil).unsqueeze(0)
        outputs = model(im_data)
        labels, boxes, scores = postprocessor(outputs, orig_size)
        results[name] = {
            "input": im_data,
            "encoder": encoder_out[0],
            "pred_logits": outputs["pred_logits"].detach(),
            "pred_boxes": outputs["pred_boxes"].detach(),
            "labels": labels.detach(), "boxes": boxes.detach(), "scores": scores.detach(),
            "digests": dict(store),
        }

    versions = {"torch": torch.__version__, "torchvision": __import__("torchvision").__version__,
                "pillow": Image.__version__, "numpy": np.__version__}
    torch.save({"results": results, "versions": versions}, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

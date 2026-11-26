"""Run upstream DEIM the way its inference script does, recording every stage, for
``tools/verify/deim.py``.

Build :class:`Upstream` **once per process**, and that is not a convenience. Upstream's
``YAMLConfig`` merges each config into a module-level registry that outlives it, so a second config
built in the same process inherits the first one's keys -- building ``nano`` and then ``large``
gives ``large`` nano's ``use_lab: True`` and decoder width. ``tools/verify/deim.py`` therefore runs
each variant in a process of its own, with upstream, the vendor and mozo side by side in it.

It does what upstream's ``tools/inference/torch_inf.py`` does, at the pinned commit: build the
model from the config with ``HGNetv2.pretrained`` off, ``load_state_dict`` the checkpoint's
``model`` (strict, upstream's default), ``deploy()`` the model and the post-processor and move them
to the device, preprocess with ``T.Compose([T.Resize((640, 640)), T.ToTensor()])`` on a PIL image
on the host, move the batch and the original size to the device, and run the network and the
post-processor there -- with autograd on, since the script never turns it off. The image is the RGB
array the driver hands over, so both sides see the same decoded pixels.

Recorded per image: the network input, the encoder's outputs, the raw logits and boxes, the
post-processed labels, boxes and scores, and a digest of every module's output in call order
(``digests``), which is what locates a divergence to one module. Everything is copied to the host,
which is exact.
"""

from __future__ import annotations

import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def digest(value) -> object:
    """A hashable record of a module output: each tensor's dtype, shape and the sha256 of its
    bytes (stricter than ``torch.equal``: it also tells ``-0.0`` from ``0.0``)."""
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


class Upstream:
    """Upstream's deploy-mode model and post-processor for one config, on *device*."""

    def __init__(self, upstream: Path, config: Path, checkpoint: Path, device: torch.device):
        sys.path.insert(0, str(Path(upstream).resolve()))
        import torchvision.transforms as T

        from engine.core import YAMLConfig

        cfg = YAMLConfig(str(config))
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
        cfg.model.load_state_dict(state)
        self.model = cfg.model.deploy().to(device)
        self.postprocessor = cfg.postprocessor.deploy().to(device)
        self.transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])
        self.device = device

        self.store, _ = record_digests(self.model)
        self.encoder_out: list = []
        self.model.encoder.register_forward_hook(
            lambda _m, _i, out: self.encoder_out.append([t.detach().to("cpu", copy=True) for t in out]))

    def __call__(self, rgb: np.ndarray) -> dict:
        from PIL import Image

        self.store.clear()
        self.encoder_out.clear()
        im_pil = Image.fromarray(rgb)
        w, h = im_pil.size
        orig_size = torch.tensor([[w, h]]).to(self.device)
        im_data = self.transforms(im_pil).unsqueeze(0).to(self.device)
        with torch.enable_grad():  # torch_inf.py runs its model without no_grad
            outputs = self.model(im_data)
            labels, boxes, scores = self.postprocessor(outputs, orig_size)
        return {
            "input": im_data.detach().cpu(),
            "encoder": self.encoder_out[0],
            "pred_logits": outputs["pred_logits"].detach().cpu(),
            "pred_boxes": outputs["pred_boxes"].detach().cpu(),
            "labels": labels.detach().cpu(), "boxes": boxes.detach().cpu(),
            "scores": scores.detach().cpu(),
            "digests": dict(self.store),
        }


def versions() -> dict:
    from PIL import Image

    return {"torch": torch.__version__, "torchvision": __import__("torchvision").__version__,
            "pillow": Image.__version__, "numpy": np.__version__}

"""
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.

Upstream's ``engine/deim/deim.py`` (the ``DEIM`` module and its ``deploy()``), plus what upstream
does around it in ``tools/inference/torch_inf.py``: build from the config, load the checkpoint's
``model`` state strictly, convert to deploy form. The loader is mozo's; the arithmetic is not.
"""

from pathlib import Path
from typing import Dict, Union

import torch
import torch.nn as nn

from .config import Spec, get_spec
from .decoder import DFINETransformer
from .hgnetv2 import HGNetv2
from .hybrid_encoder import HybridEncoder

#: Checkpoint entries the inference graph has no module for: the class embedding contrastive
#: denoising draws from during training. Dropped by name, so any *other* key the build does not
#: expect still fails the strict load.
UNUSED_KEYS = ("decoder.denoising_class_embed.weight",)


class DEIM(nn.Module):
    def __init__(self, backbone: nn.Module, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, x) -> Dict[str, torch.Tensor]:
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self


def build(spec: Spec) -> DEIM:
    """The train-form network for *spec*, uninitialised: a checkpoint is loaded over it."""
    levels = len(spec.enc_in_channels)
    backbone = HGNetv2(spec.backbone, use_lab=spec.use_lab, return_idx=spec.return_idx,
                       freeze_norm=spec.freeze_norm)
    encoder = HybridEncoder(
        in_channels=spec.enc_in_channels, feat_strides=spec.feat_strides,
        hidden_dim=spec.enc_hidden_dim, nhead=spec.nhead,
        dim_feedforward=spec.enc_dim_feedforward, dropout=0.0, enc_act=spec.enc_act,
        use_encoder_idx=(levels - 1,), num_encoder_layers=1, expansion=spec.expansion,
        depth_mult=spec.depth_mult, act=spec.enc_cross_act, eval_spatial_size=spec.size)
    decoder = DFINETransformer(
        num_classes=spec.num_classes, hidden_dim=spec.dec_hidden_dim,
        num_queries=spec.num_queries, feat_channels=(spec.enc_hidden_dim,) * levels,
        feat_strides=spec.feat_strides, num_levels=spec.num_levels,
        num_points=list(spec.num_points), nhead=spec.nhead, num_layers=spec.num_layers,
        dim_feedforward=spec.dec_dim_feedforward, activation=spec.dec_act,
        eval_spatial_size=spec.size, reg_max=spec.reg_max, reg_scale=spec.reg_scale,
        mlp_act=spec.dec_act)
    return DEIM(backbone, encoder, decoder)


def load_state(path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    """The ``model`` state of a released checkpoint, read without unpickling code.

    Upstream also accepts a training checkpoint's ``ema.module``; the released files carry only
    ``model``, and that is the one upstream's scripts take when there is no ``ema``.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "ema" in checkpoint:
        return checkpoint["ema"]["module"]
    return checkpoint["model"]


def load(variant: str, path: Union[str, Path]) -> DEIM:
    """Build *variant*, load *path* strictly, and convert to the deploy form upstream runs.

    Raises:
        ValueError: The checkpoint's keys, shapes or dtypes are not the variant's.
    """
    spec = get_spec(variant)
    model = build(spec)
    state = {k: v for k, v in load_state(path).items() if k not in UNUSED_KEYS}

    # ``load_state_dict`` checks names and shapes but casts dtypes silently, and one dtype here is
    # load-bearing: ``reg_scale`` is int64 in the checkpoints. Checking it means a spec written
    # with ``reg_scale=4.0`` fails here rather than drifting in the last bits.
    expected = model.state_dict()
    wrong = [k for k, v in state.items() if k in expected and v.dtype != expected[k].dtype]
    if wrong:
        raise ValueError(f"checkpoint dtypes differ from the {variant!r} build at {wrong[:5]}")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError(f"{path} is not a DEIM {variant!r} checkpoint: {error}") from None
    return model.deploy()

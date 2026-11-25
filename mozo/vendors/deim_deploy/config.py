"""The geometry of each released DEIM-D-FINE checkpoint, written down rather than inferred.

Each value is what upstream's YAML resolves to for that checkpoint's config (its ``__include__``
chain, read with upstream's own ``YAMLConfig``, one config per process -- see PROVENANCE for why
that matters). The strict load holds every shape here against the checkpoint; what a shape cannot
show (activations, which batch-norm arithmetic, ``reg_scale``) the gate holds against upstream's
outputs.
"""

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class Spec:
    """One checkpoint's architecture.

    Attributes:
        config: Upstream's config file for this checkpoint, for the record.
        backbone: HGNetv2 arch name.
        return_idx: Backbone stages feeding the encoder.
        use_lab: Learnable affine blocks after each backbone activation.
        freeze_norm: Backbone batch norms are ``FrozenBatchNorm2d`` (folded scale and shift)
            rather than eval-mode ``nn.BatchNorm2d``. A different arithmetic.
        enc_in_channels, feat_strides, enc_hidden_dim, enc_dim_feedforward, expansion,
            depth_mult: The hybrid encoder.
        dec_hidden_dim, dec_dim_feedforward, num_levels, num_layers, num_points, reg_scale:
            The D-FINE decoder.
        dec_act: The decoder's activation, for its layers' FFN and its MLP heads alike.
    """

    config: str
    backbone: str
    return_idx: Tuple[int, ...]
    use_lab: bool
    freeze_norm: bool
    enc_in_channels: Tuple[int, ...]
    feat_strides: Tuple[int, ...]
    enc_hidden_dim: int
    enc_dim_feedforward: int
    expansion: float
    depth_mult: float
    dec_hidden_dim: int
    dec_dim_feedforward: int
    num_levels: int
    num_layers: int
    num_points: Tuple[int, ...]
    reg_scale: int
    dec_act: str
    # The same in every released checkpoint.
    num_classes: int = 80
    num_queries: int = 300
    num_top_queries: int = 300
    reg_max: int = 32
    nhead: int = 8
    enc_act: str = "gelu"
    enc_cross_act: str = "silu"
    use_encoder_idx: int = -1  # the last level, resolved against ``enc_in_channels``
    size: Tuple[int, int] = (640, 640)  # (height, width): eval_spatial_size


_THREE_LEVELS = dict(feat_strides=(8, 16, 32), num_levels=3, num_points=(3, 6, 3),
                     return_idx=(1, 2, 3), dec_dim_feedforward=1024)

SPECS: Dict[str, Spec] = {
    "nano": Spec(
        config="configs/deim_dfine/deim_hgnetv2_n_coco.yml", backbone="B0", return_idx=(2, 3),
        use_lab=True, freeze_norm=False, enc_in_channels=(512, 1024), feat_strides=(16, 32),
        enc_hidden_dim=128, enc_dim_feedforward=512, expansion=0.34, depth_mult=0.5,
        dec_hidden_dim=128, dec_dim_feedforward=512, num_levels=2, num_layers=3,
        num_points=(6, 6), reg_scale=4, dec_act="silu"),
    "small": Spec(
        config="configs/deim_dfine/deim_hgnetv2_s_coco.yml", backbone="B0", use_lab=True,
        freeze_norm=False, enc_in_channels=(256, 512, 1024), enc_hidden_dim=256,
        enc_dim_feedforward=1024, expansion=0.5, depth_mult=0.34, dec_hidden_dim=256,
        num_layers=3, reg_scale=4, dec_act="silu", **_THREE_LEVELS),
    "medium": Spec(
        config="configs/deim_dfine/deim_hgnetv2_m_coco.yml", backbone="B2", use_lab=True,
        freeze_norm=False, enc_in_channels=(384, 768, 1536), enc_hidden_dim=256,
        enc_dim_feedforward=1024, expansion=1.0, depth_mult=0.67, dec_hidden_dim=256,
        num_layers=4, reg_scale=4, dec_act="silu", **_THREE_LEVELS),
    "large": Spec(
        config="configs/deim_dfine/deim_hgnetv2_l_coco.yml", backbone="B4", use_lab=False,
        freeze_norm=False, enc_in_channels=(512, 1024, 2048), enc_hidden_dim=256,
        enc_dim_feedforward=1024, expansion=1.0, depth_mult=1.0, dec_hidden_dim=256,
        num_layers=6, reg_scale=4, dec_act="silu", **_THREE_LEVELS),
    "xlarge": Spec(
        config="configs/deim_dfine/deim_hgnetv2_x_coco.yml", backbone="B5", use_lab=False,
        freeze_norm=False, enc_in_channels=(512, 1024, 2048), enc_hidden_dim=384,
        enc_dim_feedforward=2048, expansion=1.0, depth_mult=1.0, dec_hidden_dim=256,
        num_layers=6, reg_scale=8, dec_act="silu", **_THREE_LEVELS),
    # Pretrained on Objects365, fine-tuned 24 epochs on COCO. Its config differs from xlarge's in
    # exactly the two places a shape cannot show: frozen backbone norms and ReLU in the decoder.
    "xlarge-obj365": Spec(
        config="configs/deim_dfine/object365/deim_hgnetv2_x_obj2coco_24e.yml", backbone="B5",
        use_lab=False, freeze_norm=True, enc_in_channels=(512, 1024, 2048), enc_hidden_dim=384,
        enc_dim_feedforward=2048, expansion=1.0, depth_mult=1.0, dec_hidden_dim=256,
        num_layers=6, reg_scale=8, dec_act="relu", **_THREE_LEVELS),
}


def get_spec(variant: str) -> Spec:
    """The spec for a released variant, or a ``ValueError`` naming the ones there are."""
    try:
        return SPECS[variant]
    except KeyError:
        raise ValueError(f"unknown DEIM variant {variant!r}. Known: {list(SPECS)}") from None

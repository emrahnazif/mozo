"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2023 . All Rights Reserved.

The inference subset of upstream's ``engine/deim/utils.py``, ``engine/deim/dfine_utils.py``
and ``engine/deim/box_ops.py``. Only the code paths the released DEIM-D-FINE checkpoints take
are kept: ``method='default'`` deformable attention over a list of per-level values, and the
deploy-mode weighting function.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clip(min=0., max=1.)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))


def deformable_attention_core_func_v2(
    value,
    value_spatial_shapes,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    num_points_list: List[int],
):
    """
    Args:
        value (List[Tensor]): per level, [bs * ... ] as produced by ``TransformerDecoder.value_op``
        value_spatial_shapes (List): [n_levels, 2]
        sampling_locations (Tensor): [bs, query_length, n_head, n_levels * n_points, 2]
        attention_weights (Tensor): [bs, query_length, n_head, n_levels * n_points]

    Returns:
        output (Tensor): [bs, Length_{query}, C]
    """
    bs, n_head, c, _ = value[0].shape
    _, Len_q, _, _, _ = sampling_locations.shape

    sampling_grids = 2 * sampling_locations - 1

    sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
    sampling_locations_list = sampling_grids.split(num_points_list, dim=-2)

    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        value_l = value[level].reshape(bs * n_head, c, h, w)
        sampling_grid_l: torch.Tensor = sampling_locations_list[level]
        sampling_value_l = F.grid_sample(
            value_l,
            sampling_grid_l,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l)

    attn_weights = attention_weights.permute(0, 2, 1, 3).reshape(bs * n_head, 1, Len_q, sum(num_points_list))
    weighted_sample_locs = torch.concat(sampling_value_list, dim=-1) * attn_weights
    output = weighted_sample_locs.sum(-1).reshape(bs, n_head * c, Len_q)

    return output.permute(0, 2, 1)


def get_activation(act: str, inplace: bool = True) -> nn.Module:
    """The activations the released configs name: ``silu``, ``relu`` and ``gelu``."""
    if act is None:
        return nn.Identity()

    act = act.lower()
    if act == 'silu' or act == 'swish':
        m = nn.SiLU()
    elif act == 'relu':
        m = nn.ReLU()
    elif act == 'gelu':
        m = nn.GELU()
    else:
        raise ValueError(f'unsupported activation {act!r}')

    if hasattr(m, 'inplace'):
        m.inplace = inplace
    return m


def weighting_function(reg_max, up, reg_scale):
    """The non-uniform weighting function W(n), as upstream computes it in deploy mode.

    Deploy mode is the path ``DEIM.deploy()`` takes, and it is *not* the same arithmetic as the
    training path: the sequence is built from Python floats (double precision) and cast to the
    parameter dtype once, where the training path stays in float32 throughout. The two differ in
    the last bits, so this keeps the deploy form, which is what upstream's inference runs.
    """
    upper_bound1 = (abs(up[0]) * abs(reg_scale)).item()
    upper_bound2 = (abs(up[0]) * abs(reg_scale) * 2).item()
    step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
    left_values = [-(step) ** i + 1 for i in range(reg_max // 2 - 1, 0, -1)]
    right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
    values = [-upper_bound2] + left_values + [torch.zeros_like(up[0][None])] + right_values + [upper_bound2]
    return torch.tensor(values, dtype=up.dtype, device=up.device)


def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def distance2bbox(points, distance, reg_scale):
    """
    Decodes edge-distances into bounding box coordinates.

    Args:
        points (Tensor): (B, N, 4) or (N, 4) format, representing [x, y, w, h],
                         where (x, y) is the center and (w, h) are width and height.
        distance (Tensor): (B, N, 4) or (N, 4), representing distances from the
                           point to the left, top, right, and bottom boundaries.
        reg_scale (float): Controls the curvature of the Weighting Function.

    Returns:
        Tensor: Bounding boxes in (N, 4) or (B, N, 4) format [cx, cy, w, h].
    """
    reg_scale = abs(reg_scale)
    x1 = points[..., 0] - (0.5 * reg_scale + distance[..., 0]) * (points[..., 2] / reg_scale)
    y1 = points[..., 1] - (0.5 * reg_scale + distance[..., 1]) * (points[..., 3] / reg_scale)
    x2 = points[..., 0] + (0.5 * reg_scale + distance[..., 2]) * (points[..., 2] / reg_scale)
    y2 = points[..., 1] + (0.5 * reg_scale + distance[..., 3]) * (points[..., 3] / reg_scale)

    bboxes = torch.stack([x1, y1, x2, y2], -1)

    return box_xyxy_to_cxcywh(bboxes)

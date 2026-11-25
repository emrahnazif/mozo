"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.

Upstream's ``engine/deim/postprocessor.py`` in deploy mode (``PostProcessor.deploy()``), which is
what ``tools/inference/torch_inf.py`` runs: focal-loss scoring, top-k over every (query, class)
pair, and class ids left contiguous (0-79) rather than remapped to COCO's sparse category ids,
which upstream does only for its COCO evaluator.
"""

import torch
import torch.nn.functional as F
import torchvision


def mod(a, b):
    out = a - a // b * b
    return out


def postprocess(outputs, orig_target_sizes: torch.Tensor, num_classes: int = 80, num_top_queries: int = 300):
    """Turn raw logits and boxes into the top ``num_top_queries`` detections per image.

    Args:
        outputs: ``{"pred_logits": (B, Q, C), "pred_boxes": (B, Q, 4)}`` with boxes as normalised
            ``cxcywh``.
        orig_target_sizes: ``(B, 2)`` as ``(width, height)`` of each original image.

    Returns:
        ``(labels, boxes, scores)``, each ``(B, num_top_queries, ...)``, boxes as ``xyxy`` pixels
        in the original image, sorted by score, not thresholded.
    """
    logits, boxes = outputs['pred_logits'], outputs['pred_boxes']

    bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
    bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

    scores = F.sigmoid(logits)
    scores, index = torch.topk(scores.flatten(1), num_top_queries, dim=-1)
    labels = mod(index, num_classes)
    index = index // num_classes
    boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))

    return labels, boxes, scores

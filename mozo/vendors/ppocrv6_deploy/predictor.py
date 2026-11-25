# SPDX-License-Identifier: Apache-2.0
# The seam follows PaddleX 3.7.2, paddlex/inference/pipelines/ocr/pipeline.py (_OCRPipeline.predict)
# with the document preprocessor and the text-line orientation classifier off.
# Copyright (c) 2024-2026 PaddlePaddle Authors. Apache-2.0.
"""The seam: one page in, located and read lines out."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch

from . import boxes as _boxes
from . import image as _image
from .checkpoint import load
from .config import Spec
from .text import Alphabet

#: ``OCR.yaml``'s TextRecognition ``batch_size``: lines are read six at a time, shortest first.
#: Batching is part of the arithmetic -- each batch is zero-padded to its widest line.
REC_BATCH = 6


@dataclass(frozen=True)
class Region:
    """One line: its quadrilateral in page pixels (int, clockwise from top-left), the string,
    and the recogniser's confidence (mean of the kept CTC steps' probabilities)."""

    quad: np.ndarray
    text: str
    confidence: float


def _torch_call(module: torch.nn.Module, device: str) -> Callable[[np.ndarray], np.ndarray]:
    # Captures the module and the device only: a closure over the Reader would be a reference
    # cycle, and the weights would wait for the cyclic collector after the Reader is dropped.
    def run(batch: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return module(torch.from_numpy(batch).to(device)).float().cpu().numpy()
    return run


class Reader:
    """Detect and read text, as PaddleOCR 3.7.0 does with PP-OCRv6.

    ``detect`` and ``recognise`` are the two networks as callables on numpy batches, so a graph
    runtime can stand in for torch without the pipeline around them changing. By default both
    are the torch modules built from ``checkpoint``.
    """

    def __init__(self, checkpoint: str | Path | None, spec: Spec, device: str = "cpu", *,
                 detect: Callable[[np.ndarray], np.ndarray] | None = None,
                 recognise: Callable[[np.ndarray], np.ndarray] | None = None) -> None:
        self.spec = spec
        self.device = device
        self.alphabet = Alphabet(spec.recogniser.charset)
        if detect is None or recognise is None:
            detector, recogniser = load(checkpoint, spec)
            detect = detect or _torch_call(detector.to(device), device)
            recognise = recognise or _torch_call(recogniser.to(device), device)
        self.detect = detect
        self.recognise = recognise

    def find(self, bgr: np.ndarray, box_thresh: float = _boxes.BOX_THRESH) -> list[np.ndarray]:
        """Detected quads, sorted into upstream's reading order."""
        page, shape = _image.resize_for_detector(bgr)
        batch = _image.normalize_for_detector(page)[None].astype(np.float32, copy=False)
        prob = self.detect(np.ascontiguousarray(batch))
        found, _scores = _boxes.quads(prob[0][0], shape, box_thresh)
        return _boxes.sort_quads(found) if len(found) else []

    def read(self, bgr: np.ndarray, quads: list[np.ndarray]) -> list[Region]:
        """Crop each quad, read the crops in batches sorted by aspect, return them in quad order."""
        crops, kept = [], []
        for quad in quads:
            crop = _image.crop_line(bgr, quad)
            if crop.size:
                crops.append(crop)
                kept.append(quad)
        order = sorted(range(len(crops)), key=lambda i: crops[i].shape[1] / float(crops[i].shape[0]))
        reads: dict[int, tuple[str, float]] = {}
        for start in range(0, len(order), REC_BATCH):
            chunk = order[start:start + REC_BATCH]
            batch = _image.batch_for_recogniser(
                [_image.resize_for_recogniser(crops[i]) for i in chunk])
            for i, result in zip(chunk, self.alphabet.decode(self.recognise(batch))):
                reads[i] = result
        return [Region(np.asarray(quad), *reads[i]) for i, quad in enumerate(kept)]

    def __call__(self, rgb: np.ndarray, box_thresh: float = _boxes.BOX_THRESH) -> list[Region]:
        """Read an ``HxWx3`` RGB ``uint8`` page."""
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return self.read(bgr, self.find(bgr, box_thresh))

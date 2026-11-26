# SPDX-License-Identifier: Apache-2.0
"""Deployment-only Mayaku detection and instance segmentation.

Extracted from ``mayaku`` 3.1.1 and reduced to the inference path of its published checkpoints,
one per variant in :data:`SPECS`. The checkpoint is the publisher's own file; it loads in its
trained form and is fused at load with upstream's own arithmetic into the five-op graph (Conv,
Relu, Add, nearest Resize, MaxPool). Everything after the graph -- the prediction decode and the
per-instance mask network -- is shared by every runtime, and runs where upstream runs it: on the
GPU when the graph's maps are on CUDA, else on the CPU. See PROVENANCE.md.
"""

from .model import SPECS, Detections, Detector, Spec, detect, get_spec
from .network import output_names

__all__ = ["SPECS", "Detections", "Detector", "Spec", "detect", "get_spec", "output_names"]

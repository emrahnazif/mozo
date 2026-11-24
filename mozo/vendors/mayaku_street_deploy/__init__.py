# SPDX-License-Identifier: Apache-2.0
"""Deployment-only Mayaku street detection: 22 road-scene classes, boxes and instance masks.

Extracted from ``mayaku`` 3.1.1 and reduced to the street models' inference path. The checkpoint is
the publisher's own file; it loads in its trained form and is fused at load with upstream's own
arithmetic into the five-op graph (Conv, Relu, Add, nearest Resize, MaxPool). Everything after the
graph -- the prediction decode and the per-instance mask network -- runs on the CPU and is shared
by every runtime. See PROVENANCE.md.
"""

from .model import SPECS, Detections, Detector, Spec, detect, get_spec
from .network import output_names

__all__ = ["SPECS", "Detections", "Detector", "Spec", "detect", "get_spec", "output_names"]

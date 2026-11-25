"""Depth Anything 3 relative depth from one image, on whichever runtime the host can execute.

The network lives in :mod:`mozo.vendors.depth_anything_v3_deploy`, extracted from the authors'
repository and reduced to single-image depth. Pre-processing and the sky fill come from the
vendor, verified bit-identical to upstream. Two things happen here that upstream does not do:

* **The depth is resized to the input's size.** Upstream returns it at the resolution the network
  ran at (504 px on the long side). mozo returns every depth map at the input's resolution, with
  the bilinear resize Depth Anything V2 uses.
* **It is inverted.** Depth Anything 3 predicts depth -- larger means farther -- on a scale of its
  own. mozo's unitless depth maps are inverse depth, larger meaning nearer, so a workflow can swap
  one relative-depth model for another. ``1 / predict(image)`` gives the depth back.

Three variants. ``small`` and ``base`` are upstream's "any-view" models (DA3-SMALL, DA3-BASE),
trained for one or many views and run here on one; ``mono-large`` is the monocular model
(DA3MONO-LARGE), which also segments the sky and sets it to the far end of the scene.

Only the network changes between runtimes. A graph runtime (ONNX, TensorRT) takes the pixels and
the tensors the network derives from the input's shape, which the vendor computes here from the
published position embedding, exactly as the torch path computes them for itself.

    >>> model = DepthAnythingV3Predictor("small")              # doctest: +SKIP
    >>> inverse_depth = model.predict(image)                   # doctest: +SKIP
    >>> model.unit is None                                     # doctest: +SKIP
    True
"""

from __future__ import annotations

import functools
import logging
from typing import Union

import numpy as np
import torch

from ..image import load_image
from ..registry import DERIVES, EXECUTES
from ..runtimes import TensorRTRunner, get_default_device, make_runner, select_runtime
from ..vendors.depth_anything_v3_deploy import (MODEL_SPECS, Predictor, postprocess, preprocess,
                                                resize, shape_inputs)
from ..vendors.depth_anything_v3_deploy.config import PATCH_SIZE
from ..vendors.depth_anything_v3_deploy.graph import MIN_PATCHES, profile
from ..weights import artifacts, framework_of, parts, resolve

_log = logging.getLogger(__name__)


class DepthAnythingV3Predictor:
    """One loaded Depth Anything 3 variant, ready to run.

    Args:
        variant: ``small``, ``base`` or ``mono-large``.
        device: Where to run. Defaults to the best device this machine has.
        runtime: Which published runtime to execute; ``"auto"`` takes the best one for the device.
        revision: Pin a published revision instead of taking the latest.

    Attributes:
        variant: The variant in use.
        runtime: The runtime actually in use.
        device: The device actually in use.
        unit: Always ``None``: the output is inverse depth on an arbitrary per-image scale.
    """

    VARIANTS = tuple(MODEL_SPECS)
    FAMILY = "depth_anything_v3"
    unit = None

    def __init__(
        self,
        variant: str = "small",
        device: str | None = None,
        *,
        runtime: str = "auto",
        revision: str | None = None,
    ) -> None:
        if variant not in self.VARIANTS:
            raise ValueError(f"Unsupported variant {variant!r}. Choose from: {list(self.VARIANTS)}")

        self.variant = variant
        self.device = device or get_default_device()
        self._spec = MODEL_SPECS[variant]
        self.runtime = select_runtime(
            self.device, artifacts(self.FAMILY, variant, revision=revision), runtime,
            executes=EXECUTES.get(self.FAMILY), derives=DERIVES.get(self.FAMILY))

        self._predictor = self._runner = None
        framework = framework_of(self.runtime)
        if framework == "torch":
            weights = resolve(self.FAMILY, variant, self.runtime, revision=revision)
            self._predictor = Predictor.from_pretrained(variant, weights=weights,
                                                        device=self.device)
        else:
            # The graph carries the network; the shape inputs need only the learned position
            # embedding, published on its own so a graph runtime never downloads the checkpoint.
            pos_embed = torch.load(resolve(self.FAMILY, variant, "pos_embed", revision=revision),
                                   map_location="cpu", weights_only=True)
            if framework == "tensorrt":
                # Built once per GPU and TensorRT version from the graph it derives from, for every
                # shape the network can be fed: 3 to 36 patches a side.
                self._runner = TensorRTRunner(
                    parts(self.FAMILY, variant, self.runtime, revision=revision)[""],
                    self.device, shapes=profile(self._spec))
                self._graph_device = self._runner.device
            else:
                self._runner = make_runner(
                    resolve(self.FAMILY, variant, self.runtime, revision=revision),
                    self.runtime, device=self.device)
                self._graph_device = torch.device("cpu")
            # The shape inputs depend on the shape alone, and computing them costs 5-11 ms on a
            # GPU and 0.2 s on a CPU per call -- as much as an fp16 engine's whole forward. The
            # last two shapes are kept (up to 73 MiB each, for base): a video, a camera or a folder
            # of one camera's photographs repeats one shape. The closure holds only what it needs.
            spec, device = self._spec, self._graph_device
            pos_embed = pos_embed.to(device)
            self._shape_inputs = functools.lru_cache(maxsize=2)(
                lambda height, width: tuple(shape_inputs(spec, pos_embed, height, width, device)))
        _log.info("Depth Anything 3 %s ready on %s via %s.", variant, self.device, self.runtime)

    def _depth(self, rgb: np.ndarray) -> torch.Tensor:
        """Depth at the processed resolution -- upstream's answer -- from the loaded runtime."""
        if self._predictor is not None:
            return self._predictor.predict(rgb)
        pixels = preprocess(rgb, self._spec.process_res)[:, 0]
        height, width = pixels.shape[-2:]
        if min(height, width) < MIN_PATCHES * PATCH_SIZE:
            raise ValueError(
                f"a {rgb.shape[1]}x{rgb.shape[0]} image is too elongated for the {self.runtime} "
                f"graph: its short side would be {min(height, width)} px, and the graph takes "
                f"{MIN_PATCHES * PATCH_SIZE} or more (an aspect ratio up to 12:1). The torch-fp32 "
                "runtime takes it.")
        shape = self._shape_inputs(height, width)
        if isinstance(self._runner, TensorRTRunner):
            outputs = self._runner(pixels.to(self._graph_device), *shape)
        else:
            extra = {name: t.numpy() for name, t in zip(self._runner.inputs[1:], shape)}
            outputs = [torch.from_numpy(a) for a in self._runner(pixels.numpy(), **extra)]
        # Each output is 1 x H' x W'; the post-process reads them as upstream's 1 x 1 x H' x W'.
        return postprocess({name: out[:, None] for name, out in
                            zip(self._runner.outputs, outputs)}, self._spec.sky)

    @torch.inference_mode()
    def predict(self, image: Union[str, bytes, np.ndarray]) -> np.ndarray:
        """Estimate inverse depth for *image*.

        Args:
            image: A file path, encoded bytes, or an ``HWC`` RGB array.

        Returns:
            An ``HxW`` float32 array at the input's resolution: inverse depth on an arbitrary
            per-image scale, larger meaning nearer. No value is a distance.

        Raises:
            ValueError: If the image is too elongated for the network: beyond about 72:1 on
                torch, 12:1 on a graph runtime.
        """
        rgb = load_image(image)
        depth = self._depth(rgb)
        return (1.0 / resize(depth.float(), rgb.shape[:2])).cpu().numpy()

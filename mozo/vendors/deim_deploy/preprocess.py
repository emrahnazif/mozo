"""Image to network input, as upstream's ``tools/inference/torch_inf.py`` prepares it.

Upstream runs ``T.Compose([T.Resize((640, 640)), T.ToTensor()])`` on a PIL image: a squash (not
a letterbox) to the square evaluation size with PIL's bilinear filter, then ``uint8 / 255``. No
mean or standard deviation is subtracted. The COCO evaluation transform (``Resize`` then
``ConvertPILImage(scale=True)``) is the same arithmetic, so this is also what the published AP
numbers were measured on.

Two traps live here. PIL's bilinear filter widens its support with the downscale factor, so it is
neither ``cv2.resize`` nor ``F.interpolate`` and has to be PIL. And upstream's ``onnx_inf.py``
letterboxes instead -- a different picture; the torch script and the evaluator are the reference.
"""

from typing import Tuple

import numpy as np
import torch
from PIL import Image


def preprocess(image: np.ndarray, size: Tuple[int, int] = (640, 640)) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resize an RGB image for the network.

    Args:
        image: ``(H, W, 3)`` uint8 RGB.
        size: ``(height, width)`` the network was built for.

    Returns:
        ``(batch, orig_size)``: a ``(1, 3, height, width)`` float32 tensor in ``[0, 1]``, and a
        ``(1, 2)`` int64 tensor of the original ``(width, height)``, which is what the
        post-processor scales boxes by.
    """
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) uint8 RGB array, got {image.dtype} {image.shape}")
    h, w = image.shape[:2]
    resized = Image.fromarray(image).resize((size[1], size[0]), Image.BILINEAR)
    # torchvision's ``to_tensor`` for a PIL image: HWC uint8 -> CHW, then ``.float().div(255)``.
    array = np.array(resized, np.uint8, copy=True)
    batch = torch.from_numpy(array).permute(2, 0, 1).contiguous().to(dtype=torch.float32).div(255)
    return batch.unsqueeze(0), torch.tensor([[w, h]])

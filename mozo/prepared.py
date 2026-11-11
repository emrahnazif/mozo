"""Images a server was asked to keep, so a click can name one instead of uploading it again.

Not the model's cache. Each model that can be prepared already keeps the *encoder output* for its
last few images, and a click with the file attached finds it there. What a click cannot do over
HTTP is arrive without the file: the model's ``predict`` takes pixels. So ``/prepare`` -- and any
click that sends its file together with ``?sha256=`` -- leaves the decoded image here, under the
sha256 of the file's bytes, and a later ``/predict?sha256=...`` with no file reads it back. A
megabyte upload and a decode per click become a few hundred bytes.

The key is the file's own hash, so every worker, every restart and the client itself compute the
same one, and a wrong key can only miss -- it cannot name a different image. A miss is answered
plainly, and the client sends the file again with the same ``?sha256=``, which stores it.

Only requests that ask are kept. An ordinary upload is not: it gains nothing from being kept --
it brings its file every time -- and a script sending ten thousand photos through would push out
the images people clicking actually prepared.

Bounded by bytes, not count, because one 4K scan weighs as much as forty thumbnails: the least
recently used image goes first. ``MOZO_PREPARED_MB`` sets the bound, 512 MB by default.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from threading import Lock

import numpy as np

__all__ = ["PreparedImages", "limit_from_environment"]

_DEFAULT_MB = 512


def limit_from_environment() -> int:
    """The store's bound in bytes: ``MOZO_PREPARED_MB`` if set, else 512 MB."""
    text = os.environ.get("MOZO_PREPARED_MB", "").strip()
    return int(float(text) * 1e6) if text else _DEFAULT_MB * 1_000_000


class PreparedImages:
    """Decoded images by the sha256 of the file they came from, least recently used out first.

    Thread-safe: the server's threadpool reads and writes it from many requests at once.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._images: OrderedDict[str, np.ndarray] = OrderedDict()
        self._bytes = 0
        self._lock = Lock()

    def put(self, sha256: str, pixels: np.ndarray) -> None:
        """Keep *pixels* under *sha256*, evicting the least recently used to stay in bounds.

        The newest image is always kept, even one larger than the whole bound: refusing it
        would make ``/prepare`` say yes and then miss on the very next click.
        """
        with self._lock:
            if sha256 in self._images:
                self._images.move_to_end(sha256)
                return
            self._images[sha256] = pixels
            self._bytes += pixels.nbytes
            while self._bytes > self.limit and len(self._images) > 1:
                _, dropped = self._images.popitem(last=False)
                self._bytes -= dropped.nbytes

    def get(self, sha256: str) -> np.ndarray | None:
        """The image kept under *sha256*, or ``None`` if it was never kept or has gone."""
        with self._lock:
            pixels = self._images.get(sha256)
            if pixels is not None:
                self._images.move_to_end(sha256)
            return pixels

    def __len__(self) -> int:
        return len(self._images)

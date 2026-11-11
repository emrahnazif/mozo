"""``prepare(image)``: a prompted model's image half, run before the prompt arrives.

SAM 2, EdgeTAM, SAM 3 and OWLv2 split their cost the same way: an expensive pass that depends
only on the image, and a cheap one per prompt. Each vendor already caches the first, keyed on
the pixels, so a second click on one picture skips it. What none could do is run it *before*
the first click: the encode happened inside ``predict``, and ``predict`` needs a prompt. So an
annotator opening an image paid the whole encode -- half a second for SAM 2 large on Apple
silicon -- on the click, where it shows.

``prepare`` runs that encode now and leaves the result in the vendor's cache, where the next
``predict`` finds it. It promises nothing more than that: the caches are small (five images, or
two for SAM 3), so preparing more than they hold quietly drops the oldest. That costs time, never
a wrong answer.

The one thing it must also do is the reason it exists: an image is opened, ``prepare`` starts in
the background, and the user clicks before it finishes. That click misses the cache, because the
encode is not done, and without help it would start a second encode of the same image -- two
encodes fighting for the GPU, and a click slower than if nothing had been prepared at all. So an
encode under way is recorded here, and a ``predict`` for that image waits for it instead.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from threading import Event, Lock
from typing import Union

import numpy as np

from ..image import load_image

__all__ = ["Preparing"]

#: Guards the moment an adapter's table of encodes is first made, so two threads asking at once
#: cannot each make one. Held for that instant only, never across an encode.
_MAKING = Lock()


def _key(pixels: np.ndarray) -> bytes:
    """The same content key the vendors' caches use: sha256 over the pixels, not a sample."""
    return hashlib.sha256(np.ascontiguousarray(pixels)).digest()


class Preparing:
    """For an adapter whose vendor encodes an image once and caches it.

    The adapter supplies :meth:`_encode` -- its vendor's public encode step -- and calls
    :meth:`_after_prepare` with the pixels before its vendor's ``predict``. Nothing to set up in
    its constructor: the table of encodes under way is made the first time ``prepare`` runs, so
    an adapter built any way at all -- a subclass, a test's stand-in -- behaves the same.
    """

    def _encode(self, pixels: np.ndarray) -> None:
        """Run the vendor's image encode, which caches its result. Each adapter names it."""
        raise NotImplementedError

    def _encodes(self) -> tuple[dict[bytes, Event], Lock]:
        """Content key -> set when that image's encode, started by ``prepare``, is over; and the
        lock guarding it. Made on first use."""
        made = self.__dict__.get("_encoding")
        if made is None:
            with _MAKING:
                made = self.__dict__.setdefault("_encoding", ({}, Lock()))
        return made

    def prepare(self, image: Union[str, Path, bytes, np.ndarray]) -> None:
        """Encode *image* now, so the next prompt on it runs only the cheap half.

        Call it when the image is opened -- on a background thread, if the caller has one -- and
        the first click costs what every later click does. Returns at once if the image is
        already encoded. Asked for the same image twice at once, it encodes once.

        It warms a cache rather than reserving a place in it: the cache holds a few images per
        model, and preparing more than that drops the oldest. A dropped image is encoded again
        when next asked for -- slower, never wrong.

        Args:
            image: A file path, encoded image bytes, or an ``HxWx3`` RGB ``uint8`` array -- the
                same as ``predict`` takes. Pass the array you will predict on, so neither call
                decodes the file again.
        """
        under_way_by_key, lock = self._encodes()
        pixels = load_image(image)
        key = _key(pixels)
        with lock:
            under_way = under_way_by_key.get(key)
            if under_way is None:
                under_way_by_key[key] = mine = Event()
        if under_way is not None:
            under_way.wait()
            return
        try:
            self._encode(pixels)
        finally:
            with lock:
                del under_way_by_key[key]
            mine.set()

    def _after_prepare(self, pixels: np.ndarray) -> None:
        """If this image is being prepared right now, wait for that encode rather than repeat it.

        Free in the ordinary case: nothing is being prepared, so nothing is hashed. Only while a
        :meth:`prepare` is under way does a ``predict`` pay one extra hash to see if it is for
        this image. An adapter ``prepare`` never ran on has no table, and predicts exactly as it
        did before ``prepare`` existed.
        """
        made = self.__dict__.get("_encoding")
        if made is None or not made[0]:
            return
        under_way = made[0].get(_key(pixels))
        if under_way is not None:
            under_way.wait()

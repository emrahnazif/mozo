# SPDX-License-Identifier: Apache-2.0
"""mozo's Granite Embedding R2 adapter: text in, one vector per text out.

mozo's first text-only family. It answers nothing about an image, so there is no ``predict``:
``encode_text`` is the whole of it, and over HTTP that is ``/encode`` (``/predict`` refuses it
before loading anything). The vectors are a handoff, not an answer -- useful only next to other
vectors, which means a store mozo does not provide. The usual shape: split documents into
chunks, embed each chunk once and keep the vectors in a vector database, then embed a question
and let the database find the nearest chunks. mozo is the model in that pipeline and nothing else.

**Text only, and not CLIP's space.** Granite was never trained alongside images, so its vectors
mean nothing against CLIP's or SigLIP 2's, and it cannot search pictures by words. It can search
what was *read* off pictures: EasyOCR's text, embedded here, is searchable by meaning.

**Chunking is yours, and too long is refused.** Up to 8,192 tokens a text, ``[CLS]`` and
``[SEP]`` included. Anything longer raises rather than being cut short silently: a chunk
shortened without saying so is a different chunk.

**An index is tied to the weights that built it.** A vector from one variant or revision means
nothing against a vector from another, so a stored index has to record :attr:`revision`. Not the
runtime: torch and CoreML agree to 1.8e-07, the size of two attention kernels disagreeing.

**On Apple silicon it runs as a CoreML graph**, when ``coremltools`` is installed. CoreML runs a
fixed shape on the GPU and a flexible one only on the CPU, so the package holds the network at a
few fixed lengths, sharing one set of weights, and a text is padded to the smallest that fits and
masked (:mod:`~mozo.vendors.granite_embedding_r2_deploy.graph`). 0.27-0.67x torch's time on MPS
where a text fills its length; 1.0-1.34x where it only just spills into the next one.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from threading import Lock
from typing import Union

import numpy as np

from .. import registry
from ..runtimes import CoreMLRunner, coreml_functions, get_default_device, select_runtime
from ..vendors.granite_embedding_r2_deploy import CONTEXT, SPECS, VARIANTS, Encoder
from ..vendors.granite_embedding_r2_deploy.graph import padded
from ..vendors.granite_embedding_r2_deploy.tokenizer import Tokenizer
from ..weights import artifacts, resolve, revision_of

__all__ = ["GraniteEmbeddingR2Predictor"]

#: The family this adapter serves, as the registry and the manifest name it.
FAMILY = "granite_embedding_r2"

_log = logging.getLogger(__name__)


class GraniteEmbeddingR2Predictor:
    """English text embeddings.

    Args:
        variant: Which published model. ``small`` is the one carried.
        device: Where to run. Defaults to the best available.
        runtime: Which artifact to execute. ``auto`` picks the fastest published one.
        checkpoint_path: Your own checkpoint, instead of the published weights.
        revision: Which published revision to use. Defaults to the newest.

    Attributes:
        variant: The variant in use.
        device: Where it runs.
        runtime: Which artifact is executing.
        revision: The published revision in use, or ``None`` for your own checkpoint.

    Examples:
        >>> model = GraniteEmbeddingR2Predictor()                           # doctest: +SKIP
        >>> model.encode_text(["how do I run a workflow?"]).shape           # doctest: +SKIP
        (1, 384)
    """

    #: The variants carried. The vendor's ``config.py`` says why only the small one.
    VARIANTS = list(VARIANTS)

    #: Which runtimes this adapter can execute: torch, and the CoreML graph on Apple silicon.
    EXECUTES = registry.EXECUTES[FAMILY]

    def __init__(
        self,
        variant: str = "small",
        device: str | None = None,
        *,
        runtime: str = "auto",
        checkpoint_path: str | Path | None = None,
        revision: str | None = None,
    ) -> None:
        # Checked even with your own checkpoint: the variant selects the geometry, so a wrong one
        # is not a mislabelling but a strict load that cannot succeed.
        if variant not in self.VARIANTS:
            raise ValueError(f"Unsupported variant {variant!r}. Choose from: {list(self.VARIANTS)}")

        self.variant = variant
        self.device = device or get_default_device()
        self.runtime = (
            "torch-fp32" if checkpoint_path is not None
            else select_runtime(
                self.device,
                artifacts(FAMILY, variant, revision=revision),
                runtime,
                executes=self.EXECUTES,
            )
        )
        weights = (Path(checkpoint_path) if checkpoint_path
                   else resolve(FAMILY, variant, self.runtime, revision=revision))
        #: Which published revision produced the vectors, or ``None`` for your own checkpoint. A
        #: vector is only comparable against others from the same weights, so an index built from
        #: these has to record it -- which means the model has to be able to say.
        self.revision = None if checkpoint_path else revision_of(FAMILY, variant, revision=revision)
        # No network is built here: torch builds it on the first encode, CoreML each length on the
        # first text that needs it.
        self._encoder = (_Graph(weights) if self.runtime.startswith("coreml")
                         else Encoder(weights, SPECS[variant], device=self.device))
        _log.info("Granite Embedding R2 %s ready on %s via %s.", variant, self.device, self.runtime)

    def encode_text(self, text: Union[str, Sequence[str]]) -> "np.ndarray":
        """Return ``(N, 384)`` L2-normalised vectors for one text or many, in the order given.

        Always two-dimensional, including for a single text. A dot product between two of them is
        their cosine similarity: compare them against each other, or against a threshold you
        calibrate on your own data, never against an intuition about percentages.

        Raises:
            ValueError: If no text is given, if one is blank, if one is over 8,192 tokens, or if
                one contains an unpaired surrogate.
        """
        return np.asarray(self._encoder.encode_text(text))


class _Graph:
    """The CoreML package: the encoder at fixed lengths, one function per length.

    Each length is loaded the first time a text needs it -- about a second each -- so a server
    that only ever sees short queries never loads the 8,192-token graph. The lengths are read off
    the package's own input shapes, so nothing here has to agree with how the export named them.
    """

    def __init__(self, package: Path) -> None:
        self._package = package
        # Built here rather than on first use: 70 ms of tables, against ~1 s for each graph.
        self._tokenizer = Tokenizer(CONTEXT)
        self._lengths = {shape[1]: name for name, shape in coreml_functions(package).items()}
        self._runners: dict[int, CoreMLRunner] = {}
        self._lock = Lock()

    def _runner(self, tokens: int) -> CoreMLRunner:
        """The function for the shortest length that holds *tokens*, loaded once."""
        length = min(n for n in self._lengths if n >= tokens)
        runner = self._runners.get(length)
        if runner is None:
            with self._lock:
                runner = self._runners.get(length)
                if runner is None:
                    runner = CoreMLRunner(self._package, function=self._lengths[length])
                    self._runners[length] = runner
        return runner

    def encode_text(self, texts: str | Sequence[str]) -> np.ndarray:
        """``(N, width)`` unit vectors, as the torch encoder returns them. One text per call."""
        out = []
        for row in self._tokenizer(texts):
            runner = self._runner(len(row))
            out.append(runner(padded(row, runner.input_shape[1]),
                              length=np.array([len(row)], dtype=np.int32))[0])
        return np.concatenate(out)

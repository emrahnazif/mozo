# SPDX-License-Identifier: Apache-2.0
"""mozo's SAM 3 adapter: a phrase or a click and an image in, PixelFlow detections out.

Every other family in mozo answers a fixed question -- "where are the 80 COCO classes", "how far
away is each pixel". SAM 3 answers whatever you ask it, so its ``predict`` takes a *prompt*, and
that one difference propagates: the caller supplies the vocabulary, one concept or several.

That fits mozo's rule that a class name comes from the weights or from the user rather than being
invented. Here it comes from the user, literally -- the phrase you searched for is the name every
detection carries. There is no class list to resolve, so this adapter does not consult
:mod:`mozo.labels` at all.

It also answers a click the way SAM 2 does -- points and boxes in, the one thing pointed at out,
in the same shape SAM 2 returns -- from SAM 3's own click head and the same image encode the
phrase used.

**The weights are not Apache-2.0.** SAM 3's checkpoints carry Meta's SAM License, which restricts
what they may be used for and binds whoever you serve predictions to. See the NOTICE published
beside them.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch

from ..image import load_image
from ..runtimes import CoreMLRunner, get_default_device, select_runtime
from ..vendors.sam3_deploy import Segmenter
from ..weights import artifacts, framework_of, resolve
from ..vendors.sam3_deploy.tracker.session import split_prompts
from ._prepare import Preparing
from ._promptable import bounds, candidates, names_for

try:
    import pixelflow as pf
except ImportError:
    raise ImportError("PixelFlow is not installed. Install it with: pip install pixelflow") from None

__all__ = ["CLICK_LEVELS", "GraphVision", "LEVELS", "Sam3Predictor", "Sam3Tracker"]

#: The family this adapter serves, as the registry and the manifest name it.
FAMILY = "sam3"

#: The concept pyramid's levels, finest first, as the exported graph names its outputs. The
#: order is the contract: :class:`~mozo.vendors.sam3_deploy.grounding.concept.ConceptHead` reads
#: ``levels[-1]`` as the grid it attends over and the rest as mask features, so a graph that
#: returned them the other way round would segment confidently and wrongly.
LEVELS = ("level0", "level1", "level2")

#: The graph's names for the click pyramid's levels, finest first, in the same order as
#: :data:`LEVELS`. A graph exported before the click half was added has none of them.
CLICK_LEVELS = ("click0", "click1", "click2")


_log = logging.getLogger(__name__)


def _phrases(text: Union[str, Sequence[str]]) -> list[str]:
    """One phrase or several, as a list; refused if empty. SAM 3 will happily encode the empty
    string and return whatever it finds most salient, which is not what an empty prompt means."""
    phrases = [text] if isinstance(text, str) else list(text)
    if not phrases:
        raise ValueError("SAM 3 needs a concept to look for; no text was given.")
    if any(not p.strip() for p in phrases):
        raise ValueError("SAM 3 needs a concept to look for; text was empty.")
    return phrases


class GraphVision:
    """A graph artifact standing in for SAM 3's torch vision encoder.

    :class:`~mozo.vendors.sam3_deploy.predictor.Segmenter` asks its encoder one question -- a
    preprocessed batch in, the requested pyramids and the position encoding out -- and this
    answers it from a CoreML package instead of from the trunk. That is the whole of what the
    graph replaces; the text tower, the concept head and the click head stay in torch, so the
    checkpoint is still loaded either way.

    A graph carries the concept pyramid, and the click pyramid as well when it was exported with
    both halves of the neck. :attr:`stacks` says which, read off the package's own outputs, so a
    graph published before the click half was added still loads -- and serves phrases only.

    Args:
        runner: The loaded package.
        device: Where the torch half runs, and so where the outputs must land.

    Attributes:
        stacks: The pyramids this graph can return.

    Raises:
        ValueError: If asked for a pyramid the graph does not carry. Returning another one in its
            place is the difference between an error and a mask of the wrong pixels.
    """

    def __init__(self, runner: CoreMLRunner, device: str) -> None:
        self.runner = runner
        self.device = device
        self.stacks = (("concept", "click") if set(CLICK_LEVELS) <= set(runner.outputs)
                       else ("concept",))

    def __call__(self, batch: torch.Tensor, stacks: tuple[str, ...] = ("concept",)) -> dict:
        missing = set(stacks) - set(self.stacks)
        if missing:
            raise ValueError(
                f"this SAM 3 graph carries the {' and '.join(self.stacks)} stack(s), "
                f"not {sorted(missing)!r}"
            )
        got = dict(zip(self.runner.outputs, self.runner(batch.cpu().numpy())))
        found = {
            "concept": [torch.from_numpy(got[name]).to(self.device) for name in LEVELS],
            "positions": torch.from_numpy(got["positions"]).to(self.device),
        }
        if "click" in stacks:
            found["click"] = [torch.from_numpy(got[name]).to(self.device) for name in CLICK_LEVELS]
        return found


class Sam3Predictor(Preparing):
    """Concept segmentation -- name a thing, get every instance of it -- and clicks.

    Args:
        variant: Which published model. Meta ships one, so there is one name.
        device: Where to run. Defaults to the best available.
        runtime: Which artifact to execute. ``auto`` picks the fastest published one.
        checkpoint_path: Your own checkpoint, instead of the published weights.
        revision: Which published revision to use. Defaults to the newest.

    Attributes:
        variant: The variant in use.
        device: Where it runs.
        runtime: Which artifact is executing.

    Examples:
        >>> model = Sam3Predictor()                            # doctest: +SKIP
        >>> found = model.predict("street.jpg", "taxi")        # doctest: +SKIP
        >>> found[0].class_name                                # doctest: +SKIP
        'taxi'
        >>> many = model.predict("street.jpg", ["taxi", "cyclist"])   # doctest: +SKIP
        >>> one = model.predict("street.jpg", points=[[820, 640]], labels=[1])   # doctest: +SKIP
    """

    #: Meta publishes a single SAM 3 rather than a size ladder, so there is no choice to make.
    VARIANTS = ["sam3"]

    def __init__(
        self,
        variant: str = "sam3",
        device: str | None = None,
        *,
        runtime: str = "auto",
        checkpoint_path: str | Path | None = None,
        revision: str | None = None,
    ) -> None:
        # Checked even with your own checkpoint: unlike families where the variant selects an
        # architecture, here it names a published model, and a wrong one would be reported back
        # by ``self.variant`` as though it meant something.
        if variant not in self.VARIANTS:
            raise ValueError(f"Unsupported variant {variant!r}. Choose from: {list(self.VARIANTS)}")

        self.variant = variant
        self.device = device or get_default_device()
        # What is runnable is the manifest's to declare. ``select_runtime`` reads it and raises
        # with the available names, so there is nothing for this adapter to restate.
        self.runtime = (
            "torch-fp32" if checkpoint_path is not None
            else select_runtime(self.device, artifacts(FAMILY, variant, revision=revision), runtime)
        )

        # The graph covers the vision encoder and nothing else, so the checkpoint is resolved
        # whichever runtime won -- the text tower and the concept head come out of it either
        # way. What the graph changes is that its 1.85 GB of trunk is then never loaded.
        weights = (Path(checkpoint_path) if checkpoint_path
                   else resolve(FAMILY, variant, "torch-fp32", revision=revision))
        vision = None
        if framework_of(self.runtime) == "coreml":
            package = resolve(FAMILY, variant, self.runtime, revision=revision)
            vision = GraphVision(CoreMLRunner(package), self.device)
        self._segmenter = Segmenter(weights, device=self.device, vision=vision)
        _log.info("SAM 3 ready on %s via %s.", self.device, self.runtime)

    def _encode(self, pixels: np.ndarray) -> None:
        self._segmenter.encode_image(pixels)

    def predict(
        self,
        image: Union[str, Path, bytes, np.ndarray],
        text: Union[str, Sequence[str], None] = None,
        threshold: float = 0.5,
        *,
        points: Any = None,
        labels: Any = None,
        boxes: Any = None,
        multimask_output: bool = True,
        name: Union[str, Sequence[str], None] = None,
    ) -> "pf.detections.Detections":
        """Find every instance of each concept in ``text`` -- or segment what a click points at.

        Give a phrase, or give clicks; not both. A box beside a phrase means something else in
        SAM 3 -- an example of the concept, "find more like this" -- which this does not take yet,
        so the combination is refused rather than read one way or the other.

        Args:
            image: A file path, encoded image bytes, or an ``HxWx3`` RGB ``uint8`` array.
            text: One concept as a noun phrase -- ``"taxi"``, ``"yellow school bus"`` -- or
                several, as a sequence. Each is up to 32 tokens; longer is truncated. Several
                prompts share one image encode but each pays its own decode.
            threshold: Confidence floor for a phrase's instances. A click returns every
                candidate with its predicted IoU instead, as SAM 2 does.
            points: ``(N, 2)`` or ``(B, N, 2)`` x, y clicks in the image's own pixels.
            labels: ``(N,)`` or ``(B, N)``, 1 for a point to include and 0 for one to exclude.
                Required with *points*.
            boxes: ``(4,)`` or ``(B, 4)`` x1, y1, x2, y2 around the one thing to segment.
            multimask_output: Return three candidate masks per click rather than one. Worth
                keeping on for a single click, which is ambiguous about whether you meant the part
                or the whole.
            name: What to call what a click pointed at. One name, or one per prompt. Omitted,
                every click's detection carries ``class_name=None``.

        Returns:
            For a phrase, a PixelFlow ``Detections`` carrying a box, a mask and a score per
            instance, with ``class_name`` naming the prompt that found it. With several prompts
            this is one result carrying several classes rather than one result per prompt.

            Instances found by different prompts may overlap: ask for ``"car"`` and
            ``"vehicle"`` and the same car comes back twice, under each name. That is what was
            asked for, so nothing here suppresses it.

            For a click, what SAM 2 returns: one row per candidate mask, best first within each
            prompt, with the box enclosing it and the model's predicted IoU as the score.
            ``class_id`` is the index of the prompt.

        Raises:
            ValueError: If both a phrase and a click are given, or neither; if any concept is
                empty; or if points arrive without labels. SAM 3 will happily encode the empty
                string and return whatever it finds most salient, which is not what an empty
                prompt means.
                Also for a click on a graph exported with the phrase path only, as CoreML
                graphs published before clicks were added are.
        """
        clicked = points is not None or labels is not None or boxes is not None
        if clicked and text is not None:
            raise ValueError(
                "SAM 3 takes a phrase or a click, not both. A box beside a phrase is an example "
                "of the concept, which mozo does not take yet; send the phrase alone, or the "
                "click alone."
            )
        if clicked:
            if self._segmenter.click is None:
                raise ValueError(
                    f"SAM 3 on {self.runtime} answers phrases only: this revision's graph was "
                    "exported without the click half of the image encoder. Load it with "
                    "runtime='torch-fp32' to click, or a revision whose graph carries it."
                )
            pixels = load_image(image)
            self._after_prepare(pixels)
            found = self._segmenter.segment(
                pixels, points=points, labels=labels, boxes=boxes,
                multimask_output=multimask_output,
            )
            return candidates(found["masks"].cpu().numpy(), found["scores"].float().cpu().numpy(),
                              name, self.device)
        if text is None:
            raise ValueError(
                "SAM 3 needs a concept to look for or a click: pass text=, or points= and labels=, "
                "or boxes=."
            )

        prompts = _phrases(text)

        pixels = load_image(image)
        self._after_prepare(pixels)

        # One decode per prompt, because the head takes one prompt's features against one
        # image's -- there is no batched form to reach for. The encode is what they share, and
        # it is the expensive half, so this is the cheap direction of that trade.
        boxes, scores, masks, class_ids = [], [], [], []
        for index, prompt in enumerate(prompts):
            found = self._segmenter.predict(pixels, prompt, threshold=threshold)
            boxes.append(found["boxes"])
            scores.append(found["scores"])
            masks.append(found["masks"])
            class_ids.append(np.full(len(found["scores"]), index, dtype=np.int64))

        # The prompt list *is* the vocabulary, so class_ids index it. With one prompt this is
        # the single-class result it has always been; with several it is what class_ids are for.
        # Masks are already boolean and already in the source image's pixels, and PixelFlow's
        # converter detaches and moves tensors itself, so they go in as they are.
        # ``torch.cat`` allocates even for a one-element list, and one prompt is still the
        # common case -- on a 2 MP image that is a pointless copy of the whole mask stack.
        def joined(parts):
            return parts[0] if len(parts) == 1 else torch.cat(parts)

        return pf.detections.from_arrays(
            boxes=joined(boxes),
            scores=joined(scores),
            class_ids=class_ids[0] if len(class_ids) == 1 else np.concatenate(class_ids),
            masks=joined(masks),
            labels=prompts,
        )

    def tracker(self) -> "Sam3Tracker":
        """Start following concepts -- or clicked objects -- through a video, one frame at a time.

        A new, empty session on this loaded model. Give it phrases on the first frame and it finds
        every instance of each on every frame, starting new objects as they appear; or give it
        clicks or boxes on the first frame and it follows those objects. It reads only the frames
        before, never after, and keeps a bounded memory of them, so a long video or a camera runs
        in constant memory. Any number of trackers can share one model; one tracker follows one
        stream and is not safe to call from two threads at once.

        Raises:
            ValueError: If the model was loaded on a graph runtime. The graph is the image
                encoder alone, and the tracker has been checked only against the torch encoder.

        Examples:
            >>> model = mozo.get_model("sam3")                                # doctest: +SKIP
            >>> tracker = model.tracker()                                     # doctest: +SKIP
            >>> for frame in pf.VideoReader("street.mp4"):                    # doctest: +SKIP
            ...     found = tracker.update(frame, text="person")
        """
        if self._segmenter.tracker_net is None:
            raise ValueError(
                f"SAM 3's tracker runs on torch-fp32 only; this model was loaded on "
                f"{self.runtime}. Load it with runtime='torch-fp32' to track."
            )
        return Sam3Tracker(self._segmenter.tracker())


class Sam3Tracker:
    """One SAM 3 tracking session over one stream of frames. Made by :meth:`Sam3Predictor.tracker`.

    A session tracks phrases or clicked objects, decided by its first call. Phrases: every
    instance of each phrase is found on every frame and followed, new instances get new ids as
    they appear, and ones SAM 3 judges spurious soon after appearing are dropped. Clicks or boxes:
    each prompt on the first frame starts one object, followed as SAM 2's tracker follows it, with
    SAM 3's weights. The frames are taken in order and only forward.

    **The weights are not Apache-2.0** -- see :class:`Sam3Predictor`.
    """

    def __init__(self, session: Any) -> None:
        self._session = session
        self._texts: list[str] | None = None
        #: What to call each clicked object, by object id; ``None`` where no name was given.
        self._names: list[str | None] = []

    def update(
        self,
        image: Union[str, Path, bytes, np.ndarray],
        text: Union[str, Sequence[str], None] = None,
        *,
        points: Any = None,
        labels: Any = None,
        boxes: Any = None,
        name: Union[str, Sequence[str], None] = None,
    ) -> "pf.detections.Detections":
        """Follow every object onto the next frame.

        Args:
            image: The next frame: a file path, encoded image bytes, or an ``HxWx3`` RGB ``uint8``
                array. Every frame must be the size of the first.
            text: On the first call, one concept as a noun phrase or several; each is tracked on
                its own, sharing the frame's image encode. Later calls may repeat the same phrases
                (so a loop can pass them every time) or pass nothing.
            points: ``(N, 2)`` or ``(B, N, 2)`` x, y clicks in the frame's own pixels -- first call
                only.
            labels: ``(N,)`` or ``(B, N)``, 1 to include and 0 to exclude.
            boxes: ``(4,)`` or ``(B, 4)`` x1, y1, x2, y2 in the frame's own pixels.
            name: What to call the clicked objects: one name, or one per prompt.

        Returns:
            A PixelFlow ``Detections`` with one row per object shown on this frame, in object
            order. ``tracker_id`` is the object's id: 0, 1, 2… in the order objects first appear,
            across all phrases, never reused. ``class_id`` and ``class_name`` are the phrase's
            index and the phrase (for a click session, the prompt's index and its ``name``).
            ``confidence`` is SAM 3's probability that the object is visible on this frame (on the
            frame a phrase's object first appears, its detection score). The mask is at the
            frame's own size, with its tight box. An object hidden on a frame has no row and comes
            back under the same id.

        Raises:
            ValueError: If the first call has no prompt; a session is given both a phrase and
                clicks; a later call brings a different phrase, or clicks; a phrase is empty; a
                frame's size differs from the first's; points arrive without labels; or *name*
                does not give one name per prompt.
        """
        texts = None
        if text is not None:
            texts = _phrases(text)
            if self._texts is not None and texts == self._texts:
                texts = None
        prompts = split_prompts(points, labels, boxes)
        names = names_for(name, len(prompts))
        pixels = load_image(image)
        frame = self._session.update(pixels, texts=texts, prompts=prompts or None)
        if texts is not None:
            self._texts = texts
        self._names.extend(names)

        found = pf.Detections()
        masks = frame.masks.numpy()
        for row, obj_id in enumerate(frame.obj_ids):
            index = frame.prompts[row]
            label = self._texts[index] if self._texts is not None else self._names[index]
            found.add_detection(pf.Detection(
                bbox=bounds(masks[row][None])[0].tolist(),
                masks=[masks[row]],
                class_id=index,
                class_name=label,
                confidence=float(frame.scores[row]),
                tracker_id=obj_id,
            ))
        return found

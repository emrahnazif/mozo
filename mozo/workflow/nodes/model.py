"""One node per model family.

A node per family rather than one node with a family parameter, because the port system requires
it. What a family returns is fixed -- YOLO returns detections, Depth Anything returns a depth map,
CLIP returns classifications -- but it differs *between* families. A single node would have an
output type that depended on the value of one of its parameters, which is not a type the editor can
check a connection against. Refusing a bad wire before anything runs is the point of having port
types at all.

Each node is its family's ``predict`` with the arguments the editor can offer. Nothing here knows
how a model works, or where it comes from: each node asks the run's
:class:`~mozo.workflow.node.Models`, which is the cache behind :func:`mozo.get_model` unless the
host running the workflow supplied its own. Either way it caches, so a node that runs over fifty
images loads the checkpoint once.

Every family is here. SAM 2 and EdgeTAM come as box segmenters: they take the boxes another node
found -- a detector's, or annotations read from a file -- and add a mask to each. Their other
prompt, a click, is still not a node: it is a pixel picked on the image, and the editor has no way
to express one yet. A tool that clicks calls the model directly.

The variants each node offers are read from the registry rather than typed out again -- see
:func:`variants`. There is one list of what mozo publishes and this is not a second one.

What *is* stated twice is each family's one-line description: the registry has one for the
catalogue, and the docstrings here are what the editor's palette shows. They are written for
different places and different lengths, and deriving one from the other would mean splitting a
paragraph whose sentence separator is not consistent across families. Kept deliberately, and said
here so the duplication is a decision rather than a slip.
"""

from __future__ import annotations

import copy
from typing import Any, Literal, Optional

import numpy as np
import pixelflow as pf

from mozo.registry import get_model_info
from mozo.text import text_list

from ..node import Classifications, Depth, Detections, FilePath, Image, Models
from ..registry import node


def variants(family: str) -> Any:
    """The variants *family* publishes, as a choice the editor renders as a dropdown.

    Read from the registry so that publishing a variant offers it, and retiring one stops offering
    it, without this module being touched. ``Literal`` accepts a tuple, which is what makes a
    runtime list expressible as a static-looking annotation.

    Through :func:`~mozo.registry.get_model_info` rather than indexing the registry, so that a
    misspelled family says which families there are -- at import, before anything can run.
    """
    return Literal[tuple(get_model_info(family)["variants"])]


# --- Detection ---------------------------------------------------------------------------------

@node(category="Detect", exclusive=True)
def yolov8(image: Image, models: Models, variant: variants("yolov8") = "nano",
           threshold: float = 0.5) -> Detections:
    """YOLOv8 by Ultralytics -- real-time object detection."""
    return models("yolov8", variant).predict(image, threshold=threshold)


@node(category="Detect", exclusive=True)
def yolov11(image: Image, models: Models, variant: variants("yolov11") = "nano",
            threshold: float = 0.5) -> Detections:
    """YOLO11 by Ultralytics -- real-time object detection."""
    return models("yolov11", variant).predict(image, threshold=threshold)


@node(category="Detect", exclusive=True)
def yolov12(image: Image, models: Models, variant: variants("yolov12") = "nano",
            threshold: float = 0.5) -> Detections:
    """YOLO12 by Ultralytics -- attention-centric real-time object detection."""
    return models("yolov12", variant).predict(image, threshold=threshold)


@node(category="Detect", exclusive=True)
def yolov26(image: Image, models: Models, variant: variants("yolov26") = "nano",
            threshold: float = 0.5) -> Detections:
    """YOLO26 by Ultralytics -- NMS-free detection, and instance masks on the seg variants."""
    return models("yolov26", variant).predict(image, threshold=threshold)


@node(category="Detect", exclusive=True)
def rfdetr(image: Image, models: Models, variant: variants("rfdetr") = "nano",
           threshold: float = 0.5) -> Detections:
    """RF-DETR by Roboflow -- NMS-free transformer detection, under a permissive licence."""
    return models("rfdetr", variant).predict(image, threshold=threshold)


@node(category="Detect", exclusive=True)
def rfdetr_checkpoint(image: Image, models: Models, checkpoint: Optional[FilePath] = None,
                      architecture: variants("rfdetr") = "nano", labels: list[str] = [],
                      threshold: float = 0.5) -> Detections:
    """RF-DETR from a checkpoint of your own -- a model you trained, run like a published one.

    ``architecture`` is the RF-DETR the checkpoint was trained as. ``labels`` names its classes,
    one per entry in class-id order, each exactly as given; leave it empty to use the names the
    checkpoint carries. (A plain string, from workflows saved before, is split on commas.)
    """
    if not checkpoint:
        raise ValueError("choose the checkpoint file to run")
    names = text_list(labels)
    own = {"labels": names} if names else {}
    return models("rfdetr", architecture, checkpoint_path=checkpoint, **own).predict(
        image, threshold=threshold)


# --- Detection from a description ----------------------------------------------------------------

@node(category="Detect", exclusive=True)
def owlv2(image: Image, models: Models, text: list[str] = ["a person", "a car", "a dog"],
          variant: variants("owlv2") = "base-ensemble", threshold: float = 0.1) -> Detections:
    """OWLv2 by Google -- find anything you can name, with no training.

    One prompt per entry, taken as given; each result's ``class_id`` is its prompt's index and its
    ``class_name`` that prompt.
    """
    return models("owlv2", variant).predict(
        image, text=text_list(text), threshold=threshold)


@node(category="Detect", exclusive=True)
def grounding_dino(image: Image, models: Models, text: list[str] = ["person", "car", "dog"],
                   variant: variants("grounding_dino") = "tiny",
                   threshold: float = 0.3) -> Detections:
    """Grounding DINO by IDEA -- open-vocabulary detection from a phrase.

    One prompt per entry, taken as given; each result's ``class_id`` is its prompt's index and its
    ``class_name`` that prompt.
    """
    return models("grounding_dino", variant).predict(
        image, text=text_list(text), threshold=threshold)


# --- Segmentation --------------------------------------------------------------------------------

@node(category="Segment", exclusive=True)
def sam3(image: Image, models: Models, text: list[str] = ["person"],
         variant: variants("sam3") = "sam3", threshold: float = 0.5) -> Detections:
    """SAM 3 by Meta -- masks for every instance of a concept you name.

    One prompt per entry, taken as given; each result's ``class_id`` is its prompt's index and its
    ``class_name`` that prompt.
    """
    return models("sam3", variant).predict(
        image, text=text_list(text), threshold=threshold)


def _segment_boxes(models: Models, family: str, variant: str, image: Image,
                   detections: Detections) -> Detections:
    """Add a mask to every row of *detections*, from one call to *family* with all its boxes.

    One call rather than one per box, because the image encode is the cost and all the boxes share
    it. One mask per box: a box says which object, so the three candidates a click needs are noise.
    The adapter numbers what it returns by prompt, so each mask is put back on the row whose box
    asked for it, and every other field of that row -- its box, class, name and score -- is kept as
    it came. An image with nothing found never asks for the model, so it is never loaded for it.
    """
    if len(detections) == 0:
        return detections
    boxes = np.array([row.bbox for row in detections], dtype=np.float32)
    found = models(family, variant).predict(image, boxes=boxes, multimask_output=False)
    if len(found) != len(detections):
        raise ValueError(f"{family} returned {len(found)} masks for {len(detections)} boxes; "
                         f"it should return one per box")
    # Each row copied shallowly -- its old masks, about to be replaced, are not worth duplicating.
    rows = [copy.copy(row) for row in detections]
    for answer in found:
        rows[int(answer.class_id)].masks = answer.masks
    segmented = pf.Detections()
    for row in rows:
        segmented.add_detection(row)
    return segmented


@node(category="Segment", exclusive=True)
def sam2_boxes(image: Image, detections: Detections, models: Models,
               variant: variants("sam2") = "tiny") -> Detections:
    """SAM 2 by Meta -- a mask for each box it is given, the rows otherwise unchanged.

    Wire a detector into ``detections`` -- Grounding DINO finds boxes from a description, this
    outlines them -- or read existing annotations to give a box dataset its masks. The rows come
    back in the order they arrived, each with its ``class_id``, ``class_name``, ``confidence`` and
    ``bbox`` exactly as given and a mask added: the box stays the one the detector drew, not the
    mask's bounds, so its score still describes it. A row whose mask comes back empty is kept with
    an empty mask rather than dropped, so no box is ever lost. All of an image's boxes are
    segmented in one call, and an image with none is passed through without running the model.
    """
    return _segment_boxes(models, "sam2", variant, image, detections)


@node(category="Segment", exclusive=True)
def edgetam_boxes(image: Image, detections: Detections, models: Models,
                  variant: variants("edgetam") = "edgetam") -> Detections:
    """EdgeTAM -- SAM 2 made small: a mask for each box it is given, the rows otherwise unchanged.

    Everything ``sam2_boxes`` promises, at a fraction of the cost.
    """
    return _segment_boxes(models, "edgetam", variant, image, detections)


# --- Edit ------------------------------------------------------------------------------------------

@node(category="Edit", exclusive=True)
def moebius(image: Image, detections: Detections, models: Models,
            variant: variants("moebius") = "general",
            seed: int = 0, dilate: int = 0) -> Image:
    """Moebius -- remove what you point it at and repaint the gap. Wire a segmenter into ``detections``.

    The only node here that answers with a new picture rather than a description of one, and the
    only one where running it twice is meant to give two different answers: ``seed`` picks which.
    Raise ``dilate`` when the thing is still faintly there -- an object's shadow and its soft edge
    usually sit outside the mask that found it.
    """
    return models("moebius", variant).predict(image, detections, seed=seed, dilate=dilate)


@node(category="Edit", exclusive=True)
def ben2(image: Image, models: Models, variant: variants("ben2") = "base",
         stretch: bool = True, refine: bool = False) -> Image:
    """BEN2 -- cut the background away, keeping a soft edge. Takes no prompt and no boxes.

    Answers with a cut-out: the frame with the model's per-pixel opacity in its alpha channel --
    an ordinary picture on an ordinary image wire, so it saves, previews and connects like any
    other. A matte rather than a mask is the point: threshold a head of hair and the hair leaves
    with the background.

    Wire it into a detector or an annotator and the transparency is simply dropped: an image port
    carries three channels or four, and a node is handed three unless it said it wanted four.
    Nothing downstream needs to know this one came from here.

    ``stretch`` reproduces upstream's per-image contrast normalisation of the matte, which is on by
    default because it is what BEN2's own output looks like. Turn it off for the calibrated
    sigmoid, which is what you want if a later node thresholds the alpha or compares two frames.
    ``refine`` re-estimates foreground colour along soft edges so a composite does not carry a
    fringe of the old background; it costs two full-resolution blurs.
    """
    return models("ben2", variant).cutout(image, stretch=stretch, refine=refine)


# --- Pose ------------------------------------------------------------------------------------------

@node(category="Pose", exclusive=True)
def vitpose(image: Image, detections: Detections, models: Models,
            variant: variants("vitpose") = "small") -> Detections:
    """ViTPose++ -- the joints of everyone you point it at. Wire a detector into ``detections``."""
    return models("vitpose", variant).predict(image, detections)


# --- Classification ------------------------------------------------------------------------------

@node(category="Classify", exclusive=True)
def clip(image: Image, models: Models, text: list[str] = ["a photo of a cat", "a photo of a dog"],
         variant: variants("clip") = "base") -> Classifications:
    """CLIP by OpenAI -- score an image against phrases you make up.

    One prompt per entry, taken as given; each result's ``class_id`` is its prompt's index and its
    ``class_name`` that prompt.
    """
    return models("clip", variant).predict(image, text=text_list(text))


@node(category="Classify", exclusive=True)
def siglip2(image: Image, models: Models, text: list[str] = ["a photo of a cat", "a photo of a dog"],
            variant: variants("siglip2") = "base-224") -> Classifications:
    """SigLIP 2 by Google -- zero-shot classification against phrases you make up.

    One prompt per entry, taken as given; each result's ``class_id`` is its prompt's index and its
    ``class_name`` that prompt.
    """
    return models("siglip2", variant).predict(image, text=text_list(text))


# --- Reading -------------------------------------------------------------------------------------

@node(category="Read", exclusive=True)
def easyocr(image: Image, models: Models, variant: variants("easyocr") = "english") -> Detections:
    """EasyOCR by JaidedAI -- find text in an image and read it."""
    return models("easyocr", variant).predict(image)


# --- Depth ---------------------------------------------------------------------------------------

@node(category="Depth", exclusive=True)
def depth_anything_v2(image: Image, models: Models,
                      variant: variants("depth_anything_v2") = "small") -> Depth:
    """Depth Anything V2 -- how far away everything is, from one photograph."""
    return models("depth_anything_v2", variant).predict(image)

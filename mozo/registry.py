"""What model families exist, answerable without importing one.

Every adapter pulls torch in, and the server has to list what it serves before anything is
loaded. So the catalogue is plain data here, and each family's variant list is written twice --
once here, once as the adapter's ``VARIANTS``. A test per family holds the two in step
(``tests/families/test_*.py::test_registry_agrees_with_the_adapter``); importing one from the
other would put torch back on the import path and defeat the point.

To add a family: add an entry below, and give its adapter the same ``VARIANTS``.

    >>> from mozo.registry import get_model_info
    >>> get_model_info("rfdetr")["task_type"]
    'object_detection'
"""

from __future__ import annotations

__all__ = ["BOXED", "CLICKABLE", "ENCODE_ONLY", "ENCODES", "EXECUTES", "MODEL_REGISTRY", "NEEDS", "PREPARES",
           "PROMPTED", "get_model_info"]

from typing import Any

#: Task types whose model is asked a question in words. They differ in what comes back -- SAM 3
#: answers with masks, OWLv2 with boxes -- but the request is the same shape, and so is the way a
#: missing prompt has to be refused. Named here rather than in the server because it is a
#: statement about the task vocabulary, and this is the module that owns those strings; the
#: endpoint and the tests both read it from here rather than keeping copies in step.
PROMPTED = frozenset({
    "concept_segmentation", "open_vocabulary_detection", "zero_shot_classification"})

#: Task types whose model is told *where* to look and cannot answer without being told. Two of
#: them, needing a box for opposite reasons: a top-down pose model has no detector and is handed
#: the subject's box, while an inpainter is handed the thing to delete. What they share is all
#: this set encodes -- a request with no box is a mistake rather than an empty frame, and the
#: editor has to offer a rectangle before the request can be made at all. The refusals read
#: differently, and ``mozo.server`` branches on the task to get each one right.
#:
#: Named here for the same reason PROMPTED is, and it is the same shape of fact: the endpoint has
#: to refuse before the image decode and the multi-gigabyte load, and the browser page has to know
#: which families need boxes drawn on the picture before it can offer the control. Neither can be
#: derived from the task name by a reader who does not already know.
#:
#: Distinct from promptable segmentation, which *accepts* a box and equally accepts a click. This
#: is the set that requires one.
BOXED = frozenset({"pose_estimation", "image_inpainting"})

#: Families that segment what a click or a box points at: ``?point=`` and ``?box=`` mean "this
#: one thing", as they do for SAM 2. Every promptable-segmentation family, and SAM 3, whose task
#: is concept segmentation but which carries a click head too -- so a family rather than a task
#: set, because a task name cannot say that one of its families also does this. Read before
#: anything is loaded, for the reason PROMPTED is, and served by ``/models`` so a page knows when
#: to let a picture be clicked.
CLICKABLE = frozenset({"edgetam", "sam2", "sam3"})

#: What each family can encode, and from what. A family absent from here does not encode at all.
#:
#: Read before anything is loaded, for the same reason PROMPTED is: ``/encode``'s refusal has to
#: come before the image decode and the multi-gigabyte download, not after. ``/predict`` can afford
#: a late 501 because a test proves every registered task has a branch there, so it never fires;
#: on ``/encode`` the reverse holds, because only the families that embed have one.
#:
#: A dict rather than a set because the kinds differ: CLIP takes both, a re-identification embedder
#: would take images only, and ``/models`` should be able to say which without loading anything.
#: The task type cannot carry this -- two families can classify while only one of them embeds.
ENCODES: dict[str, frozenset[str]] = {
    "clip": frozenset({"image", "text"}),
    "granite_embedding_r2": frozenset({"text"}),
    "siglip2": frozenset({"image", "text"}),
}

#: Task types whose model answers nothing about an image: it returns vectors, and ``/encode`` is
#: the whole of what it does. ``/predict`` takes an image and returns an answer, so it refuses these
#: before decoding or loading anything and points at ``/encode`` instead -- rather than demanding a
#: picture the model would ignore, downloading the weights, and only then giving up.
#:
#: Named here for the reason PROMPTED is: the endpoint has to refuse before the load, and a test
#: holds every registered task to having somewhere to go.
ENCODE_ONLY = frozenset({"text_embedding"})

#: Families whose model can be prepared: given an image ahead of its prompts, so it runs the
#: expensive, image-only half now and each prompt afterwards runs only the cheap half. Their
#: adapters have ``prepare(image)``, and ``/predict`` accepts ``?sha256=`` for an image
#: ``/prepare`` was given. Read before anything is loaded, for the reason ENCODES is: an endpoint
#: refuses the rest before decoding an upload, and a page reads it to know when to offer it.
PREPARES = frozenset({"edgetam", "owlv2", "sam2", "sam3"})

#: Which frameworks each family's adapter has code to execute. A family absent from here runs
#: whatever it publishes. Passed to :func:`mozo.runtimes.select_runtime` as ``executes``.
#:
#: Data here rather than an attribute on the adapter, because which runtime ``auto`` lands on
#: decides which files a load will fetch, and :mod:`mozo.weights` has to answer that -- "is this
#: downloaded", "how much is left to fetch" -- without importing an adapter and torch with it.
#: The adapters read it from here, so there is one statement of it.
EXECUTES: dict[str, tuple[str, ...]] = {
    **{family: ("torch",)
       for family in ("ben2", "clip", "easyocr", "edgetam", "moebius", "sam2", "siglip2")},
    # A TensorRT engine exported from the checkpoint at the input size the user picks.
    "grounding_dino": ("torch", "tensorrt"),
    "granite_embedding_r2": ("torch", "coreml"),
    # The graph is the image path; the text tower always runs in torch, once per vocabulary.
    "omdet_turbo": ("torch", "onnx", "tensorrt"),
    # Not "onnx": the vision graph is published to build TensorRT engines from, and ONNX Runtime
    # ran it twice as slow as torch on the CPU. See the OWLv2 PROVENANCE, under Export.
    "owlv2": ("torch", "tensorrt"),
}

#: Runtimes a family builds at load time from another published artifact, rather than
#: downloading as a file of their own: derived runtime -> the artifact it is built from.
#:
#: ``torch-fp16`` is the fp32 checkpoint with part of the network cast to half precision, so it
#: has no file to publish. A TensorRT engine is specific to the GPU and the TensorRT version that
#: built it, so it *cannot* be published: it is built on first load and cached -- from a published
#: fp16 ONNX graph (OWLv2), or from a graph exported from the checkpoint there and then, where the
#: graph depends on an input size the user chooses (Grounding DINO). A derived runtime is offered
#: only where its source is published, and it fetches its source's files, so ``fetch``,
#: ``is_cached`` and ``download_size`` stay truthful.
#:
#: Deriving ``tensorrt-fp32`` opts a family into ``auto`` choosing it on CUDA
#: (:data:`mozo.runtimes._PREFERENCE`): add it only with measured exactness and speed.
#:
#: Each entry is a measured decision, not a capability: each family's fp16 was measured on 500
#: COCO images against fp32's own noise floor before it was offered. See the family's page.
DERIVES: dict[str, dict[str, str]] = {
    "owlv2": {"torch-fp16": "torch-fp32", "tensorrt-fp16": "onnx-fp16"},
    "grounding_dino": {"torch-fp16": "torch-fp32", "tensorrt-fp16": "torch-fp32"},
    # ``tensorrt-fp32`` from the published graph, for every variant that publishes one, and
    # ``auto`` takes it on CUDA where TensorRT is installed: on an RTX 3090, 500 COCO images per
    # variant, not one detection changed and AP was identical, at 1.65x (``large``) to 2.55x
    # (``seg-nano``) torch fp32 -- RF-DETR is launch-bound at batch 1. fp16 only where an fp16
    # graph is published (the detection variants), and only on request.
    "rfdetr": {"tensorrt-fp32": "onnx-fp32", "tensorrt-fp16": "onnx-fp16"},
    # The same shape as RF-DETR's, measured the same way (RTX 3090, 500 street-validation images
    # per variant): the fp32 engine changed no detection and left box and mask AP identical, at
    # 1.4-1.9x torch's forward pass; fp16 changed none either, moving AP by at most 0.1, so
    # ``auto`` takes fp16 (:data:`PREFERS`). See docs/models/mayaku.md.
    "mayaku": {"tensorrt-fp32": "onnx-fp32", "tensorrt-fp16": "onnx-fp16"},
    # The fp32 engine is exact, at 1.6x torch.
    "omdet_turbo": {"tensorrt-fp32": "onnx-fp32", "tensorrt-fp16": "onnx-fp16"},
}

#: Runtimes a family's ``auto`` tries before the shared order (:data:`mozo.runtimes._PREFERENCE`),
#: by device type. The shared order is fp32-first because fp16 measured badly on some families
#: (OWLv2, Grounding DINO); a family where the faster runtime measured as accurate says so here,
#: so its users get the fast path without asking. Each entry records its measurement.
PREFERS: dict[str, dict[str, tuple[str, ...]]] = {
    # RTX 3090, 500 street-validation images per variant: the fp16 engine changed no confident
    # detection and moved box/mask AP by at most 0.1, at 1.5-2.4x the fp32 engine. On the CPU,
    # ONNX Runtime runs the same exact graph 1.9-2.3x faster than torch (i9-10900X). On Apple
    # silicon (M1 Max, 301 photographs), the Core ML fp16 package changed at most 1 of ~1,500
    # confident detections, at 1.9-4.7x torch MPS (5.1 against 23.8 ms for street-nano).
    "mayaku": {"cuda": ("tensorrt-fp16",), "cpu": ("onnx-fp32",), "mps": ("coreml-fp16",)},
    # RTX 3090, 500 COCO images, 80 names: the fp16 engine changed 5 of 4,843 detections above 0.3
    # and left AP at 43.76 (fp32 43.74), at 1.8x the fp32 engine (7.8 against 13.9 ms). The CPU
    # stays on torch: ONNX Runtime measured 5-18% faster on a loaded i9 but 32% slower on an
    # M1 Max (563 against 425 ms), so neither wins everywhere.
    "omdet_turbo": {"cuda": ("tensorrt-fp16",)},
}

#: Artifacts a load fetches beside its runtime, by the framework that runtime belongs to; ``"*"``
#: for every runtime. A key a revision does not publish is skipped, as the adapter skips it.
#:
#: The graphs carry an architecture but no vocabulary, so a YOLO run from one reads its class
#: names from the published ``labels`` file; run from torch it reads them off the checkpoint.
#: RF-DETR's and ViTPose's checkpoints carry no names at all, so they read labels on every
#: runtime. SAM 3's graph covers only the vision encoder, so its checkpoint is fetched whichever
#: runtime won. OWLv2's graph is the same shape of thing -- the vision tower alone -- so its text
#: tower and heads come from the checkpoint, on ONNX and on the TensorRT engine built from it.
#:
#: The same reason as :data:`EXECUTES`: what a load downloads has to be answerable without the
#: adapter. A test loads each family and holds this to the files it actually fetched.
NEEDS: dict[str, dict[str, tuple[str, ...]]] = {
    **{family: {"onnx": ("labels",), "coreml": ("labels",)}
       for family in ("yolov8", "yolov11", "yolov12", "yolov26")},
    "rfdetr": {"*": ("labels",)},
    "mayaku": {"onnx": ("labels",), "tensorrt": ("labels",), "coreml": ("labels",)},
    "deim": {"*": ("labels",)},
    "vitpose": {"*": ("labels",)},
    "sam3": {"onnx": ("torch-fp32",), "coreml": ("torch-fp32",)},
    "owlv2": {"onnx": ("torch-fp32",), "tensorrt": ("torch-fp32",)},
    # Its graph is the image path and takes the text features as inputs; the text tower that
    # makes them comes from the checkpoint.
    "omdet_turbo": {"onnx": ("torch-fp32",), "tensorrt": ("torch-fp32",)},
}

#: Family -> where its adapter lives, what it does, and which variants it publishes.
#: An empty ``variants`` list means the family accepts any variant name.
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    'ben2': {
        'adapter_class': 'Ben2Predictor',
        'module': 'mozo.adapters.ben2',
        'task_type': 'image_matting',
        'description': (
            'BEN2 by Prama LLC — background removal. Hand it a photograph and it returns an '
            'alpha matte: a per-pixel opacity, not a binary mask, which is the difference '
            'between keeping a head of hair and cutting it off. 94.6M parameters, a Swin-B '
            'backbone under a confidence-guided decoder that re-attends to the pixels the model '
            'was least sure of — which is where hair and fur live. 1 variant. It takes no '
            'prompt and no box: the whole answer is the picture. Runs at 1024x1024 and nothing '
            'else, splitting each image into four quadrants plus a half-scale global view. Note '
            'that the alpha is contrast-stretched per image by default, as upstream does, so 255 '
            'means the most foreground pixel in *this* picture — pass stretch=False for the '
            'calibrated sigmoid. MIT on both the code and the weights.'
        ),
        'variants': ['base'],
    },

    'clip': {
        'adapter_class': 'ClipPredictor',
        'module': 'mozo.adapters.clip',
        'task_type': 'zero_shot_classification',
        'description': (
            'CLIP by OpenAI — zero-shot classification, and the embeddings behind it. Name any '
            'classes in words and it scores an image against them, with no training and no '
            'labelled data. 4 variants (base/base-16/large/large-336), all Vision Transformers. '
            'It also hands back the vectors: an image and a phrase embed into one shared space, '
            'so a corpus embedded once can be searched by words afterwards through a vector '
            'database of your own. Scores are cosine similarities, not probabilities. Code and '
            'weights are both MIT.'
        ),
        'variants': ['base', 'base-16', 'large', 'large-336'],
    },

    'deim': {
        'adapter_class': 'DEIMPredictor',
        'module': 'mozo.adapters.deim',
        'task_type': 'object_detection',
        'description': (
            'DEIM by Intellindust AI Lab (CVPR 2025) — real-time object detection, COCO\'s 80 '
            'classes. A DETR: 300 queries each propose a box, and there is no non-maximum '
            'suppression to tune. D-FINE\'s architecture (HGNetv2 backbone, hybrid encoder, a '
            'decoder that refines each box edge as a distribution) trained with DEIM\'s denser '
            'matching. 6 variants: nano/small/medium/large/xlarge trained on COCO (43.0 to 56.5 '
            'AP, upstream\'s numbers), and xlarge-obj365, pretrained on Objects365 then '
            'fine-tuned on COCO, the most accurate (59.5 AP). Runs at 640x640. Code is '
            'Apache-2.0; the weights are published beside it with no terms of their own.'
        ),
        'variants': ['nano', 'small', 'medium', 'large', 'xlarge', 'xlarge-obj365'],
    },

    'depth_anything_v2': {
        'adapter_class': 'DepthAnythingV2Predictor',
        'module': 'mozo.adapters.depth_anything_v2',
        'task_type': 'depth_estimation',
        'description': (
            'Depth Anything V2 by TikTok & HKU — monocular depth estimation. '
            '3 relative-depth variants (small/base/large) and the same three sizes '
            'fine-tuned for metric depth indoors (indoor-*) and outdoors (outdoor-*). '
            'Relative base/large are CC-BY-NC-4.0; the rest are Apache 2.0.'
        ),
        'variants': [
            'small', 'base', 'large',
            'indoor-small', 'indoor-base', 'indoor-large',
            'outdoor-small', 'outdoor-base', 'outdoor-large',
        ],
    },

    'easyocr': {
        'adapter_class': 'EasyOCRPredictor',
        'module': 'mozo.adapters.easyocr',
        'task_type': 'text_recognition',
        'description': (
            'EasyOCR by Jaided AI — text detection and recognition. Finds every line of text '
            'and reads it, returning the string, the four corners as read and a confidence. '
            'Two graphs: CRAFT locates, a CRNN reads. A variant is a script rather than a '
            'language — 5 of them, covering 88% of upstream\'s own downloads, with latin alone '
            'spanning 41 languages. Detections carry text, not a class name: OCR reads content, '
            'it does not pick a class out of a vocabulary. Code and weights are Apache 2.0.'
        ),
        'variants': ['english', 'latin', 'chinese-simplified', 'japanese', 'korean'],
    },

    'edgetam': {
        'adapter_class': 'EdgeTamPredictor',
        'module': 'mozo.adapters.edgetam',
        'task_type': 'promptable_segmentation',
        'description': (
            'EdgeTAM by Meta — promptable segmentation, SAM 2 distilled for phones. Click a '
            'point or draw a box and it returns the thing you pointed at, with a mask, a box '
            'and the model\'s own predicted IoU. A 9.1M-parameter image path against SAM 2 '
            'tiny\'s 31.4M. One published model, so one variant. Code and weights are both '
            'Apache 2.0.'
        ),
        'variants': ['edgetam'],
    },

    'granite_embedding_r2': {
        'adapter_class': 'GraniteEmbeddingR2Predictor',
        'module': 'mozo.adapters.granite_embedding_r2',
        'task_type': 'text_embedding',
        'description': (
            'Granite Embedding R2 by IBM — English text embeddings. Text in, one vector out, and '
            'texts that mean the same thing land close together: search documentation, notes or '
            'what OCR read off an image by meaning rather than by keyword. Text only — it has no '
            'image tower, so its vectors do not share a space with CLIP\'s or SigLIP 2\'s. 47.7M '
            'parameters, 384-d vectors, up to 8,192 tokens a text. Served through /encode only. '
            'Code and weights are both Apache-2.0.'
        ),
        'variants': ['small'],
    },

    'grounding_dino': {
        'adapter_class': 'GroundingDinoPredictor',
        'module': 'mozo.adapters.grounding_dino',
        'task_type': 'open_vocabulary_detection',
        'description': (
            'Grounding DINO by IDEA Research — open-vocabulary detection. Describe anything in '
            'words and it returns boxes for it, with no class list and no training. 2 variants: '
            'tiny (Swin-T) and base (Swin-B, 8.3 box AP better zero-shot). Text is fused into '
            'the image features six times over and the decoder attends back to the words, so a '
            'phrase is read rather than treated as a bag of words — reach for it over OWLv2 '
            'when the prompt is a description. Boxes, not masks — pair it with SAM 2 or EdgeTAM '
            'for those. Code is Apache 2.0, and the authors publish the weights under it too.'
        ),
        'variants': ['tiny', 'base'],
    },

    'mayaku': {
        'adapter_class': 'MayakuPredictor',
        'module': 'mozo.adapters.mayaku',
        'task_type': 'object_detection',
        'description': (
            'Mayaku — real-time detection and instance segmentation, a five-op convolutional '
            'graph. Its published models are its variants; the first are the street models: road '
            'scenes from dashcam to drone and straight-down aerial views, used as they are with '
            'no training. 22 classes: person, face, helmet, bicycle, e-scooter, motorcycle, '
            'wheelchair, stroller, car, truck, bus, emergency vehicle, train, three-wheeler, '
            'construction vehicle, trailer, license plate, traffic light, safety sign, stop sign, '
            'traffic cone, animal. Faces and license plates are there so footage can be blurred '
            'before it is stored or shared. 4 variants: street-nano and street-medium for boxes, '
            'street-seg-nano and street-seg-medium for boxes and masks (the same boxes). Code and '
            'weights are both Apache-2.0; the street training labels were made with SAM 3.'
        ),
        'variants': ['street-nano', 'street-medium', 'street-seg-nano', 'street-seg-medium'],
    },

    'moebius': {
        'adapter_class': 'MoebiusPredictor',
        'module': 'mozo.adapters.moebius',
        'task_type': 'image_inpainting',
        'description': (
            'Moebius by HUST and vivo AI Lab — object removal. Hand it an image and a mask and '
            'it repaints the hole so the thing was never there. 226M parameters matching '
            'FLUX.1-Fill-dev at 11.9B across six benchmarks, which is what makes it small '
            'enough to be worth deploying. 2 variants: general and places2 (scenes and '
            'backgrounds). Pair the mask with SAM 3 or EdgeTAM. Unlike every other family here '
            'it answers with an image rather than a description of one, and it answers with a '
            'sample rather than an estimate — change the seed and you get a different, equally '
            'valid removal. Runs at 512x512 and nothing else. Apache 2.0 code; the authors '
            'state the weights as Apache 2.0 on GitHub and MIT on the model card, both '
            'permissive and both permitting commercial use.'
        ),
        'variants': ['general', 'places2'],
    },

    'omdet_turbo': {
        'adapter_class': 'OmDetTurboPredictor',
        'module': 'mozo.adapters.omdet_turbo',
        'task_type': 'open_vocabulary_detection',
        'description': (
            'OmDet-Turbo by Om AI Lab — real-time open-vocabulary detection. Name things in '
            'words and it returns boxes for them, with no class list and no training. 1 variant, '
            'tiny (Swin-T). The prompts are encoded once per vocabulary and the decoder reads '
            'them as a sentence as well as a list, so it sits between OWLv2 and Grounding DINO: '
            'faster than either, and the one to reach for when a fixed vocabulary meets many '
            'images. Boxes, not masks. Code and weights are both Apache-2.0.'
        ),
        'variants': ['tiny'],
    },

    'owlv2': {
        'adapter_class': 'OwlV2Predictor',
        'module': 'mozo.adapters.owlv2',
        'task_type': 'open_vocabulary_detection',
        'description': (
            'OWLv2 by Google Research — open-vocabulary detection. Name anything in words '
            'and it returns boxes for it, with no class list and no training. 4 variants: '
            'base-ensemble/large-ensemble average the self-trained and fine-tuned checkpoints, '
            'base/large are self-training only. Boxes, not masks — pair it with SAM 2 or '
            'EdgeTAM for those. Unlike SAM 3, the code and all four checkpoints are Apache 2.0.'
        ),
        'variants': ['base-ensemble', 'base', 'large-ensemble', 'large'],
    },

    'rfdetr': {
        'adapter_class': 'RFDETRPredictor',
        'module': 'mozo.adapters.rfdetr',
        'task_type': 'object_detection',
        'description': (
            'RF-DETR by Roboflow — real-time transformer detection, instance segmentation & '
            'keypoints. 4 detection variants (nano/small/medium/large), 4 segmentation variants '
            '(seg-nano/seg-small/seg-medium/seg-large), and keypoint-preview, which returns '
            "COCO's 17 person joints. All variants Apache 2.0 licensed."
        ),
        'variants': [
            'nano', 'small', 'medium', 'large',
            'seg-nano', 'seg-small', 'seg-medium', 'seg-large',
            'keypoint-preview',
        ],
    },

    'sam2': {
        'adapter_class': 'Sam2Predictor',
        'module': 'mozo.adapters.sam2',
        'task_type': 'promptable_segmentation',
        'description': (
            'SAM 2 by Meta — promptable segmentation. Click a point or draw a box and it '
            'returns the thing you pointed at, with a mask, a box and the model\'s own '
            'predicted IoU. 4 variants (tiny/small/base_plus/large); the image path only, not '
            'the video tracker. Unlike SAM 3, the weights are Apache 2.0 like the code.'
        ),
        'variants': ['tiny', 'small', 'base_plus', 'large'],
    },

    'sam3': {
        'adapter_class': 'Sam3Predictor',
        'module': 'mozo.adapters.sam3',
        'task_type': 'concept_segmentation',
        'description': (
            'SAM 3 by Meta — promptable concept segmentation. Name a thing in words and it '
            'returns every instance of it, with a mask, a box and a score; or click a point or '
            'draw a box and it returns the one thing you pointed at, as SAM 2 does. One '
            'published model, so one variant. Unlike every other family here the weights are neither Apache-2.0 '
            'nor AGPL: they carry Meta\'s SAM License, which restricts what they may be used '
            'for and binds whoever you serve predictions to. See the NOTICE published beside '
            'the checkpoint.'
        ),
        'variants': ['sam3'],
    },

    'siglip2': {
        'adapter_class': 'Siglip2Predictor',
        'module': 'mozo.adapters.siglip2',
        'task_type': 'zero_shot_classification',
        'description': (
            'SigLIP 2 by Google — zero-shot classification, and the embeddings behind it. Name '
            'any classes in words and it scores an image against them, with no training and no '
            'labelled data. 5 variants across three sizes (base/so400m/giant). Unlike CLIP it '
            'was trained pair by pair with a sigmoid loss, so each score is a probability for '
            'that one image-and-phrase on its own: adding a phrase moves no other score, the set '
            'does not sum to one, and every phrase can be near zero '
            'when none of them fits. Multilingual. It also hands back the vectors, for a vector '
            'database of your own. Code and weights are both Apache-2.0.'
        ),
        'variants': ['base-224', 'base-256', 'so400m-384', 'so400m16-256', 'giant-384'],
    },

    'vitpose': {
        'adapter_class': 'ViTPosePredictor',
        'module': 'mozo.adapters.vitpose',
        'task_type': 'pose_estimation',
        'description': (
            'ViTPose++ by the University of Sydney — human pose estimation. Give it a frame and '
            "the boxes of the people in it, and it returns those same detections with COCO's 17 "
            'joints attached to each. 4 variants (small/base/large/huge). It is top-down: it does '
            'not find people, so pair it with a detector — RF-DETR and the YOLO families all '
            'produce boxes it accepts. It also does not filter them, so pass it the people rather '
            'than everything, or it will place joints on a car as readily as on a person. Code '
            'and weights are both Apache-2.0.'
        ),
        'variants': ['small', 'base', 'large', 'huge'],
    },

    'yolov8': {
        'adapter_class': 'YOLOv8Predictor',
        'module': 'mozo.adapters.yolov8',
        'task_type': 'object_detection',
        'description': (
            'YOLOv8 by Ultralytics -- real-time object detection. '
            '5 variants (nano/small/medium/large/xlarge). '
            'The weights are AGPL-3.0, unlike the rest of mozo: serving predictions from them '
            'over a network places obligations on you. See the NOTICE published beside each '
            'checkpoint.'
        ),
        'variants': ['nano', 'small', 'medium', 'large', 'xlarge'],
    },

    'yolov11': {
        'adapter_class': 'YOLOv11Predictor',
        'module': 'mozo.adapters.yolov11',
        'task_type': 'object_detection',
        'description': (
            'YOLO11 by Ultralytics -- real-time object detection and instance segmentation, the '
            'generation after YOLOv8. 10 variants -- 5 detection '
            '(nano/small/medium/large/xlarge) and 5 segmentation '
            '(seg-nano/seg-small/seg-medium/seg-large/seg-xlarge), which add a mask per detection '
            'and change nothing else. '
            'The weights are AGPL-3.0, unlike the rest of mozo: serving predictions from them '
            'over a network places obligations on you. See the NOTICE published beside each '
            'checkpoint.'
        ),
        'variants': ['nano', 'small', 'medium', 'large', 'xlarge',
                     'seg-nano', 'seg-small', 'seg-medium', 'seg-large', 'seg-xlarge'],
    },

    'yolov12': {
        'adapter_class': 'YOLOv12Predictor',
        'module': 'mozo.adapters.yolov12',
        'task_type': 'object_detection',
        'description': (
            'YOLO12 by Ultralytics -- real-time object detection with area attention. '
            '5 variants (nano/small/medium/large/xlarge). '
            'The weights are AGPL-3.0, unlike the rest of mozo: serving predictions from them '
            'over a network places obligations on you. See the NOTICE published beside each '
            'checkpoint.'
        ),
        'variants': ['nano', 'small', 'medium', 'large', 'xlarge'],
    },

    'yolov26': {
        'adapter_class': 'YOLOv26Predictor',
        'module': 'mozo.adapters.yolov26',
        'task_type': 'object_detection',
        'description': (
            'YOLO26 by Ultralytics -- real-time object detection and instance segmentation, '
            'NMS-free: the head fires once per object and the network returns a ranked detection '
            'list. 10 variants -- 5 detection (nano/small/medium/large/xlarge) and 5 segmentation '
            '(seg-nano/seg-small/seg-medium/seg-large/seg-xlarge), which add a mask per detection '
            'and change nothing else. '
            'The weights are AGPL-3.0, unlike the rest of mozo: serving predictions from them '
            'over a network places obligations on you. See the NOTICE published beside each '
            'checkpoint.'
        ),
        'variants': ['nano', 'small', 'medium', 'large', 'xlarge',
                     'seg-nano', 'seg-small', 'seg-medium', 'seg-large', 'seg-xlarge'],
    },
}


def get_model_info(family: str, variant: str | None = None) -> dict[str, Any]:
    """Return one family's catalogue entry, optionally checking that a variant is in it.

    Args:
        family: Model family, e.g. ``"rfdetr"``.
        variant: When given, verify the family publishes it. Families with an empty variant
            list accept anything, so there is nothing to check for those.

    Returns:
        The family's entry from :data:`MODEL_REGISTRY`, not a copy of it.

    Raises:
        ValueError: If the family is unknown, or *variant* is not one it publishes. The message
            names what is available, so callers can pass it straight through rather than
            composing a second one.

    Examples:
        >>> get_model_info("rfdetr", "nano")["module"]
        'mozo.adapters.rfdetr'
        >>> get_model_info("rfdetr", "giant")
        Traceback (most recent call last):
        ValueError: Unknown variant 'giant' for family 'rfdetr'. Available: ...
    """
    entry = MODEL_REGISTRY.get(family)
    if entry is None:
        raise ValueError(
            f"Unknown model family: '{family}'. Available families: {list(MODEL_REGISTRY)}")

    variants = entry.get('variants', [])
    if variant is not None and variants and variant not in variants:
        raise ValueError(
            f"Unknown variant '{variant}' for family '{family}'. Available: {variants}")

    return entry

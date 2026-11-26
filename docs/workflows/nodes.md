# Every built-in workflow node

<!--
summary: All 51 nodes mozo's workflows are built from, by category, with their inputs, outputs, parameters and defaults.
label: Nodes
keywords: mozo, workflow nodes, node catalogue, workflow editor, pipeline
-->

Every node a workflow can use, in the editor's categories. Inputs and outputs are port names; a
port's type is its name (`image`, `detections`, `classifications`, `depth`). An output marked
(many) gives several values per item, and every node after it runs once per value. A parameter with no
value shown has no default: it is either optional or, for a file, chosen when you build the
workflow.

Model nodes take a `variant`, one of the family's variants on its [model page](/docs/models/all-models.md).
Nodes that take `text` take a list of phrases, one per line in the editor, each used exactly as
written.

The same list is available from a running server at `GET /workflow/nodes`, and in Python from
`mozo.workflow.catalogue()`. Both are generated from the nodes themselves, so they always match
what runs.

## Input

| Node | Inputs → outputs | Parameters | What it does |
|---|---|---|---|
| `read_media` | none → image | `source`, `stride`=1, `start`=0, `count`, `rate` | Read an image or a video: one frame for the first, every frame for the second. |
| `read_annotations` | image → detections | `path` | The detections `save_annotations` wrote for this image, read back. |

## Output

| Node | Inputs → outputs | Parameters | What it does |
|---|---|---|---|
| `save_image` | image → none | `path`=output, `format`=.png | Write each image to its own file. |
| `save_video` | image → none | `path`=output.mp4, `fps` | Write every frame that reaches this node to one video file. |
| `save_annotations` | image, detections → none | `path`=annotations.jsonl | Write what was detected, one line of JSON per item, as each item finishes. |

## Detect

| Node | Inputs → outputs | Parameters | What it does |
|---|---|---|---|
| `yolov8` | image → detections | `variant`=nano, `threshold`=0.5 | YOLOv8 by Ultralytics: real-time object detection. |
| `yolov11` | image → detections | `variant`=nano, `threshold`=0.5 | YOLO11 by Ultralytics: real-time object detection. |
| `yolov12` | image → detections | `variant`=nano, `threshold`=0.5 | YOLO12 by Ultralytics: attention-centric real-time object detection. |
| `yolov26` | image → detections | `variant`=nano, `threshold`=0.5 | YOLO26 by Ultralytics: NMS-free detection, and instance masks on the seg variants. |
| `rfdetr` | image → detections | `variant`=nano, `threshold`=0.5 | RF-DETR by Roboflow: NMS-free transformer detection, under a permissive licence. |
| `rfdetr_checkpoint` | image → detections | `checkpoint`, `architecture`=nano, `labels`, `threshold`=0.5 | RF-DETR from a checkpoint of your own: a model you trained, run like a published one. |
| `owlv2` | image → detections | `text` (a list of phrases), `variant`=base-ensemble, `threshold`=0.1 | OWLv2 by Google: find anything you can name, with no training. |
| `grounding_dino` | image → detections | `text` (a list of phrases), `variant`=tiny, `threshold`=0.3 | Grounding DINO by IDEA: open-vocabulary detection from a phrase. |

## Segment

| Node | Inputs → outputs | Parameters | What it does |
|---|---|---|---|
| `sam3` | image → detections | `text` (a list of phrases), `variant`=sam3, `threshold`=0.5 | SAM 3 by Meta: masks for every instance of a concept you name. |
| `sam2_boxes` | image, detections → detections | `variant`=tiny | SAM 2 by Meta: a mask for each box it is given, the rows otherwise unchanged. |
| `edgetam_boxes` | image, detections → detections | `variant`=edgetam | EdgeTAM: SAM 2 made small: a mask for each box it is given, the rows otherwise unchanged. |
| `sam3_boxes` | image, detections → detections | `variant`=sam3 | SAM 3 by Meta: a mask for each box it is given, the rows otherwise unchanged. |

## Edit

| Node | Inputs → outputs | Parameters | What it does |
|---|---|---|---|
| `moebius` | image, detections → image | `variant`=general, `seed`=0, `dilate`=0 | Moebius: remove what you point it at and repaint the gap. Wire a segmenter into `detections`. |
| `ben2` | image → image | `variant`=base, `stretch`=True, `refine`=False | BEN2: cut the background away, keeping a soft edge. Takes no prompt and no boxes. |

## Pose

| Node | Inputs → outputs | Parameters | What it does |
|---|---|---|---|
| `vitpose` | image, detections → detections | `variant`=small | ViTPose++: the joints of everyone you point it at. Wire a detector into `detections`. |

## Classify

| Node | Inputs → outputs | Parameters | What it does |
|---|---|---|---|
| `clip` | image → classifications | `text` (a list of phrases), `variant`=base | CLIP by OpenAI: score an image against phrases you make up. |
| `siglip2` | image → classifications | `text` (a list of phrases), `variant`=base-224 | SigLIP 2 by Google: zero-shot classification against phrases you make up. |

## Read

| Node | Inputs → outputs | Parameters | What it does |
|---|---|---|---|
| `easyocr` | image → detections | `variant`=english | EasyOCR by JaidedAI: find text in an image and read it. |

## Depth

| Node | Inputs → outputs | Parameters | What it does |
|---|---|---|---|
| `depth_anything_v2` | image → depth | `variant`=small | Depth Anything V2: how far away everything is, from one photograph. |

## Track

| Node | Inputs → outputs | Parameters | What it does |
|---|---|---|---|
| `track` | detections → detections | `tracker`=bytetrack, `lost_after`=1.0 | Give each object the same `tracker_id` from frame to frame, so it's counted once however many frames it is in. |
| `sam2_track` | image, detections → detections | `variant`=tiny | SAM 2 by Meta: follow objects through a video with a mask on every frame. The first frame's detections start the objects; each keeps its `tracker_id` and the class of the row that started it. |
| `sam3_track` | image → detections | `text`=person, `variant`=sam3 | SAM 3 by Meta: follow every instance of a phrase through a video with a mask on every frame. New instances start new objects; each keeps its `tracker_id`, with the phrase as its class. SAM License weights. |

## Transform

| Node | Inputs → outputs | Parameters | What it does |
|---|---|---|---|
| `rotate` | image → image | `angle`=90.0 | Rotate the image about its centre, in degrees. |
| `flip_horizontal` | image → image | none | Mirror the image left to right. |
| `flip_vertical` | image → image | none | Mirror the image top to bottom. |
| `crop` | image → image | `left`=0, `top`=0, `right`, `bottom` | Cut a rectangle out of the image. Right and bottom default to the image's own edges. |
| `resize` | image → image | `height`=720 | Scale the image to a height, keeping its shape. |
| `to_grayscale` | image → image | none | Drain the colour, keeping three channels so the result is still an image. |
| `rotate_with_detections` | image, detections → image, detections | `angle`=90.0 | Rotate the image and carry the detections round with it. |
| `flip_horizontal_with_detections` | image, detections → image, detections | none | Mirror the image left to right, and the detections with it. |
| `flip_vertical_with_detections` | image, detections → image, detections | none | Mirror the image top to bottom, and the detections with it. |
| `crop_with_detections` | image, detections → image, detections | `left`=0, `top`=0, `right`, `bottom` | Cut a rectangle out, keeping the detections that fall inside it. |
| `align_by_keypoints` | image, detections → image, detections | `first`=left_eye, `second`=right_eye, `target_angle`=0.0, `detection_index`=0 | Rotate so that two named keypoints sit at a chosen angle: how a face is straightened. |
| `crop_around_detections` | image, detections → image (many) | `padding`=0.0 | Cut out one image per detection. |
| `pad_detections` | detections → detections | `padding`=0.1, `reference`=shorter | Grow every box by a fraction of its own size. |
| `bbox_from_keypoints` | detections → detections | `keypoints` | Redraw each box to enclose its keypoints. Unset uses all of them. |

## Adjust

| Node | Inputs → outputs | Parameters | What it does |
|---|---|---|---|
| `enhance_clahe` | image → image | `clip_limit`=2.0, `tile_size`=8 | Lift local contrast, which brings detail out of shadow without blowing out the rest. |
| `auto_contrast` | image → image | `cutoff`=1.0 | Stretch the histogram to the full range, ignoring the brightest and darkest few percent. |
| `gamma_correction` | image → image | `gamma`=1.0 | Brighten or darken without clipping. Below 1 brightens, above 1 darkens. |

## Annotate

| Node | Inputs → outputs | Parameters | What it does |
|---|---|---|---|
| `draw_boxes` | image, detections → image | `thickness`, `color` | Draw a box around each detection. |
| `draw_labels` | image, detections → image | `position`=top_left, `font_scale`, `padding`=6, `text_color`=#FFFFFF, `background` | Write each detection's class and score beside it. |
| `draw_masks` | image, detections → image | `opacity`=0.5, `color` | Shade each detection's mask over the image. |
| `draw_polygons` | image, detections → image | `thickness`, `color` | Outline each detection's mask rather than filling it. |
| `draw_keypoints` | image, detections → image | `radius`, `thickness`, `color`, `show_names`=False, `min_confidence`=0.0 | Mark each detection's keypoints. |
| `draw_skeleton` | image, detections → image | `thickness`, `color`, `min_confidence`=0.0 | Join each detection's keypoints into a skeleton. |
| `blur_regions` | image, detections → image | `kernel_size`, `padding_percent`=0.05 | Blur whatever was detected: faces, plates, anything you would rather not publish. |
| `pixelate_regions` | image, detections → image | `pixel_size`, `padding_percent`=0.05 | Pixelate whatever was detected, where a blur would look like a mistake. |

## Prompts read back the same from every model

The prompted nodes are `sam3`, `owlv2`, `grounding_dino`, `clip` and `siglip2`. Each result's
`class_id` is the index of the phrase that found it (for CLIP and SigLIP 2, of the phrase it
scored), and its `class_name` is that phrase. So results read the same whichever model produced
them. A list with an empty entry is refused rather than shortened, because shortening it would
change every later phrase's `class_id`.

Write phrases as short noun phrases: `"hard hat"`, not `"a photo of a hard hat"`. The models that
locate things find nothing for a caption. On the example photograph, SAM 3 and Grounding DINO each
found six or more people for `"person"` and none for `"a photo of a person"`, while CLIP scored both
the same.

## Boxes into masks

`sam2_boxes`, `edgetam_boxes` and `sam3_boxes` add a mask to every box they are given, in one call
per image, and keep every row as it came: same order, class, name, score and box. A row whose mask
comes back empty keeps its box with an empty mask. An image with no boxes never runs the model.
`sam3_boxes` after a `sam3` node on the same image reuses that node's image encode.

Three graphs cover most annotation jobs:

| Job | Graph |
|---|---|
| Find objects by name, with masks | `read_media` → `sam3` |
| Find boxes, then outline them | `read_media` → `grounding_dino` → `edgetam_boxes` |
| Give boxes you already have masks | `read_media` + `read_annotations` → `edgetam_boxes` |

`rfdetr_checkpoint` runs an RF-DETR you trained: choose the checkpoint file, the architecture it
was trained from, and its class names one per line in class-id order, or leave the names empty to
use the ones the checkpoint carries.

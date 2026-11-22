# Run a workflow over videos and folders

<!--
summary: Run a mozo workflow over every frame of a video or every image in a folder: sample frames, track objects, and save videos and annotations.
label: Videos and folders
keywords: mozo, video processing, object tracking, bytetrack, batch annotation, workflow
-->

A workflow built on one image runs over a whole video unchanged: `read_media` yields every frame,
and each frame goes through the graph. This page covers reading frames, knowing which frame a result
belongs to, tracking objects across frames, and writing results back out.

## Read a video

`read_media` reads an image, a video, or a folder of images. Give it a video and it yields every
frame. Two parameters take fewer:

| Parameter | Effect |
|---|---|
| `stride` | Take every n-th frame: `stride=5` keeps one frame in five. |
| `rate` | Take this many frames a second, chosen by the frames' own times. |
| `start` | Skip this many frames first. |
| `count` | Stop after this many frames. |

Use `stride` or `rate`, not both. `rate` samples evenly even when a phone video's frame rate
changes as it plays. Either way the source tells the rest of the workflow the rate it yields, so
`save_video` writes a video that plays at the right speed and `track` measures time correctly.

This workflow reads a street video at six frames a second, finds people with RF-DETR, tracks them,
draws their boxes into a new video, and logs every frame's detections:

```json
{
  "name": "Track people",
  "nodes": [
    {"id": "read", "type": "read_media", "source": "street.mp4", "rate": 6},
    {"id": "find", "type": "rfdetr", "variant": "nano"},
    {"id": "track", "type": "track"},
    {"id": "boxes", "type": "draw_boxes"},
    {"id": "save", "type": "save_video", "path": "tracked.mp4"},
    {"id": "log", "type": "save_annotations", "inputs": {"image": "read"}, "path": "tracked.jsonl"}
  ]
}
```

```bash
mozo run track_people.json
```

## Know which frame a result belongs to

`process()` runs the workflow over everything its source produces and yields each item as it
finishes. With `context=True` it also yields the item's place in the run:

```python
from mozo.workflow import Workflow

workflow = Workflow.load("track_people.json")

for image, results, run in workflow.process(context=True):
    print(run.index, run.frame, round(run.time, 3), len(results["track"]))
```

```text
0 0 0.0 11
1 4 0.167 11
2 8 0.333 10
```

| Field | Meaning |
|---|---|
| `run.index` | The item's place in this run, from 0. |
| `run.frame` | Which frame of the file it is, from 0, correct with `stride`, `rate` and `start`. `None` for an image or a folder. |
| `run.time` | That frame's time in the file, in seconds. With `start` it does not begin at 0. |
| `run.frames` | How many items the source expects, or `None` for a live stream. |
| `run.fps` | The rate the source yields, after `stride` or `rate`. |

Use it when your own code stores the results and has to know where in the video each one belongs.
Without `context=True`, `process()` yields `(item, results)`.

## Track objects across frames

`track` gives each object the same `tracker_id` from frame to frame, so it is counted once however
many frames it is in. Wire a detector's detections into it. The trackers are PixelFlow's:
ByteTrack by default, or SORT, OC-SORT and C-BIoU. Each run keeps its own tracker, told the run's
rate and each frame's time, so a video read at `rate=6` is tracked at six frames a second.

`lost_after` is in seconds: how long an object may go unseen before one seen again counts as a new
object. Over the four-second clip above, ten distinct `tracker_id`s were assigned. An object gets
its id once the tracker has confirmed it, so on the first frame detections can have none.

A node that needs these facts reads them the same way: see
[Write a node](/docs/workflows/writing-nodes.md#facts-about-the-run).

## Save results

| Node | Writes |
|---|---|
| `save_video` | One video file from every frame that reaches it, at the run's rate. |
| `save_image` | One image file per item into a folder. A video frame's file is named by its frame number. |
| `save_annotations` | One line of JSON per item: its index, label, size, and for a video `frame` and `time`, then every detection. |

A line of `save_annotations`, shortened:

```json
{"index": 1, "label": "street_000004", "width": 1280, "height": 720, "time": 0.1667, "frame": 4,
 "detections": [{"bbox": [751.89, 318.92, 806.11, 454.03], "class_id": 1, "class_name": "person",
                 "confidence": 0.894, "tracker_id": 1, "total_time": 0.0}]}
```

`read_annotations` reads that file back, matching each line to its image by file name and checking
the line was written for an image of that size. Feed it files, a folder or a list of paths, not
images in memory, which have no name to match. Names must be unique within one file: the name has
no extension, so `a.jpg` and `a.png` clash. To write the same format from your own code, use
`mozo.workflow.annotation_line(detections, label=..., width=..., height=...)`, the function
`save_annotations` itself uses.

## Annotate a folder

Give `read_media` a folder and it yields every image in it, so the same `process()` loop annotates
a dataset. To run over a list you build yourself, use `run_many` with the paths:

```python
from pathlib import Path

for path, results in workflow.run_many(sorted(Path("photos").glob("*.jpg"))):
    ...
```

[Run a workflow](/docs/workflows/python.md#run-it-over-many-images) covers `run_many`'s workers and
error handling.

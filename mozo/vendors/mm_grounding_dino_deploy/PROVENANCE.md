# Provenance

This package is a deployment-only extraction of **MM-Grounding-DINO**'s detection path, serving
**LLMDet**'s checkpoints.

## The reference, and why

The checkpoints mozo publishes are `iSEE-Laboratory/llmdet_{tiny,base,large}`: LLMDet's weights in
`transformers`' format, published by the authors in their own Hugging Face organisation, and named
in the LLMDet README as the way to run the model (`transformers>=4.55`). What those files reproduce
is `transformers`' `MMGroundingDinoForObjectDetection`, so that is the reference -- **pinned at
4.57.3**, with its `AutoProcessor` (the slow, NumPy `GroundingDinoImageProcessor` and
`BertTokenizerFast` over `tokenizers` 0.22.2), on the CPU, torch 2.13. 4.55.0, the version the
README names, gives the same numbers to the bit.

**`transformers` 5.x runs a different model from these files.** Its loader ties
`bbox_embed.1`-`5` to `bbox_embed.0` and every score bias to layer 0's (a `_tied_weights_keys`
regex), so all six decoder layers refine boxes with the first layer's head. Measured on 5.9.0 and
5.12.1: the loaded tensors differ from the file's. Upstream fixed it on 2026-10-21 ("Only tie MM
Grounding DINO decoder heads when the config shares them", #49552).

**Where the chain stops.** The authors trained with MMDetection; the HF files are a conversion of
their `.pth` checkpoints (`huggingface.co/fushh7/LLMDet`), done by the `transformers` port's author
and published under the authors' organisation. That the conversion matches MMDetection is
upstream's claim. mozo measures the link from the HF files to this package, exactly, and nothing
before it.

## What was taken

| from (`transformers` 4.57.3) | what |
|---|---|
| `models/mm_grounding_dino/modeling_mm_grounding_dino.py` | the assembly, the contrastive head with its scale and bias, per-layer heads, encoder (fusion, text enhancer, deformable layers), query selection, decoder, sine embeddings, phrase masks |
| `models/swin/modeling_swin.py` | `SwinBackbone`: patch embedding, windowed attention, patch merging, per-stage norms |
| `models/bert/modeling_bert.py` | the BERT encoder, SDPA attention -- **weights from the checkpoint**, not Google's |
| `models/grounding_dino/image_processing_grounding_dino.py` | the 800/1333 resize, rescale and normalise |
| `models/grounding_dino/processing_grounding_dino.py` | joining labels into one caption; the score and box decoding of `post_process_grounded_object_detection` |
| `bert-base-uncased` | `vocab.txt`, and the WordPiece algorithm |
| `tokenizers` (by probing) | `assets/bert_chars.json.gz`: what the normaliser and pre-tokenizer do to every codepoint |

The checkpoint's tensors are published unchanged under their own names; `checkpoint.py` maps the
names and drops five unused scalars (the score biases of decoder layers 0-4, read only by the
training loss). Strict load.

## What was deliberately left behind

Training: the loss, denoising, auxiliary per-layer outputs, gradient checkpointing, drop-path and
dropout. The one-stage branch and the learned position embedding (no published config uses them).
The timm and ResNet backbone options. The kernel-hub deformable attention (the pure-torch
`grid_sample` path is the one carried, and the one the reference took). `encoder_logits` (computed
upstream for the loss, never read at inference). The language model LLMDet was trained with --
it is not in the published checkpoints.

**MM-Grounding-DINO's own checkpoints** (`openmmlab-community/mm_grounding_dino_*`) run on exactly
this code -- their configs are identical per size -- and are not published, because LLMDet is
fine-tuned from them and is the stronger detector at every size on zero-shot LVIS. Adding one is a
`SPECS` row and a fetch row.

## Where this diverges, and why

### 1. The name on a detection is the caller's phrase, not a decoded span

As in mozo's `grounding_dino`, and for the same reasons: upstream decodes the tokens above
`text_threshold` back into a string, which can be a fragment; mozo reports which prompt the
query's peak token belongs to, verbatim, and drops a query whose peak is a separator. Scores and
boxes are upstream's post-processing exactly (checked by the gate); only the string differs, and
`text_threshold` is not carried.

### 2. One caption per image, restarted separator scan

`phrase_masks` restarts its separator scan per batch row, where upstream carries the previous
row's last separator into the next. With one caption per call the two are identical.

### 3. The tokenizer reads Unicode from a table, not from the interpreter

Written against Python's `unicodedata` (Unicode 15.0 on Python 3.12), the tokenizer disagreed with
the reference on 826 assigned codepoints -- combining marks added in Unicode 14-15 that one side
strips and the other keeps, punctuation the Rust tables predate, CJK Extension I -- and on all
825,345 codepoints Python calls unassigned, which it dropped as control characters where the
reference emits `[UNK]`. `assets/bert_chars.json.gz` records the reference's own answer for every
codepoint; with it the two agree on all 1,112,064, and on 40,000 random multi-character prompts.
This is not a change of behaviour but the removal of one: the vendor's answer no longer depends on
which Python runs it. (mozo's `grounding_dino` tokenizer has the same exposure; noted, not changed
here.) The one interpreter-dependent step left is upstream's own: the processor lowercases each
label with `str.lower` before tokenizing, and `caption_for` does the same.

### 4. Written to export

Level sizes travel as Python integers beside the `spatial_shapes` tensor (upstream carries the same
`spatial_shapes_list`), `forward` takes an optional padding mask for the fixed-square path
(`fit_square`), and `export.py` is mozo's. None changes a number on upstream's path.

## Traps

Each runs, produces plausible output, and is wrong.

- **`transformers` 5.x head tying** (above). A strict load would not catch it either: the keys all
  match; the values are copied over.
- **Per-layer heads under two prefixes.** The file stores layer 0's box head as `bbox_embed.0` and
  layers 1-5 as `model.decoder.bbox_embed.1`-`5` (the writer saves each shared tensor once).
- **Not Grounding DINO's resize.** When the 1333 cap binds, this processor scales the long side
  from the *unrounded* short side: 500x4000 becomes 167x1333, not 167x1336.
- **Rescale in float64, normalise in float32**, not torchvision's `to_tensor`/`normalize`.
- **The contrastive score is `q.t / sqrt(256) + bias`.** Grounding DINO's is a bare dot product.
- **`logit(eps=1e-5)`**, not Grounding DINO's `inverse_sigmoid(eps=1e-3)`.
- **Boxes are refined from the decoder layer's raw output; the heads read the normalised one.**
- **The fusion residual is added to the normalised input**, and its first softmax subtracts one
  global maximum.
- **Temperature 20** for image positions, 10000 for text and box positions; boxes swap x and y.
- **BERT runs SDPA** when loaded the way the model card loads it; eager is a different float.
- **Swin's `relative_position_index` is a persistent buffer** in the checkpoint; it is loaded,
  and a test holds it to the window geometry.
- **`SwinBackbone` passes `always_partition=True`**, so `SwinLayer`'s window-shrinking branch --
  which mutates the layer's window size in place on a small feature map -- is never taken.

## Numbers

Parity, on the fixture photograph and three crops of it (a panorama that makes the size cap bind,
a portrait strip, a 37x45 corner), five prompt sets each, CPU, `MKL_CBWR=AVX2,STRICT`:
pixels, token ids, BERT, projected text, every Swin level, encoder image and text memory, each
decoder layer's output and reference boxes, logits, boxes, and post-processed detections --
every one `torch.equal`. The tokenizer fuzz adds 20,020 prompts per run. See the family's plan
results for counts per variant and the falsification runs.

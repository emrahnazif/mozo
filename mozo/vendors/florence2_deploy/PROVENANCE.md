# Provenance

This package is a deployment-only extraction of **Florence-2** (Microsoft): its DaViT vision
tower, its BART encoder-decoder, the decoding `generate()` runs by default, and its post-processor
for two of its tasks, `<OD>` and `<OCR_WITH_REGION>`.

## The reference, and why

Microsoft publishes Florence-2 only as Hugging Face model repositories that carry their own
implementation (`modeling_florence2.py`, `processing_florence2.py`, `configuration_florence2.py`,
loaded with `trust_remote_code=True`). There is no GitHub implementation
(`github.com/microsoft/Florence-2` does not exist; Microsoft's only Florence-2 repository there is a
fine-tuning toolkit). So the remote code *is* the authors' implementation, and it is the one the
published weights are run with on every model card. That is the reference (VENDORING §1.1).

**Pinned:** `microsoft/Florence-2-base` @ `5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac` (code and
post-processor, for every variant), each variant's own repository for its modeling code and
weights (below), run under **transformers 4.53.3**, torch 2.13.0, on CPU in float32 with
`attn_implementation="eager"`, `MKL_CBWR=AVX2,STRICT`.

- **4.53.3 is the newest transformers the remote code runs on.** 4.49.0, 4.51.3 and 4.53.3 run it,
  with every decoding step's scores bit-identical across the three on the fixture; 4.54.1 and 4.55.4
  fail inside it (`Cache only has 0 layers`), 4.56.2 and 4.57.6 too (`past_key_values[0][0]` is
  `None`).
- **The post-processor comes from the base repository for every variant.** The `-ft` repositories
  carry an older `processing_florence2.py` whose `post_process_generation` predates the `<OD>`
  scores (Microsoft's commit "add confidence scores for `od` and `description_with_bboxes` tasks",
  2025-08-04, landed in base and large only). The scoring is a function of the decoder's output,
  so it applies to every checkpoint; mozo uses the newest copy of Microsoft's own code. The
  modeling files of all repositories differ only in weight initialisation and
  `resize_token_embeddings`, never in arithmetic.
- **Not the reference: transformers' native `Florence2ForConditionalGeneration`** (5.x). It is a
  port onto converted checkpoints (`florence-community/*`, not a Microsoft organisation) with
  renamed keys, `image_projection` transposed and the vocabulary padded from 51,289 to 51,328 rows
  with 39 non-zero rows the output layer can score. Measured: given the same pixels it wrote the same
  tokens on 8 tasks, but its per-step scores differ (up to 2.4e-04), and with its default
  torchvision image processor the pixels move by one 8-bit level and `<OD>` came out 6 tokens
  different.

**What mozo guarantees:** for the same image, on CPU in float32, the same `pixel_values`, the same
output of each DaViT stage, the same projected image tokens, the same encoder input and each
encoder layer's output, **the same processed log-probabilities of all three beams at every decoding
step**, the same number of steps, the same sequence, beam indices and transition scores, and the same
parsed boxes, labels and scores (floats compared with `==`) -- `tools/verify/florence2_detect.py`.

**Where the chain stops.** Microsoft's benchmark numbers (COCO detection 34.7 / 37.5 / 41.4 / 43.4
mAP for base / large / base-ft / large-ft on the model card) are claimed, not reproduced: they were
measured with Microsoft's own evaluation, and for `large` with the checkpoint Microsoft replaced in
December 2024. On CUDA and in fp16 the results are not bit-identical (measured below).

## The weights

| variant | repository @ commit | file, sha256 (Hub LFS record, re-hashed after download) |
|---|---|---|
| `base` | `microsoft/Florence-2-base` @ `5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac` | `model.safetensors` 463,221,266 B, `03075d2d2d2bbd3e180b9ba0afae4aa8563226e2d32911656966e05b2f2ee060` |
| `large` | `microsoft/Florence-2-large` @ `21a599d414c4d928c9032694c424fb94458e3594` | `model.safetensors` 1,553,563,458 B, `4f38ce741c6b71188fe2b3419a55e11917a8a7b321ae2e63c61da0191b0ebad7` |
| `base-ft` | `microsoft/Florence-2-base-ft` @ `f6c1a25888ffc1d945ee8a1a77ac833c7303d46e` | `model.safetensors` 463,221,266 B, `58757d657ff44051314c8030b68e04cb1bb618ca9a4885418f111f6fb708185a` |
| `large-ft` | `microsoft/Florence-2-large-ft` @ `4a12a2b54b7016a48a22037fbd62da90cd566f2a` | `model.safetensors` 1,540,980,506 B, `8b4e610c952eef90a836c56cda0f398a672a3a6ca7b4d96b0e09a86dee42e2c3` |

**Licence: MIT, per checkpoint**, stated on the checkpoint itself: each model card's metadata says
`license: mit` with `license_link` to the repository's own `LICENSE`, and that file is present
with the MIT text ("Copyright (c) Microsoft Corporation"); none of the four is gated (checked
2026-11-25). mozo publishes that `LICENSE` file, fetched from the same repository and commit.

**What the files contain.** Every tensor is float16 -- Microsoft's 2024 `pytorch_model.bin` files are
float16 too, so this is the checkpoint as released. 666 tensors for base, 918 for large: the vision
tower, `image_projection`, `image_proj_norm`, the learned 2-D position table, the cosine temporal
table (`visual_temporal_embed.pos_idx_to_embed`, stored in fp16 and *loaded*, not recomputed -- so
the values that run are the rounded ones, as upstream), and the BART model with one embedding table
(`shared`) that the encoder, decoder and output layer all use, and an all-zero
`final_logits_bias`. mozo re-saves the file with `torch.save` (no `safetensors` dependency): nothing
renamed, cast or dropped. `torch-fp32` is the same file upcast at load, which is exact.

**`large` is Microsoft's December-2024 replacement, chosen by measurement.** Commit `00d2f1570b`
("initial tuned 4k context length model") replaced the weights; the card now says the model is "a
continued pretrained version ... with 4k context length, only 0.1B samples ... thus it might not be
trained well", and its text model has 4,096 positions (the others 1,024). Both revisions were scored
on the same 500 COCO val2017 images, fp32, RTX 3060, the same way (a box counts when its label is a
COCO name): **current 36.98 AP / 47.95 AP50, original (`f0acedbf9b`) 36.77 / 48.43.** Within noise
of each other, so mozo serves the current revision: it is what `from_pretrained` gives everyone
today, it is published as safetensors with a Hub-recorded hash, and the original's
`pytorch_model.bin` would need its three stored copies of the tied embedding dropped.

**The vocabulary** (`assets/florence2_vocab.json.gz`, 205,319 B, sha256
`82087a6965003e0e8b6ca9d5f3a678aaa700c4d2c7dcdc95b321478f8ca978de`) is the 50,265 pieces of
`tokenizer.json` (identical in all repositories) plus the 1,024 tokens `Florence2Processor` adds at
load, in its order, derived by `tools/fetch/florence2_detect.py --derive-vocab`.

## What was taken

| from | what |
|---|---|
| `modeling_florence2.py` | `DaViT` and its blocks (`vision.py`); `Florence2Encoder`, `Florence2Decoder`, their layers and `Florence2SdpaAttention` (`language.py`); `_encode_image`, `_merge_input_ids_with_image_features`, `LearnedAbsolutePositionEmbedding2D`, `PositionalEmbeddingCosine1D` (`network.py`) |
| `processing_florence2.py` | the task prompts (as token ids, `tokenizer.py`), `decode_with_spans`, `BoxQuantizer`/`CoordinatesQuantizer` dequantisation, `parse_description_with_bboxes_from_text_and_spans` with `avg_loc_scores`, `parse_ocr_from_text_and_spans` (`parse.py`) |
| `preprocessor_config.json` + transformers' slow `CLIPImageProcessor` | 768 x 768 PIL bicubic, float64 rescale, ImageNet normalisation (`image.py`) |
| `config.json` (`text_config`) | the geometry (`config.py`) and the default decoding: 3 beams, `no_repeat_ngram_size=3`, `early_stopping`, forced BOS/EOS |
| transformers 4.53.3 `generation/utils.py`, `logits_process.py`, `stopping_criteria.py` | `_beam_search` and helpers, the three processors, the two stopping criteria, `compute_transition_scores` (`decoding.py`, Apache-2.0) |

## What was left behind

Every other task (captions, plain OCR, phrase grounding, open-vocabulary detection, dense region
captions, region proposals, segmentation, region-to-text), sampling, greedy decoding, flash
attention, training code (drop path, initialisation), the classification head and final norm of
DaViT (Florence-2 deletes them itself), and video input (`T > 1` frames).

`read()` (`<OCR_WITH_REGION>`) is vendored and gated but not served: upstream gives it no score,
PixelFlow's `from_arrays` requires one, and mozo does not invent one. Open-vocabulary detection
additionally needs a text encoder for user phrases, which is not built.

## Where this diverges from the obvious reading, and why

- **The attention is SDPA, not eager.** The reference is loaded with
  `attn_implementation="eager"`, but that reaches only the outer config; the BART model's config
  keeps transformers' default and builds `Florence2SdpaAttention` everywhere (found by the gate at
  the first encoder layer: eager differs by 2e-07). With no padding, every mask is `None`, so
  `scaled_dot_product_attention` runs unmasked and without `is_causal` -- as here.
- **The encoder runs on one row and its output is repeated for the three beams,** and the
  cross-attention keys/values are projected from the repeated rows, as `generate` does.
- **Each token is decoded on its own** when finding spans, as upstream: a character split across
  two byte-level tokens becomes replacement characters.
- **The sequence path, not the model card's text path.** The card parses a decoded string, which
  yields no scores and leaves the decoder's leading `</s>` in the first OCR line
  (`'</s>Hello World'`). mozo parses the sequence with its start token dropped, as
  `post_process_generation(sequence=...)` does.
- **Upstream behaviour kept:** non-ASCII characters are dropped from `<OD>` labels; of the two
  `replace` calls on a phrase only the second (`<obj>`) takes effect; boxes are bin centres; the
  half-precision clamp in the encoder layer (a no-op in fp32) is kept for the fp16 runtime.

## Gate

`tools/verify/florence2_detect.py`, CPU, float32, `MKL_CBWR=AVX2,STRICT`, transformers 4.53.3:

| variant | fixtures | comparisons | machine |
|---|---|---|---|
| `base` | 13: the photograph, the photograph cut at 24 tokens (forced `</s>`), a small copy, a 120 x 1,281 strip, a blank image, RGBA PNG and greyscale JPEG through `load_image`, a text image under `<OD>`, and five text images under `<OCR_WITH_REGION>` (printed, rotated, CJK, vertical, blank) | 803, mozo's served path included | this Linux box, 4 threads |
| `large` | 3: the photograph, the 24-token cut, printed text under OCR | 215 | vision-003, CPU |
| `base-ft` | the 13 | 815 | vision-003, CPU |
| `large-ft` | the 3 | 263 | vision-003, CPU |

All 2,096 equal. **Falsified, 12 perturbations, each caught at the stage it reaches first**
(`--falsify all`): bilinear resize and PIL's `reducing_gap` at `pixel_values`; channel attention
scaled by the group width at `vision_stage_0`; row/column position tables swapped and the temporal
embedding dropped at `image_features`; BART's position offset 0 at `encoder_layer_0`; decoder
LayerNorm eps 1e-6 at `step_1`; the n-gram ban off at `step_6` (on the blank image, whose beams would
repeat a trigram); the forced `</s>` off at `step_23` of the 24-token case; length penalty 2 at
`sequence` (it only ranks finished hypotheses, never shows in a step's scores); the OD score read
from shifted tokens at `parsed_scores`; boxes without the half-bin at `parsed_boxes`.

## Runtimes (measured; nothing beyond torch published)

- **torch-fp32**: the gate. CPU latency on the example photograph, `<OD>`, vision-003 (8 threads),
  median of 5, interleaved with the reference on the same weights: base 2,837 ms vs 2,979 ms
  (reference), large 8,327 ms vs 8,614 ms.
- **torch-fp16 on CUDA** (the published file as is): `auto`'s choice on CUDA. RTX 3060, 500 COCO
  val2017 images: AP fp32 → fp16 base 34.19 → 34.41, large 36.98 → 36.89, base-ft 44.40 → 44.47,
  large-ft 46.09 → 45.81; 97.5–99.0% of fp32's boxes returned with the same label and IoU ≥ 0.9;
  about twice as fast (0.34 → 0.18 s per image for base, 0.82 → 0.40 for large).
- **ONNX** of the encoder half (vision tower + projection + BART encoder over the fixed `<OD>`
  prompt -- the only part with static shapes): exports cleanly. ONNX Runtime on vision-003's CPU:
  base 1.12x faster than torch on the encoder (1,667 vs 1,859 ms) with max |d| 8.3e-03 on its
  output; large 1.02x (7,910 vs 8,107 ms), max |d| 8.7e-04. The encoder is 59% (base) / 64%
  (large) of a CPU call, so at most a 6% saving, with drift that would move boxes. Not published.
- **On the GPU** (RTX 3060, vision-002, TensorRT 11.5 through mozo's `TensorRTRunner`, TF32 off):
  the encoder half is 30% (base: 112 of 377 ms) / 39% (large: 342 of 885 ms) of an fp32 `<OD>`
  call; the rest is the 3-beam decoder loop, which is not a graph. TensorRT fp32: 1.35x on the
  encoder (83 / 254 ms), max |d| 1.6e-02 / 7.7e-04 -- about a tenth off the whole call, with drift.
  TensorRT fp16: 4.5x / 5.2x on the encoder, max |d| 19.8 / 4.8 -- broken outputs. ONNX Runtime's
  CUDA provider (1.21.1): 91 / 238 ms, max |d| 2.45 / 0.39. None published: `torch-fp16` already
  halves the whole call with COCO AP unchanged.
- **Latency on CUDA**, `<OD>` on the example photograph, median of 10, interleaved with the
  reference: base fp32 380 ms (reference 401), fp16 222 ms (241); large fp32 889 ms (909), fp16
  447 ms (469); base-ft fp32 627 / fp16 373 ms; large-ft fp32 1,246 / fp16 654 ms (the `-ft`
  checkpoints write more boxes on this photograph).

# Provenance

This package is a deployment-only extraction of **Qwen3-VL**'s inference path, for one image and
one prompt.

It derives from `transformers` v5.24.1 (Apache-2.0, © 2025 The Qwen Team and The HuggingFace Inc.
team): `models/qwen3_vl/modeling_qwen3_vl.py`, `models/qwen2_vl/image_processing_qwen2_vl.py`, and
the generic code those call that changes a number on this model's path. The Qwen team trained the
weights and publishes them for `transformers` only: the model card's one code path is
`Qwen3VLForConditionalGeneration.from_pretrained` and `generate`, and the authors' repository
(`QwenLM/Qwen3-VL`, Apache-2.0) holds cookbooks and fine-tuning code, not a model. So the
`transformers` implementation is the one whose numbers the published checkpoints reproduce
(VENDORING §1.1).

**The weights are Apache-2.0**, stated in each model card's metadata (`license: apache-2.0`) on
the checkpoints themselves, served ungated (`gated: false`, Hub API, 2026-11-25). The repositories
carry no LICENSE file; mozo publishes the Apache-2.0 text beside each checkpoint.

## What mozo guarantees, exactly

**On the CPU in float32, with SDPA attention, mozo's answer is `generate`'s, token for token, and
every logit on the way is bit-identical.** `tools/verify/qwen3_vl.py` compares, with `torch.equal`
and no tolerance: the pixels, the prompt ids, the patch embedding with its interpolated position
table, each of the 24 vision blocks, the three DeepStack features and the merger, the input
embeddings with the image scattered in, each decoder layer at prefill, the final norm, and then
every decoding step's float32 logits, the chosen ids and the decoded string. Greedy and sampled:
for sampling, `transformers` runs after `torch.manual_seed(seed)` and mozo draws from a
`torch.Generator` seeded the same, and the sampled ids agree exactly. Through three paths: this
package, the reference, and mozo's adapter loaded through the registry and the weights tree.

**`2b`: 918 comparisons on 12 fixtures, all `torch.equal`** -- every pre-decoding stage (61) plus
every step's logits, the ids and the text: the photograph at the default 2.6 MP budget (2,414
prompt tokens, 32 steps); two sampled cases (t=0.7/p=0.8/k=20, seed 3; t=1.3, seed 11); an answer
that stops on its own (2 tokens); a 9x13 image scaled up to the pixel floor; a 2000x120 strip; a
512x512 image (no resize); the photograph at a 262,144-pixel budget; an RGBA PNG and a greyscale
JPEG through `load_image`; CJK + emoji + a combining mark + a newline; a 630-token prompt. The
mozo path (registry, `file://` weights, adapter, `auto` → `torch-fp32` on CPU) agreed on the
photograph and both sampled cases. **`4b`: 271 comparisons on 3 fixtures (the photograph at 2.6 MP, the sampled case, the stopping case), all `torch.equal`, mozo path included.**

Falsified on 13 perturbations (`--falsify all`), each caught at the first stage it reaches: the
resize's antialias, `round` in `smart_resize`, the position table's corner alignment, the vision
GELU, the mergers' LayerNorm epsilon, the vision rotary positions (×(1+1e-7), 2.1e-06), M-RoPE's
interleaving, DeepStack's layers, the text rotary base, the RMSNorm epsilon, the warpers' order, the
seed, and the tokenizer's NFC.

**Claimed only against `transformers` 5.24.1.** 4.57.3 gives the same pixels, the same prompt ids
and the same 64 greedy tokens on the example photograph, and different logits at every one of the
64 steps (4.0e-05 at the first, up to 4.9e-04): its vision tower interpolated the position table
differently. The gate prints the versions it ran under (torch 2.13.0, torchvision 0.28.0,
transformers 5.24.1, tokenizers 0.23.3).

**Not claimed:** any benchmark score. Qwen's published numbers were measured with sampling and
their own settings; mozo does not re-run them. bfloat16, and any GPU, are outside the bit-exact
guarantee (see *Runtimes*).

## What was taken

| from | what |
|---|---|
| `models/qwen3_vl/modeling_qwen3_vl.py` | the vision tower (patch embed, position interpolation, 2-D rotary, blocks, mergers, DeepStack), the text decoder (q/k-normed attention, M-RoPE, MLP, DeepStack injection), `get_rope_index`, the image scatter, the tied `lm_head` |
| `vision_utils.py` | `get_vision_interpolation_indices_and_weights` (bilinear, `align_corners=True`), `get_vision_position_ids`, the cumulative lengths: rewritten for one image in `vision.py` |
| `models/qwen2_vl/image_processing_qwen2_vl.py` + `image_processing_backends.py` | `smart_resize`, the torchvision resize, the folded rescale-and-normalise, `patchify`: `image.py` |
| `integrations/sdpa_attention.py`, `masking_utils.py` | the one SDPA call each attention makes (no mask, `is_causal` with more than one query, `enable_gqa` in the text decoder) |
| `cache_utils.py` | `DynamicLayer`'s growing `torch.cat` cache: `language.Cache` |
| `generation/utils.py`, `generation/logits_process.py` | the decoding loop's arithmetic: last-row `lm_head` (`logits_to_keep=1`), float32 logits, `argmax`, or temperature → top-k → top-p → `softmax` → `torch.multinomial`; the two stop ids |
| `chat_template.json` | the one-turn, one-image, one-text rendering of the template, as a format string (`predictor.template`) |
| `tokenizer.json` at commit `89644892…` | vocabulary, merges, added tokens, re-encoded into `assets/qwen_bpe.json.gz` |
| the `tokenizers` library, probed | the character classes, case folding and NFC boundaries in the same asset (*Divergences* 4) |
| each variant's `config.json` | the geometry, written out in `config.py` |

`assets/qwen_bpe.json.gz` is 1,374,410 bytes, sha256
`c5b19bc0854d29793c388056463c0c91a362014f295870448c9911e949f129f4`, rebuilt by
`tools/fetch/qwen3_vl.py --derive-vocab` under Python 3.10 with `tokenizers` 0.23.3.

The checkpoints are consumed as Qwen publishes them, modulo one repack: `tools/fetch/qwen3_vl.py`
verifies each safetensors file against the Hub's sha256 and writes the same tensors as one ordinary
state dict, still in bfloat16. Nothing is renamed or dropped. The keys are this package's module
names behind `model.`; there is no `lm_head.weight` (tied to the embedding).

| variant | upstream @ commit | source files (sha256) | published |
|---|---|---|---|
| `2b` | `Qwen/Qwen3-VL-2B-Instruct` @ `89644892e4d85e24eaac8bacfd4f463576704203` | `model.safetensors` `7de1838c…977a0` | `torch-bf16.pth`, 625 tensors, 4,255,268,482 B |
| `4b` | `Qwen/Qwen3-VL-4B-Instruct` @ `ebb281ec70b05090aa6165b016eac8ec08e71b17` | two shards, `30a01a05…3339a9`, `046296a2…d002a6` | `torch-bf16.pth`, 713 tensors, 8,875,871,858 B |

## What was left out

- **Video, several images, multi-turn chat, system prompts, tools.** One image, one prompt.
- **Beam search, assisted decoding, repetition/presence penalties, min-p, typical-p** and the rest
  of `generate`'s options. Temperature, top-k and top-p are carried, in upstream's order.
- **FlashAttention and the packed-sequence path**; eager attention (a different reduction order).
- **8B** (16.3 GB, a 27-block vision tower and an untied head: a third geometry for a size that is
  slow on a CPU and does not fit a 12 GB GPU in bf16). The most downloaded size, and the largest
  omission. **Thinking variants** (they spend hundreds of tokens on a reasoning trace before
  answering). **MoE sizes, FP8 and GGUF repositories.**
- **Qwen3.5**, the successor: 3 of every 4 decoder layers are Gated DeltaNet linear attention. Its
  own family, later.

## Divergences from upstream, and why

1. **Upstream's processor default lets an image through at up to 16,777,216 pixels; mozo's default
   is 2,621,440** (`max_pixels`, ≤ 2,560 image tokens), and a caller may raise it to upstream's
   ceiling. Same `smart_resize` rule; the gate hands the reference the same budget. A 3840×2562
   photograph at upstream's default ran out of memory on a 24 GB RTX 3090 in fp32.
2. **A prompt containing one of the 26 added tokens' text is refused.** Upstream tokenizes
   `<|im_end|>` in a question as the end of the user's turn and `<|image_pad|>` as part of the image
   (which then fails its count check); neither is reading the question.
3. **Greedy is the default; upstream's `generation_config.json` samples** (`temperature 0.7, top_p
   0.8, top_k 20`), so the model card's example gives a different answer on every run. mozo samples
   only when asked (`temperature > 0`), with an explicit `seed` (default 0), so a call is
   reproducible either way.
4. **The tokenizer is rebuilt from its parts, standard library only.** `\p{L}`, `\p{N}`, `\s`, the
   `(?i:...)` folding of the contractions (`ſ` folds onto `s`) and NFC are read out of the reference
   by probing every codepoint, not borrowed from Python or `regex`, so the ids do not depend on the
   interpreter; NFC is split at the 690 runs where the reference's normalisation and Unicode 13's
   could differ, as Granite Embedding R2's tokenizer does. Fuzzed against `tokenizers` on 20,000
   strings and 5,000 id sequences for decoding, under Python 3.10 and 3.12: identical.
5. **The decoding loop is not `GenerationMixin`.** It runs the same operations in the same order for
   batch one with no padding; everything `generate` does that cannot change a number here (stopping
   criteria objects, streamers, padding bookkeeping) is absent.

## Runtimes

torch only. Published: one artifact per variant, `torch-bf16` (the checkpoint as stored); mozo's
`torch-fp32` is that file upcast at load (`DERIVES`), which is exact and the gated path.

**bf16 on CUDA is what `auto` picks there**, measured on an RTX 3060 12 GB with the example
photograph and 32 new tokens: `2b` 1.74 s against fp32's 6.71 s (4.7 GB against 10.2 GB), `4b`
2.77 s (9.5 GB; fp32 does not fit 12 GB). mozo and `transformers` take the same time on both
(6.71 s / 1.74 s for `2b`). On 300 VQAv2 questions, `2b` bf16 gave 295 of fp32's 300 answers
unchanged (VQA accuracy 76.9% against 77.3%). bf16 is not bit-identical to anything, and on the
CPU it is refused: 2.7x slower than fp32 there, and a different answer from the second token.

**ONNX and TensorRT, vision tower only, not published.** The tower exports (fixed grid, the
grid-dependent tables as inputs): ONNX Runtime matches torch fp32 to 3.7e-04 at 1,040 patches and
5.3e-03 at 9,600, and is 12-30x slower than torch on the GPU (it has only its CPU provider in
mozo's install). A TensorRT 11 fp32 engine at 1,040 patches is 1.22x torch fp32 (max |d|
1.9e-04); at 9,600 patches its execution context wants 12.15 GB and does not fit a 12 GB card. An
fp16 engine is 5.2x torch fp32 at 1,040 patches and 8.2x at 9,600 (466 ms against 3,820 ms),
against features that then differ by up to 0.97 and 3.1.
The tower is a fraction of a bf16 call's time (1.19 s to the first token for the whole prefill),
so neither is worth an artifact. The text decoder was not exported: one graph per decoding step
with a growing cache as inputs and outputs, three rotary rows and DeepStack outside it.

## Traps found while building it

- **The reference's version moves every logit** (4.57.3 vs 5.24.1, above). Pin it.
- **`logits_to_keep=1` is part of the arithmetic.** `generate` projects only the last position;
  projecting all 2,414 and taking the last row differs by 1.1e-05. On a short prompt the two agree,
  so the falsifier for it needs the full photograph.
- **Recomputing without the cache is not the cached step**: 2.1–3.4e-05 on the logits. The vendored
  cache grows by `torch.cat` from an empty tensor exactly as `DynamicLayer` does.
- **The processor has two backends that disagree**: torchvision (the default) and PIL differ by an
  8-bit level after any resize and by 5.9e-08 without one (the folded normalisation). And
  transformers 4.57.3's slow processor ignores the repository's pixel ceiling for Qwen2-VL's own: a
  different resolution (964 prompt tokens instead of 2,414) and a different answer from the first
  word.
- **Contraction folding is a union, not a per-letter table.** Probing `'` + letter + `a` finds the
  letters that complete *any* one-letter contraction (`s`, `t`, `m`, `d` and `ſ`, in either case);
  `'re` and `'ve` share their second letter, so their first-letter sets merge. The pattern carries
  exactly those sets, so it matches what the reference matches.
- **bfloat16 on a CPU is slower and different**: 2.7x fp32's time on the example, and a different
  answer from the second token. `torch-bf16` runs on CUDA only.

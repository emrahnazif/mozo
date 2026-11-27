# Qwen3-VL: ask about an image, get the answer in words

<!--
summary: Qwen3-VL in mozo: an image and a question in, a text answer out, token for token identical to transformers. 2B and 4B Instruct, Apache-2.0.
label: Qwen3-VL
keywords: mozo, qwen3-vl, vision language model, visual question answering, image captioning, vlm
-->

Qwen3-VL looks at a picture and answers in words: describe it, count what is in it, read the text
in it, say what is wrong with it. It is mozo's first model that writes, so its answer is text, a
token at a time, and it says whether it finished or was cut off at the token budget.

## Quick start

```python
import mozo

model = mozo.get_model("qwen3_vl/2b")
answer = model.predict("example.jpg", "How many people are at the table?")
answer.text             # the answer, in words
answer.finish_reason    # 'stop', or 'length' if it was cut off
```

Over HTTP the question is `text`, once:

```bash
curl -X POST "http://localhost:8000/predict/qwen3_vl/2b?text=What%20is%20on%20the%20table%3F" \
  -F "file=@example.jpg"
# {"text": "...", "finish_reason": "stop", "prompt_tokens": 2414, "generated_tokens": 23}
```

## Variants

| ID | Weights (bf16) | Runtimes | Licence |
|---|---|---|---|
| `qwen3_vl/2b` | 4.26 GB | torch (fp32 on CPU; fp32 or bf16 on CUDA) | Apache-2.0 |
| `qwen3_vl/4b` | 8.88 GB | torch (fp32 on CPU; fp32 or bf16 on CUDA) | Apache-2.0 |

Both are the Instruct models and share one vision tower; `4b` has a larger language model. The 8B
model, the Thinking variants and the mixture-of-experts sizes are not carried:
`mozo/vendors/qwen3_vl_deploy/PROVENANCE.md` says why.

## Inputs and outputs

```python
model.predict(image, prompt, *, max_new_tokens=128, temperature=0.0, top_p=1.0, top_k=0, seed=0,
              max_pixels=2_621_440)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image` | `str`, `bytes` or `numpy.ndarray` | | A file path, encoded bytes or an `HxWx3` RGB array. |
| `prompt` | `str` | | The question or instruction. It may not contain the model's control tokens (`<\|im_start\|>`, `<\|image_pad\|>` ...): the model would act on them rather than read them. |
| `max_new_tokens` | `int` | `128` | The most tokens to write. |
| `temperature` | `float` | `0.0` | `0` takes the most likely token every time: the same answer on every call. Above `0`, samples. |
| `top_p` | `float` | `1.0` | When sampling: keep the most likely tokens whose probabilities add up to this. |
| `top_k` | `int` | `0` | When sampling: keep only this many most likely tokens. `0` keeps all. |
| `seed` | `int` | `0` | When sampling: which sample. The same seed gives the same answer on the same device. |
| `max_pixels` | `int` | `2_621_440` | Resize the image to at most this many pixels first, between 65,536 and 16,777,216. |

Returns a `mozo.Generation`:

| Field | Description |
|---|---|
| `text` | The answer. |
| `finish_reason` | `"stop"` if the model ended it, `"length"` if it reached `max_new_tokens` and is cut off. |
| `prompt_tokens` | Tokens read; the image is most of them. |
| `generated_tokens` | Tokens written, the stopping token included. |

`str(answer)` is the text; `answer.to_dict()` is what `/predict` returns. Over HTTP every parameter
above is a query parameter of the same name, and the question is `text`.

## How it behaves

### Greedy by default: the same question gets the same answer

With `temperature=0` the model takes its most likely next token every time, so an answer is a
function of the image and the question. Qwen recommends sampling for this model
(`temperature=0.7, top_p=0.8, top_k=20`): a small model decoding greedily can repeat itself on long
answers. Sampling is reproducible: the same `seed` on the same device gives the same answer.

### The text is fluent, not checked

There is no confidence number. The model can be wrong with complete fluency, and miscount, misread
or invent detail. Ask for what can be verified, and check what matters.

### A cut-off answer says so

An answer that reaches `max_new_tokens` stops mid-sentence as often as not; `finish_reason` is
`"length"` then. Raise the budget, or ask for something shorter ("answer in one sentence").

### The image is the cost

The image is resized so both sides are multiples of 32 and the pixel count is at most `max_pixels`,
keeping its shape, and becomes one token per 32×32 pixels. The 1920×1281 example becomes 2,400
image tokens; a 640×427 photo, 260. Every one is read before the first word is written, so halving
`max_pixels` roughly halves the time to the first word. Fine print and small objects suffer first.

Upstream's own ceiling is 16,777,216 pixels, which lets a phone photograph through at up to 16,384
image tokens, more than a 24 GB GPU holds in fp32; mozo's default is 2,621,440 (2,560 tokens).

### Cost

The example photograph (2,400 image tokens), "Describe this image.", 32 new tokens, greedy:

| | `2b` | `4b` |
|---|---|---|
| RTX 3060 12 GB, `torch-bf16` (what `auto` picks on CUDA) | 1.74 s (1.19 s to the first token, 18 ms per token after) | 2.77 s (1.73 s, 33 ms) |
| RTX 3060 12 GB, `torch-fp32` | 6.71 s (5.43 s, 42 ms) | does not fit |
| CPU, 8 threads of a shared machine, `torch-fp32` | 42 s to the first token, 190 ms per token after | not measured |

On the GPU mozo takes the same time as `transformers` to within 1% (6.71 s and 1.74 s for `2b`, both
sides). GPU memory peaks at 4.7 GB for `2b` and 9.5 GB for `4b` in bf16, 10.2 GB for `2b` in fp32.

On 300 VQAv2 questions (COCO images, short answers), bf16 changed 5 of the 2B model's 300 answers
against fp32, and VQA accuracy went from 77.3% to 76.9%; for the 4B model, 4 of 300 answers changed
and accuracy went from 76.6% to 77.0%.

## Exactness

mozo's answer is the answer `transformers` 5.24.1 gives with `generate`, run the same way, for the
same image, question and settings: the same pixels and prompt tokens, every intermediate stage, and
the same logits at every decoding step, compared with `torch.equal`. On CUDA (an RTX 3060, torch
2.13, CUDA 12.6) that holds in bf16, the precision `auto` uses there, for both variants, and in fp32
for the 2B (the 4B's fp32 weights do not fit a 12 GB card); on the CPU, in fp32. It holds for
sampling too: with `transformers` seeded by `torch.manual_seed(seed)`, the sampled tokens are
identical. `tools/verify/qwen3_vl.py` is the check. Across devices and precisions the numbers
differ, and the words can: bf16 is not fp32, and CUDA is not the CPU.

Comparing against `transformers` yourself: pass the same pixel budget
(`processor(..., size={"longest_edge": 2621440, "shortest_edge": 65536})`), use the default image
processor (its PIL backend gives different pixels), and transformers 5 (4.57 gives different logits;
its slow image processor a different resolution altogether).

## Weights and licence

Code and weights are Apache-2.0: the license is stated in each model card's metadata, and the
authors' code repository `QwenLM/Qwen3-VL` is Apache-2.0. mozo publishes the checkpoints exactly as
Qwen stores them, in bfloat16; fp32 is that file upcast when it loads, which is exact.

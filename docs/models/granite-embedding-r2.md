# Granite Embedding R2: search text by meaning

<!--
summary: IBM's Granite Embedding R2 in mozo: English text in, one 384-d vector out, for searching documents by meaning. Apache-2.0.
label: Granite Embedding R2
keywords: mozo, granite embedding, text embeddings, semantic search, retrieval, rag
-->

Granite Embedding R2 turns English text into one 384-dimensional vector, and texts that mean the
same thing land close together. Use it to search documentation, notes, or what
[EasyOCR](/docs/models/easyocr.md) read off an image, by meaning rather than by keyword: "invoice
total" finds "amount due". It is mozo's only text-only family: it takes no image. The model is
IBM's `granite-embedding-small-english-r2`, 47.7M parameters.

## Quick start

```python
import mozo

model = mozo.get_model("granite_embedding_r2/small")
vectors = model.encode_text(["How do I count people crossing a line?",
                             "Track each person and record when they cross the line.",
                             "What licence are the YOLO weights under?"])

vectors.shape            # (3, 384), each of unit length
vectors[0] @ vectors[1]  # 0.904, the same question and its answer
vectors[0] @ vectors[2]  # 0.732, a different question
```

Over HTTP it is `/encode` and nothing else, with `mozo start` running:

```bash
curl -X POST "http://localhost:8000/encode/granite_embedding_r2/small?text=how%20do%20I%20count%20people%3F"
```

```json
{"model": "granite_embedding_r2/small", "revision": "2026-11-20", "dim": 384, "embeddings": [[0.05614, -0.00832, ...]]}
```

`/predict` answers `501` and says where to go instead, because the model answers nothing about an
image:

```json
{"detail": "granite_embedding_r2 returns vectors, not an answer about an image, so it has no /predict. Use /encode/granite_embedding_r2/small with ?text=."}
```

## Variants

| ID | Vector | Context | Weights (torch) | Runtimes | Licence |
|---|---|---|---|---|---|
| `granite_embedding_r2/small` | 384-d | 8,192 tokens | 191 MB | torch | Apache-2.0 |

## Inputs and outputs

```python
model.encode_text(text)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text` | `str` or `list[str]` | | One text or several, up to 8,192 tokens each. |

Returns a float32 array of shape `(N, 384)`, in the order given, always two-dimensional. The
vectors are L2-normalised, so a dot product is a cosine similarity. The model's `.revision` names
the weights that produced them.

`ValueError` if no text is given, if one is blank, if one is over 8,192 tokens, or if one contains
an unpaired surrogate.

Over HTTP, `/encode` takes `text`, repeated once per text (`?text=…&text=…`), and returns `model`,
`revision`, `dim` and `embeddings`.

## How it behaves

### Text only, and not CLIP's space

Granite was never trained alongside images. Its vectors mean nothing against
[CLIP](/docs/models/clip.md)'s or [SigLIP 2](/docs/models/siglip2.md)'s, so it cannot search
pictures by words: use those two for that. Granite is for text against text.

### You split documents into chunks, and too long is refused

The usual pipeline: split your documents into chunks, embed each chunk once and keep the vectors in
a vector database, then embed a question and let the database find the nearest chunks. Mozo is the
model in that pipeline and nothing else.

A text can be up to 8,192 tokens, `[CLS]` and `[SEP]` included. A longer one raises a `ValueError`
naming its length instead of being cut short, because a chunk shortened without telling you is a
different chunk:

```text
ValueError: text 0 of 1 is 9003 tokens, over the 8192-token context. Split it into chunks; mozo does not truncate.
```

Upstream truncates silently, or runs past the limit on positions the model never saw.

How big to make chunks is a question of quality more than cost. One vector summarises the whole
chunk: large chunks match broad questions to whole sections, small ones pin a precise passage. For
documentation, 1,024 to 2,048 tokens split at headings, so that a chunk is a section, is a sound
start. Then measure recall on a few dozen real questions of your own before you settle.

Per token, a chunk of a thousand or two costs about what a short one does; only the full 8,192 costs
several times more. The shape of that curve depends on the device, so measure where you deploy.
What has been measured, and on which machines, is in the vendor's `PROVENANCE.md`.

### A vector does not depend on what was encoded beside it

Texts are never padded, because a padded batch is not bit-identical to its rows run alone. On the
CPU, texts of exactly the same length share one forward pass, which was measured identical to
running each alone. On other devices each text runs alone. So the same text always gets the same
vector, whatever arrived with it.

One consequence: for many short texts, such as search queries, the CPU can be the faster device,
because only there do they share a forward pass. On an Apple M1 Max it was several times faster than
the GPU for a batch of short queries.

### The scores are cosine similarities

A dot product between two vectors is their cosine. It is not a probability and has no calibrated
zero: compare scores against each other, or against a threshold you calibrate on your own data.

### An index is tied to the weights that built it

A vector from one variant or revision means nothing against a vector from another. A stored index
is tied to both, and changing either means embedding the corpus again. `/encode` returns `model`
and `revision` with every response for that reason, and the model has `.revision`.

### The same text gets the same tokens on every Python

The tokenizer needs Unicode's answers (is this a letter, how does this character compose), and those
change between Python versions: Python 3.10 ships Unicode 13, Python 3.14 ships Unicode 16, and the
reference tokenizer uses neither. Left alone, a document's vector would depend on which Python
embedded it. So mozo's tokenizer carries the reference's own tables, and was checked identical to
the reference on every codepoint, runs of combining marks, random mixtures and every line of these
docs, under Python 3.10, 3.11, 3.12, 3.13 and 3.14.

### Parity

Bit-identical to `transformers`' `ModernBertModel` on the CPU at every stage: token ids, the
embeddings, all twelve layers, the final norm and the unit vector, compared with `torch.equal` and
no tolerance. It runs at the reference's speed.

Two traps that would pass a code review:

- The checkpoint is stored in bfloat16, and `transformers` 5 loads it that way by default. That is
  3e-03 away from float32, and the failure looks like the extraction's fault.
- The sliding-attention window is ±64 tokens, while the attention module carries the number 65.
  Only FlashAttention reads it; copying it into the mask changes every text longer than 65 tokens.

## Weights and licence

Code and weights are Apache-2.0, stated in the model card's metadata and downloadable without a
gated login. IBM trained the weights and publishes no inference code of its own; the extraction
derives from `transformers/models/modernbert` (© 2024 Answer.AI, LightOn, and contributors, and the
HuggingFace Inc. team). The pinned reference versions and what was taken are in
`mozo/vendors/granite_embedding_r2_deploy/PROVENANCE.md`.

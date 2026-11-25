# Granite Embedding R2: search text by meaning

<!--
summary: IBM's Granite Embedding R2 in mozo: English or multilingual text in, one 384-d vector out, for searching documents by meaning. Apache-2.0.
label: Granite Embedding R2
keywords: mozo, granite embedding, text embeddings, multilingual embeddings, semantic search, retrieval, rag
-->

Granite Embedding R2 turns text into one 384-dimensional vector, and texts that mean the same thing
land close together. Use it to search documentation, notes, or what
[EasyOCR](/docs/models/easyocr.md) read off an image, by meaning rather than by keyword: "invoice
total" finds "amount due". It is mozo's only text-only family: it takes no image. Two variants:
`small` is IBM's `granite-embedding-small-english-r2` (47.7M parameters) for English, and
`multilingual-small` is `granite-embedding-97m-multilingual-r2` (97M) for 200+ languages, where a
question in one language finds its answer in another.

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
{"model": "granite_embedding_r2/small", "revision": "2026-11-24", "dim": 384, "embeddings": [[0.05614, -0.00832, ...]]}
```

Across languages, with the multilingual variant (IBM's own example: an English, a German and a
Japanese question against passages in Japanese, English and German):

```python
model = mozo.get_model("granite_embedding_r2/multilingual-small")
questions = model.encode_text(["What is the tallest mountain in Japan?",
                               "Wer hat das Lied Achy Breaky Heart geschrieben?",
                               "ドイツの首都はどこですか？"])
passages = model.encode_text(["富士山は、静岡県と山梨県にまたがる活火山で、標高3776.12 mで日本最高峰の独立峰である。",
                              "Achy Breaky Heart is a country song written by Don Von Tress.",
                              "Berlin ist die Hauptstadt und ein Land der Bundesrepublik Deutschland."])
(questions @ passages.T).argmax(axis=1)  # array([0, 1, 2]): each question finds its passage
```

`/predict` answers `501` and says where to go instead, because the model answers nothing about an
image:

```json
{"detail": "granite_embedding_r2 returns vectors, not an answer about an image, so it has no /predict. Use /encode/granite_embedding_r2/small with ?text=."}
```

## Variants

| ID | Languages | Vector | Context | Weights (torch) | Runtimes | Licence |
|---|---|---|---|---|---|---|
| `granite_embedding_r2/small` | English | 384-d | 8,192 tokens | 95 MB | torch, coreml | Apache-2.0 |
| `granite_embedding_r2/multilingual-small` | 200+ (52 with retrieval training), and code | 384-d | 32,768 tokens | 195 MB | torch | Apache-2.0 |

The two variants are different spaces, despite the same size: a vector from one means nothing
against a vector from the other. IBM's larger multilingual model, `granite-embedding-311m-multilingual-r2`,
is not carried: its tokenizer is subject to the Gemma Terms of Use, not Apache-2.0. Nor are the R1
models (`granite-embedding-278m-multilingual` and `-107m-multilingual`), which IBM says the R2
multilingual models replace.

## Inputs and outputs

```python
model.encode_text(text)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text` | `str` or `list[str]` | | One text or several, up to the variant's context each: 8,192 tokens (`small`) or 32,768 (`multilingual-small`). |

Returns a float32 array of shape `(N, 384)`, in the order given, always two-dimensional. The
vectors are L2-normalised, so a dot product is a cosine similarity. The model's `.revision` names
the weights that produced them.

`ValueError` if no text is given, if one is blank, if one is over the context, or if one contains
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

A text can be up to 8,192 tokens with `small` and 32,768 with `multilingual-small`, the special
tokens at either end included. A longer one raises a `ValueError`
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

Every text runs alone: texts are never padded and never batched. A padded batch is not
bit-identical to its rows run alone, and on a multi-threaded x86 CPU not even a batch of texts of
exactly equal length is (up to 1.6e-03 on the raw vector, measured with both variants). So the
same text always gets the same vector, whatever arrived with it. Earlier versions batched
equal-length texts on the CPU, which was measured identical on an Apple M1 Max and on one thread,
and not on a many-core x86 machine.

### On Apple silicon `small` runs as a CoreML graph

With `pip install 'mozo[coreml]'`, `auto` runs `small` through CoreML on a Mac (macOS 15 or later).
CoreML only reaches the GPU with a fixed input size, so the graph holds the model at eight fixed
lengths, from 64 to 8,192 tokens, and each text is padded to the shortest one that fits. The padding
is masked: no real token attends to it, so the vector is the one torch gives, within 1.8e-07. An
index built on either runtime works with queries from the other.

Measured on an Apple M1 Max, against torch on the same GPU:

| Text length | CoreML | torch (MPS) |
|---|---|---|
| 64 tokens | 5.1 ms | 7.6 ms |
| 512 tokens | 8.9 ms | 15.1 ms |
| 2,048 tokens | 47.7 ms | 99.3 ms |
| 8,192 tokens | 394 ms | 1,471 ms |

A text that only just spills into the next length, such as 2,049 tokens, runs at 4,096 and costs
1.0 to 1.3 times torch's time. Each length loads the first time a text needs it, in about a
second, so a server that only sees short queries never loads the long ones. Pass
`runtime="torch-fp32"` to run on torch instead.

### `multilingual-small` runs on torch only

ONNX Runtime and TensorRT were measured for it on an RTX 3090 and are not published. TensorRT is
4.6 times faster than torch on a 16-token query (1.3 ms against 6.1 ms in fp32, 0.95 ms in fp16)
and Belebele retrieval in 20 languages does not move (nDCG@10 77.14 in fp32, 77.16 in fp16). But
no engine could hold the 32,768-token context on a 24 GB card, fp32 engines are slower than torch
from about 4,000 tokens, and fp16's distance from fp32 grows with the text's length. On the CPU,
ONNX Runtime is faster only below about 100 tokens. The numbers are in the vendor's
`PROVENANCE.md`, under *Export*.

### The scores are cosine similarities

A dot product between two vectors is their cosine. It is not a probability and has no calibrated
zero: compare scores against each other, or against a threshold you calibrate on your own data.

### An index is tied to the weights that built it

A vector from one variant or revision means nothing against a vector from another. A stored index
is tied to both, and changing either means embedding the corpus again. `/encode` returns `model`
and `revision` with every response for that reason, and the model has `.revision`.

One exception: revision `2026-11-24` is `2026-11-20`'s weights stored as IBM ships them, in
bfloat16, and runs in float32 just the same. Its vectors are bit-identical, so an index built on
either revision works with the other; only the download halved.

### The same text gets the same tokens on every Python

The tokenizers need Unicode's answers (is this a letter, upper or lower case, how does this
character compose), and those change between Python versions: Python 3.10 ships Unicode 13, Python
3.14 ships Unicode 16, and the reference tokenizer uses neither. Left alone, a document's vector
would depend on which Python embedded it. So mozo's tokenizers carry the reference's own tables.
The English one was checked identical to the reference on every codepoint, runs of combining marks,
random mixtures and every line of these docs, under Python 3.10, 3.11, 3.12, 3.13 and 3.14; the
multilingual one on every codepoint, random mixtures, and every passage, question and answer of
Belebele in its 122 languages.

The multilingual tokenizer does not normalise: `e` followed by a combining accent and the
precomposed `é` are different tokens, as they are upstream. Nor does mozo strip leading and
trailing whitespace, which `sentence-transformers` does before tokenizing: a text with a trailing
newline gets a slightly different vector from mozo than from `SentenceTransformer.encode`. Strip it
yourself if your index was built with `sentence-transformers`.

### Parity

Both variants are bit-identical to `transformers`' `ModernBertModel` on the CPU at every stage:
token ids, the embeddings, all twelve layers, the final norm and the unit vector, compared with
`torch.equal` and no tolerance, up to the full context. It runs at the reference's speed.

Two traps that would pass a code review:

- The checkpoint is stored in bfloat16, and `transformers` 5 loads it that way by default. That is
  3e-03 away from float32, and the failure looks like the extraction's fault.
- The sliding-attention window is ±64 tokens, while the attention module carries the number 65.
  Only FlashAttention reads it; copying it into the mask changes every text longer than 65 tokens.

## Weights and licence

Code and weights are Apache-2.0, stated in each model card's metadata and downloadable without a
gated login. IBM trained the weights and publishes no inference code of its own; the extraction
derives from `transformers/models/modernbert` (© 2024 Answer.AI, LightOn, and contributors, and the
HuggingFace Inc. team). The pinned reference versions and what was taken are in
`mozo/vendors/granite_embedding_r2_deploy/PROVENANCE.md`.

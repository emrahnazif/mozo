# Provenance

This package is a deployment-only extraction of **Granite Embedding R2**'s inference path.

It derives from `transformers/models/modernbert` (Apache-2.0, © 2024 Answer.AI, LightOn, and
contributors, and the HuggingFace Inc. team). IBM trained the weights and publishes no inference
code of its own: the checkpoints are `ModernBertModel` state dicts, the model card loads them
through `sentence-transformers` (which is that model plus taking the `[CLS]` position and, for
retrieval, normalising it), and so the `transformers` implementation is the one whose numbers the
published checkpoints reproduce.

**The weights are Apache-2.0**, stated in the model card's metadata on the checkpoint itself and
served ungated (`gated: false` on the Hub API, checked 2026-11-20).

## What mozo guarantees, exactly

**mozo's vector equals `F.normalize(ModernBertModel(ids).last_hidden_state[:, 0])`, bit for bit,
on the CPU in float32 with SDPA attention.** `tools/verify/granite_embedding_r2.py` proves it on
every run, with `torch.equal` and no tolerance: token ids, the embeddings, each of the twelve
layers, the final norm (whose first row is the raw `[CLS]` vector) and the normalised vector, on
fixtures either side of the attention window, a documentation page and a text at the full 8,192
tokens; and the token ids alone on every line of the documentation and 20,000 random mixtures of
Unicode's awkward corners. `--fuzz` adds every codepoint.

**What the weights owe IBM's training run is not something mozo can measure**, and it is not
claimed. IBM's benchmark numbers are self-reported; mozo claims the link it can check.

**MPS and CUDA are outside the exact guarantee**, as for every family: MPS differs from the CPU by
1.8e-07 on the unit vectors. Nothing here is device-specific; the arithmetic is simply ordered
differently there.

## The reference is pinned, and its version does not move the answer

Parity was established against **torch 2.11.0, transformers 5.8.0**. The checkpoint was saved by
transformers 4.51.3, and 5.x rewrote this model — its masks now come from `masking_utils`, its
rotary embedding from `modeling_rope_utils` — so both were run on the same 19 fixtures when the
family was built: token ids identical, and **all 323 stage comparisons `torch.equal`**. There is
one reference, not two.

## What was taken

| from | what |
|---|---|
| `modeling_modernbert.py` | the embeddings, the rotary embedding, the attention, the gated MLP, the encoder layer, the model's forward |
| `masking_utils.py` | the bidirectional sliding-window mask, rewritten in `layers.sliding_mask` |
| `integrations/sdpa_attention.py` | the call to `scaled_dot_product_attention` |
| `configuration_modernbert.py` and the variant's `config.json` | the geometry, written out as a frozen dataclass in `config.py` |
| `tokenizer.json` at commit `2ab6fa8e…` | the vocabulary, merges, added tokens and template, re-encoded into `assets/granite_bpe.json.gz` |
| the `tokenizers` library, probed | the character classes and NFC boundaries in the same asset (see *Divergences* 3) |

`assets/granite_bpe.json.gz` is 429,293 bytes, sha256
`7fb1f214fdb8a9491fbced271c6b5d956ed521082576371442ba8ce911f5e479`, rebuilt by
`tools/fetch/granite_embedding_r2.py --derive-vocab` under Python 3.10 with `tokenizers` 0.22.2.

The checkpoint is consumed as IBM publishes it, modulo one repack: `tools/fetch` verifies
`model.safetensors` against the Hub's sha256 and writes the same tensors as an ordinary state dict,
still in bfloat16 (95 MB). The encoder upcasts them to float32 when loading, which is exact, and
runs in float32. No tensor is renamed or dropped. Revision `2026-11-20` stored the same tensors
already cast to float32 (191 MB); its vectors and `2026-11-24`'s are bit-identical.

**The tokenizer is ModernBERT's.** IBM's model card calls it newly trained; measured, its
vocabulary, merges, added tokens, pre-tokenizer and post-processor all equal
`answerdotai/ModernBERT-base`'s `tokenizer.json` (the merges differ only in JSON form, pairs
against strings).

## What was deliberately left behind

**`granite-embedding-english-r2`** (149M parameters, 768-d, 22 layers). The same architecture, so
one `Spec` and a checkpoint with no new code, and it exercises no path `small` does not. Absent
because `small` took 99% of the downloads across the two (6.66M a month against 54K, November
2026), not because anything here cannot run it.

**The multilingual R2 models** (`granite-embedding-97m-multilingual-r2`, `-311m-…`). Not examined:
their tokenizer, context and architecture may all differ.

**The R1 models.** XLM-RoBERTa, a different architecture.

**The task heads** — masked language model, sequence and token classification, question answering,
multiple choice. The published repository is the bare encoder.

**FlashAttention-2 and its unpadding.** A different order of the same arithmetic, outside any exact
guarantee. The SDPA path is the one the gate can hold.

**Silent truncation.** `sentence-transformers` truncates at `max_seq_length`; raw `transformers`
warns and runs past 8,192 tokens on rotary positions the model never saw. Not carried: a chunk
shortened without saying so is a different chunk. `Tokenizer.__call__` raises with the count and
the cap.

**Inert config fields.** `hidden_act: "gelu"` and `position_embedding_type: "relative_key_query"`
are BERT's names, never read by ModernBERT, which uses `hidden_activation` and rotary positions.

## Where this diverges from the reference, and why

### 1. Vectors leave L2-normalised

Upstream returns the `[CLS]` hidden state as it is, and the model card says so ("the model
produces unnormalized vectors"). mozo normalises, with `F.normalize` — what
`sentence-transformers` applies under `normalize_embeddings=True`, the setting the card uses for
retrieval — so that a dot product between any two vectors is a cosine, the contract every family
that embeds keeps. The gate compares the raw vector too, so this is an addition, not a change.

### 2. Nothing pads, and only equal lengths batch, and only on the CPU

A padded batch is not bit-identical to its rows run alone (1.1e-07, measured), which would make a
stored vector depend on what else happened to be in its request. So rows are never padded. Texts of
exactly equal length share a forward pass on the CPU, where that was measured identical at every
length and batch size tried (16 to 1,500 tokens, batches of 2 to 64); on MPS it was not (1.8e-07
at some sizes) and CUDA is unmeasured, so there every text runs alone.

### 3. The tokenizer carries its Unicode tables

The reference tokenizer asks Unicode three questions — is this a letter, a digit, whitespace; how
does this compose — and answers from tables compiled into `tokenizers`, which are neither Python's
nor the `regex` package's. Measured against it:

- `regex` 2026.4 calls **4,657** codepoints letters or digits that the reference treats as
  punctuation — characters assigned in recent Unicode versions.
- Python 3.14 (Unicode 16) decomposes **20** characters that the reference leaves alone.
- Even Python 3.10 (Unicode 13) composes across **58** marks that the reference will not: marks
  Unicode 10 to 13 gave a nonzero combining class (Sogdian, Dogra, Dives Akuru, Gondi and others)
  that the reference treats as class 0. Its decompositions and compositions are Unicode 13's; its
  combining classes are older.

Any of these would make a document's vector depend on which Python wrote it, and an index written
from two machines would disagree with itself. So the derivation probes the reference for every
codepoint and the asset carries the answers: the split pattern is built from the reference's own
letter, digit and whitespace classes, and NFC is applied as the reference applies it — Unicode 13
NFC, split at the codepoints Unicode 13 had not assigned and at the 58 marks, which behave as walls
(class 0, no decomposition, part of no composition, all checked by the derivation). Unicode's
normalisation stability policy then guarantees each run between walls normalises identically on
every later Python. The derivation refuses to write an asset if the reference does anything this
cannot express.

Verified when the family was built: **every text identical to the reference on Python 3.10, 3.11,
3.12, 3.13 and 3.14** — every codepoint in contexts that cross each class boundary, 20,000 runs of
mixed combining marks, 20,000 random mixtures, and every line of the documentation, about 1.15
million texts. Disabling the boundaries failed 4,556 of them; replacing the carried classes with
`regex`'s failed 477. Rerun it with `tools/verify/granite_embedding_r2.py --fuzz --corpus PATH`,
then `tools/verify/granite_embedding_r2_corpus.py PATH` under each Python — that one needs only
the standard library.

### 4. The sliding mask is always built

Upstream passes no mask to a sliding layer below 64 tokens, which is the same as one that allows
everything. mozo always builds it — measured bit-identical at every length from 2 to 8,192 — so
there is no branch on length for an exported graph to freeze.

## Traps

Each of these runs, produces plausible output, and is wrong. Each is covered by the gate, and the
ones with a number were confirmed by perturbing them and watching the gate fail where they reach.

- **The reference runs in bfloat16 unless told otherwise.** The config's `torch_dtype` is
  bfloat16 and transformers 5 loads in it by default. 3.0e-03 on the unit vectors.
- **The window is ±64, and the +1 is a decoy.** `local_attention: 128` is the whole window; the mask
  allows `abs(q - k) <= 64`. `ModernBertAttention` carries `64 + 1`, read only by FlashAttention.
  Copying 65 is exact below 66 tokens and 7.7 out at 66.
- **Two rotary bases, chosen by layer.** Full-attention layers (0, 3, 6, 9) turn at 80,000, sliding
  ones at 10,000; the frequencies come from an *integer* `arange`. A base of 10,001 fails from
  layer 1 onward and nowhere before.
- **Layer 0 has no attention norm.** An `nn.Identity` upstream, so no weight in the checkpoint; the
  strict load catches the extra one.
- **No biases**, in any linear layer or LayerNorm.
- **The MLP is gated, with exact GELU.** `Wo(gelu(value) * gate)`, value first. tanh GELU is
  1.1e-02 out.
- **LayerNorm eps is 1e-5** here — torch's default, but stated. 1e-6 is 1.7e-03 out.
- **Runs of spaces are added tokens**, matched before byte-pair encoding, longest first; the word
  after one starts without a space of its own.
- **`[MASK]` strips whitespace to its left**, by Rust's `char::is_whitespace`.
- **Special tokens written out become those tokens.** A document that mentions `[SEP]` carries a
  separator. Reproduced, not escaped.
- **The template's ids are read from `tokenizer.json`**, not assumed.
- **Python's `re` scans a large character class range by range once it reaches past U+FFFF**, at
  every position, even in ASCII text. The astral ranges sit behind a lookahead: 16× faster on the
  boundary scan, identical on every codepoint.

## Speed

The one record of what this costs; `tools/bench/granite_embedding_r2.py` produces it. Measured on
an Apple M1 Max with torch 2.11.0, against `transformers` on the same weights, synchronised, each
side warmed and the two interleaved:

| tokens | CPU, 8 threads | per token | vs reference | MPS | per token | vs reference |
|---|---|---|---|---|---|---|
| 32 | 15.3 ms | 478 µs | 0.94× | 8.3 ms | 258 µs | 0.98× |
| 512 | 71.9 ms | 141 µs | 0.98× | 15.9 ms | 31 µs | 0.93× |
| 1,024 | 130 ms | 127 µs | 0.95× | 36.7 ms | 36 µs | 0.96× |
| 2,048 | 291 ms | 142 µs | 0.99× | 103 ms | 50 µs | 0.97× |
| 8,192 | 2.92 s | 356 µs | 1.01× | 1.50 s | 183 µs | 0.99× |

What it says:

- **mozo runs at the reference's speed**: 0.94–1.01× its time on the CPU, 0.93–0.99× on MPS. Same
  layers, same SDPA call.
- **Per-token cost is not flat, and its shape differs by device.** On the CPU, 1,024 and 2,048
  tokens cost about what 512 does per token; on MPS, 2,048 costs 1.6× what 512 does. The full
  8,192 costs 2.5× (CPU) to 5.9× (MPS) the per-token price of 512.
- **For many short texts the CPU wins.** Sixty-four short queries in one call: 86 ms on the CPU,
  where equal-length texts share a forward pass, and 585 ms on MPS, where each runs alone.
- **The tokenizer is not the bottleneck**: 5.6 ms cold for 2,048 tokens of documentation — about
  2% of the CPU encode — and 1.6 ms once its word cache is warm (`transformers`' tokenizer, also
  warm: 2.2 ms).

NVIDIA is unmeasured.

## Export

CoreML is published; ONNX is built and measured, and not. `tools/export/granite_embedding_r2.py`
produces both from the vendored encoder rather than the reference. Its ONNX run writes nothing into
`weights/`; `--coreml` writes `coreml-fp32.zip`, and only once every length agrees with torch.

### CoreML

**Fixed lengths, because CoreML reaches the GPU only with a fixed shape.** Measured on an Apple M1
Max, coremltools 9.0: a `RangeDim` input segfaults inside CoreML on load, at any upper bound, and
`EnumeratedShapes` loads but runs on the CPU alone, 3–29× slower than torch on MPS. So the package
holds `graph.Bucketed` at 64, 128, 256 … 8,192 tokens, one function each (`tokens_<N>`), sharing
one copy of the weights: 192 MB unzipped, 90 MB zipped. macOS 15 is the minimum, for multifunction
packages.

**Padded and masked.** A text is padded at the end to the shortest length that fits, and every
layer hides the padded keys, so no real token sees them. Against the network on the unpadded text,
in torch: 1.8e-07 or less, and zero at a full length.

**Positions are derived from the ids.** Built with `torch.arange` they are trace-time constants,
and the converter folded every mask into the package: 290 MB, and 40 s to load, every load. A
precompiled `.mlmodelc` still took 10.7 s. Derived as `cumsum(ones_like(ids)) - 1`, nothing folds:
192 MB, and about 1 s to load the first length, 0.6 s each after (the package compiles once). An additive float mask folded anyway (728 MB for one length), so the
mask stays bool.

**fp32.** An fp16 package was a little faster on the GPU (37 against 46 ms at 2,048 tokens) and
moved vectors by up to 1e-3. In a mixed index — fp16 documents, fp32 queries — the top result
changed for 1–1.5% of queries, so an index would depend on the runtime. The Neural Engine executes
only fp16, and won only at 128 tokens.

Against torch on MPS, synchronised, interleaved, at the shortest text that needs each length and
the longest that fits it:

| Length | Text | max\|Δ\| | CoreML | torch MPS | ratio |
|---|---|---|---|---|---|
| 64 | 33 / 64 | ≤8.2e-08 | 5.1 / 5.1 ms | 8.3 / 7.6 ms | 0.61 / 0.68× |
| 128 | 65 / 128 | ≤7.6e-08 | 6.3 / 4.7 ms | 8.3 / 6.7 ms | 0.76 / 0.71× |
| 256 | 129 / 256 | ≤1.8e-07 | 5.4 / 5.4 ms | 7.1 / 8.9 ms | 0.75 / 0.61× |
| 512 | 257 / 512 | ≤1.2e-07 | 8.7 / 8.9 ms | 10.0 / 15.1 ms | 0.87 / 0.59× |
| 1,024 | 513 / 1,024 | ≤8.4e-08 | 18.2 / 18.4 ms | 16.8 / 35.2 ms | 1.08 / 0.52× |
| 2,048 | 1,025 / 2,048 | ≤1.8e-07 | 46.8 / 47.7 ms | 38.7 / 99.3 ms | 1.21 / 0.48× |
| 4,096 | 2,049 / 4,096 | ≤8.8e-08 | 146 / 147 ms | 109 / 362 ms | 1.34 / 0.41× |
| 8,192 | 4,097 / 8,192 | ≤1.8e-07 | 395 / 394 ms | 392 / 1,471 ms | 1.01 / 0.27× |

### ONNX

**It exports cleanly**: the TorchScript exporter at opset 17, 191 MB in float32, with batch and
sequence length dynamic. Both masks are built inside the graph from the length — the reason the
vendor never skips the sliding mask (*Divergences* 4).

**Parity**: max|Δ| on the unit vectors between 1.2e-07 and 3.0e-07 at each of ten lengths from 2
to 8,192 tokens, against the vendored torch encoder. That is the size of eager against SDPA inside
torch: arithmetic order, not a defect. Not bit-exact, as no graph runtime in mozo is.

**Speed** on an Apple M1 Max, CPU, 8 threads, onnxruntime 1.21.1, as ONNX Runtime's time over
torch's — each side warmed, the two interleaved over three trials, ONNX Runtime's thread spinning
off:

| tokens | 32 | 64 | 128 | 256 | 512 | 1,024 | 2,048 | 8,192 |
|---|---|---|---|---|---|---|---|---|
| ONNX Runtime / torch | 0.59× | 0.83× | 0.93× | 1.11× | 1.41× | 2.00× | 3.04× | 3.34× |

Ratios rather than times, and still only roughly: across runs on the same machine, with other work
on it, the 512-token ratio landed anywhere from 1.07× to 1.41×. The shape held every time.

ONNX Runtime wins below about 200 tokens, where torch's per-call overhead dominates, and loses more
the longer the input: opset 17 writes attention as MatMul, Softmax, MatMul and materialises every
score, where torch calls a fused kernel. ONNX Runtime's transformer optimizer, asked to fuse as for
BERT, fuses GELU and LayerNorm and no attention — it does not recognise ModernBERT's rotary
attention — and that changes nothing measurable.

**Why it is not published.** For indexing documents it is slower; for short queries it is faster,
but mozo picks one runtime per load, not per input. The case for publishing it is CUDA, and as the
graph TensorRT would build an engine from — and that needs a measurement on an NVIDIA machine:
`tools/export/granite_embedding_r2.py --device cuda`. A TensorRT engine cannot be published at all:
it is built for one GPU architecture and one TensorRT version, so it would be built on the machine
that runs it, which is runtime work mozo does not have yet for any family.

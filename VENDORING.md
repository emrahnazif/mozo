# Vendoring a model, and planning one

A working document for whoever — person or agent — is adding the next family to mozo. It is
distilled from the plans for CLIP, EasyOCR, Grounding DINO and SigLIP 2, and from what each of
those got wrong before it got it right.

Every model is different. Some are one graph, some are five. Some export to ONNX cleanly, some
cannot be exported at all, and some export but run slower than torch. Some have an authors'
PyTorch implementation to check against and some do not. What does *not* change is the method, and
that is what this document is.

---

## 0. What vendoring means here

A vendored family is a **deployment-only extraction of one model's inference path**, living at
`mozo/vendors/<family>_deploy/`, that reproduces the published implementation **bit for bit** and
depends on nothing but torch, numpy, opencv and Pillow.

It is not a wrapper. Wrapping a library means inheriting its dependency tree, its import-time side
effects, its version churn and its training code. Extracting means the code that runs *is* the
implementation, which is why the extraction can be checked exactly and why it costs nothing at run
time — EasyOCR runs within 1% of the published package, SigLIP 2 within 4% of `transformers` on
CPU and 11% faster on Metal.

Three rules hold everywhere and are never bent:

1. **A vendored package may not import another vendored package, and may not import `mozo`.**
   Duplication between vendors is deliberate. Three families each carry their own copy of CLIP's
   byte-pair vocabulary; that is correct.
2. **Each family gets its own vendored package *and* its own adapter.** No shared base classes
   between families.
3. **The gate is exact.** `torch.equal`, no tolerance, because a tolerance hides precisely the
   drift a gate exists to catch.

---

## 1. Decide five things before writing a plan

Answer these first. Every later decision follows from them, and getting one wrong invalidates the
plan rather than complicating it.

### 1.1 What is the reference?

**The reference is the implementation whose numbers the published checkpoints reproduce.** Not the
most famous implementation, not the most convenient one.

| situation | reference | example |
|---|---|---|
| Authors publish PyTorch | the authors' repository | CLIP (`openai/CLIP`), Grounding DINO (`IDEA-Research`) |
| Authors publish JAX/Flax, someone else ports to PyTorch | the **PyTorch port**, because that is what the PyTorch checkpoints reproduce | OWLv2, SigLIP 2 (`transformers`, not `big_vision`) |
| Authors publish only an ONNX file | that file | some InsightFace models |

Write down *why*. `owlv2_deploy/PROVENANCE.md` states the rule in one sentence and three later
families cite it.

**Then find out what the reference itself claims.** This is the step everyone skips. SigLIP 2's
first plan asserted the PyTorch checkpoints matched the JAX originals to 1e-3, citing a conversion
script — which turned out to convert two *different* checkpoints that mozo does not ship. For the
fifteen it does ship, upstream states no agreement at any tolerance. That is a stronger and more
honest thing to write down, and PROVENANCE now says exactly that.

So: **claim only the link you can measure, and say where the chain stops.**

### 1.2 What is the licence — on the code *and* on the weights, separately?

They are different works and frequently different licences. Record **where the claim is made**, not
just what it says.

- **Stated on the checkpoint itself** — the strong case. SigLIP 2: `license: apache-2.0` on every
  model card, ungated.
- **Stated on the repository that serves the file, but not in the project** — Grounding DINO's
  GitHub has no licence section at all; the claim lives on the HF mirrors the README links to. Cite
  both places, because a reader who checks GitHub alone finds a code licence and no mention of a
  checkpoint.
- **Inferred from silence** — CLIP publishes one MIT `LICENSE` and no separate weights terms.
  Publishable, and the NOTICE must say it is an inference *as* an inference.
- **A chain through a redistributor** — EasyOCR ships CRAFT weights that CLOVA released without
  terms; the chain is JaidedAI's redistribution under its own Apache-2.0. One link weaker than the
  others, and PROVENANCE says so in those words.

Watch for a licence that looks alarming and isn't, and check rather than assume: SigLIP 2's
tokenizer is Gemma's, whose own weights are gated under the Gemma terms — but Google publishes that
same vocabulary ungated under Apache-2.0 inside every SigLIP 2 repository. That is an affirmative
grant on the artifact actually taken.

Watch also for the opposite: research-only face weights, NVIDIA non-commercial SegFormer
checkpoints, AGPL YOLO weights, CC-BY-NC Depth Anything B/L. Check each checkpoint, not the family.

### 1.3 Which variants, and on what evidence?

Not "all of them" and not "the one in the README". Use download counts and say where you counted:

- SigLIP 2: 15 fixed-resolution of 17 repos, naflex excluded and *named* as the largest omission
  (20% of usage).
- EasyOCR: 5 of 17 recognisers = 88% of real installs, counted from GitHub release
  `download_count` against 26.7M total pulls.
- Grounding DINO: upstream publishes exactly two.
- CLIP: four ViTs first; five ResNets need a second image tower, so they land behind the same gate
  later.

**Publish enough variants to exercise every distinct code path, not every variant.** SigLIP 2
carries five of fifteen deliberately. They are the most-used, and between them they still exercise
every distinct geometry: a patch grid that floors, head dimensions of 64, 72 and 96, asymmetric
towers, and a sharded download. The other ten exercise nothing new.

Always write down **what is deliberately left out and why**, so a later reader sees a decision
rather than an oversight.

### 1.4 What comes back?

Pick the PixelFlow container and state the one design decision the task forces.

- Boxes → `Detections`. Classification → `Classifications` via `pf.from_scores(...).top_k(...)`.
  OCR → `Detections` with `text=` and `segments=`.
- **Never write a private sort, a private rounding or a private container.** Route through
  PixelFlow's own entry points; `tests/conftest.py::as_pixelflow_*` exists so the gate compares
  both sides through the same door.
- Say plainly what the confidence number *is*. A cosine similarity, a sigmoid probability and a
  softmax are all legitimate and mean different things. CLIP's is a raw cosine and the docs say
  compare it against other scores, not against an intuition about percentages. SigLIP 2's is a
  per-pair probability with a meaningful zero, and the docs say it is still not P(class | image).

One design decision usually dominates. For Grounding DINO it was reporting the *prompt index* rather
than upstream's decoded token span, because a span can return `"yellow school"` for
`"yellow school bus"`. That is a divergence in a reported string, not in a number, and it goes in
PROVENANCE under "where this diverges, and why".

### 1.5 What can actually run, and where?

Answer before promising a runtime.

- **torch is the floor.** Every family ships `torch-fp32`. `EXECUTES = ("torch",)` unless a graph
  artifact is genuinely published.
- **ONNX/CoreML are measured, never assumed.** See §6.
- **Some models cannot export at all**, and the reason belongs in the plan: EasyOCR's recogniser
  uses `AdaptiveAvgPool2d((None, 1))` under a dynamic width, which cannot trace.
- **Device changes numbers.** CLIP's archives are mixed precision and `clip.load` floats them
  *only on CPU* — so "bit-exact" is undefined until the device is pinned. EasyOCR's mps path is not
  bit-identical to its cpu path (58 of 275 comparisons differ; strings exact, confidence moves
  2.2e-05). Gate on CPU and say so.

---

## 2. Phase 0 — ground truth

**Do this before writing a line of the extraction.** It is the phase that decides whether
everything after it is checkable, and it is the phase most often rushed.

1. **Pin the upstream commit or version**, and record it. `openai/CLIP` is not on PyPI and is still
   maintained, so pinning is real work. A frozen upstream (EasyOCR, last release Sept 2024) is a
   feature: the parity target cannot move under the gate.

2. **Pin the whole reference environment, not just the model.** SigLIP 2's `SiglipImageProcessor`
   was refactored from PIL to a torchvision backend, which *changed the pixels it produces*. The
   gate records `torch`, `torchvision` and `transformers` versions and prints them on every run.

3. **Stand it up in a throwaway environment** so the working tree stays clean.

4. **Disable anything that changes the arithmetic.** Grounding DINO's reference must run with the
   CUDA extension *off*, so it takes the same `grid_sample` path the extraction will; comparing
   against the CUDA kernel measures a kernel difference and calls it drift. Pin the attention
   implementation on both sides — `transformers` defaults to sdpa, and eager and sdpa are the same
   arithmetic in a different order.

5. **Record every intermediate, not just the final answer.** This is the highest-leverage thing in
   this document. Two implementations can agree on the last tensor and disagree in the middle, and
   when they do, the stage name is the difference between "something moved" and knowing which of
   eleven rewritten pieces moved it.

   Grounding DINO records: preprocessed tensor, token ids, BERT `last_hidden_state`, projected
   text, all three Swin levels, encoder memory, fused text, each of six decoder layers,
   `pred_logits`, `pred_boxes`.

6. **Build a fixture set that includes the awkward cases**, not one clean photograph: an image with
   nothing in it, rotated text, a prompt matching nothing, a prompt at the token cap, mixed case,
   punctuation, non-Latin scripts.

7. **Confirm determinism**: same input twice, byte-identical output.

---

## 3. The plan document

Write it before implementing. These sections, in this order. Sections marked *(after)* get filled
in as the work lands.

| section | contents |
|---|---|
| **Header** | One paragraph: what the model does and what is genuinely new about it here. |
| **Where it fits** | Table: vendor path, adapter, `task_type`, endpoint, result shape, artifacts, licence, `EXECUTES`. Say explicitly which shared files change and which do not. |
| **The reference, and what mozo guarantees** | §1.1. State the guarantee exactly and where the chain stops. |
| **Provenance** | Upstream, commit, code licence, weights licence with citations, what the checkpoint actually contains. |
| **Variants to publish** | Table with evidence. What is left out and why. |
| **The result shape** | The container, the one design decision, what the number means. |
| **Still open** | Decisions not yet made. Being explicit here is what stops them being made by accident. |
| **Tasks / phases** | Phase 0 first, then bottom-up, gating each piece as it lands. |
| **Traps** | Predicted from reading the source. See §5. |
| **Size estimate** | Lines vendored, plus the non-vendor deliverables — which are usually as large. |
| **Results** *(after)* | §8. |

### Phase ordering that works

Phase 0 ground truth → the pieces bottom-up, each gated alone before assembly → preprocessing →
assembly and the seam → checkpoint → mozo integration → gate and tests → bench → docs and counts.

Gate each piece the moment it exists. CRAFT was gated alone (`max|delta| = 0.0` at four input
shapes) before the pipeline was assembled. SigLIP 2 gated pixels, then the vision tower, then the
tokenizer, then the text tower, then scoring — so every failure had one candidate cause.

---

## 4. The gate

`tools/verify/<family>.py`. This is the deliverable that makes the extraction worth anything.

**Three paths, compared exactly:** the vendored package, the upstream reference, and mozo itself
(registry → weights → adapter → PixelFlow).

**Every stage, not the final answer.** See §2.5.

**No tolerance.** `torch.equal`. Real divergences that any sane tolerance would have swallowed:
QuickGELU vs `nn.GELU`; operator precedence moving CLIP's logits by 1.9e-06; a `LayerNorm` epsilon
of 1e-6 vs torch's 1e-5; rescaling by 1/255 before normalising instead of folding it into the
statistics, 5.9e-08.

**Pin what makes the comparison defined:** device, attention implementation, and — for a reference
loaded with `from_pretrained` — re-allocate its parameters before comparing. `from_pretrained`
places tensors at the checkpoint's own offsets and BLAS picks different vectorised paths for
page-aligned storage; OWLv2 measured 1.2e-07 on that alone. Clone first, so what is compared is
arithmetic rather than addresses.

**The gate must be falsifiable, and you must actually falsify it.** Perturb a constant and confirm
it fails *at the right stage*.

- Grounding DINO: `pe_temperature_h` 20 → 21 fails 40 of 65 comparisons and leaves the backbone,
  BERT and projected-text rows passing — exactly where that constant does and does not reach.
- SigLIP 2: eight perturbations, all caught, each at its own stage — LayerNorm eps (5.1e-04 on
  image features), antialias off (9.4e-01 on pixels), GELU approximation (5.5e-03), MAP-head
  residual misread (7.1e-01), pooling the wrong slot (1.0e+01), attention scale ×1.0000001
  (2.4e-06).

A gate that has never failed has not been shown to work.

**Fuzz anything you wrote from scratch.** SigLIP 2's tokenizer was checked on 2,815 phrases —
mixed case, CJK, Arabic, emoji, combining characters, ligatures, reserved names, and thousands of
random Unicode strings. That found a bug no reading would have: `str.lower()` and the reference's
Rust normaliser disagree about 95 codepoints, because Python 3.10 ships Unicode 13 and the
reference ships later. Left alone, a stored vector would have depended on which interpreter wrote
it.

---

## 5. The trap catalogue

A trap is code that **runs, produces plausible output, and is wrong**. Predict them by reading the
source before implementing; append the ones you find by measuring afterwards. Every family here
found traps it had not predicted, and the predicted ones were mostly real.

### Resize and preprocessing

- **PIL's bicubic is not OpenCV's and not `F.interpolate`'s.** PIL's filter support scales with the
  downsampling factor. Neither `cv2.resize` nor `F.interpolate(antialias=True)` reproduces it.
- **`Resize(224)` scales the short side; `Resize((224, 224))` squashes.** Different pictures.
- **Antialiasing is a filter change, not a rounding change.** SigLIP 2: 9.4e-01 on a tensor whose
  whole range is about two.
- **Resizing uint8 vs float differs**, because torchvision rounds and clamps back to eight bits
  before the normalise. 3.9e-03.
- **The rescale may be folded into the statistics.** `(x - 127.5) / 127.5` is not
  `(x / 255 - 0.5) / 0.5`. 5.9e-08.
- **A resize can be applied twice.** EasyOCR resizes each crop with cv2 and then again with PIL
  BICUBIC. Collapsing them is the obvious optimisation and it changes pixels.
- **A named constant may not be what it says.** EasyOCR passes `Image.Resampling.LANCZOS` (= 1) to
  `cv2.resize`, where 1 is `INTER_LINEAR`. It is bilinear. Reproduce it with a comment, or someone
  "fixes" it later.
- **Aspect-preserving resizes can exceed their own cap.** Grounding DINO's 800/1333 returns
  167×1336 on a 500×4000 image, because the short side rounds first. Clamping looks like a fix and
  moves every box.
- **Padding may replicate rather than zero-fill.** EasyOCR copies the last column.
- **Normalisation constants are usually not ImageNet's.** CLIP has its own; SigLIP 2 is 0.5/0.5.
- **Upstream's preprocessing may depend on the device, and on CUDA `x / 255` is `x * (1/255)`.**
  torch's CUDA division by a scalar multiplies by the reciprocal, so the same division on the host
  and on the GPU differ in the last bit. Mayaku divides on the host for the CPU and sends the uint8
  canvas to the GPU otherwise: a vendor that kept the host path was exact on the CPU gate and wrong
  at the input on the GPU one. Copy the device dispatch, and keep the result contiguous: a
  channels-last input is `torch.equal` to the contiguous one and runs other conv kernels.
- **So may the decode.** Mayaku decodes on the GPU when its maps are on CUDA (`host_side`); a vendor
  that decoded the same GPU maps on the CPU differed from upstream on 55 of 64 runs (boxes, scores
  or masks in the last bits).

### Tokenisation

- **Case handling is often undocumented or wrong in the config.** SigLIP 2's repos declare
  `do_lower_case: true` on a tokenizer class that ignores it, so `AutoTokenizer` preserves case
  while training lowercased — worth a factor of 350 on a capitalised prompt.
- **Python's case tables belong to the interpreter.** Carry the mapping if bit-exactness matters.
- **Normalisation order can disable a later step.** SigLIP 2 replaces spaces with `▁` *before* a
  `Split(" ")` pre-tokenizer, so the split never fires and merges cross word boundaries.
- **Prefix spaces, or their absence.** `'a'` (235250) is not `'▁a'` (476).
- **Byte-fallback tables are neither complete nor contiguous.** `<0x00>` at 217, `<0x09>` absent.
- **Reserved names in a prompt become tokens.** A caption containing `<eos>` carries a control
  token. Longest match wins: `<unused1>` prefixes `<unused10>`.
- **Padding may be attended and pooling may read it.** SigLIP 2 pools the last slot, which is
  coherent only because every row is padded to a fixed 64. Adding the "obviously correct" attention
  mask changes every vector.
- **Pooling by `argmax` over token ids, not by position** (CLIP).
- **Silent truncation.** Upstream usually truncates; mozo raises, naming the count and the cap.
- **`str.lower()` is not the reference's lowercase even codepoint by codepoint.** OmDet-Turbo's
  `CLIPTokenizer` lowercases in Rust, one codepoint at a time: Python applies the Greek final-sigma
  rule (`"ΑΣ".lower()` is `"ας"`, the reference gives `"ασ"`), and 55 single codepoints differ
  between Python 3.12's tables and the reference's. Fuzzing 20,034 prompts found 17 wrong; a
  1,488-entry table derived from `tokenizers` fixed all of them.
- **Lowercasing and separator rules are part of the contract.** Grounding DINO lowercases, appends
  a trailing `.` if absent, and treats `?` as a separator — all of which move the phrase map.

### Numerics

- **The activation.** QuickGELU (`x * sigmoid(1.702x)`), `gelu_pytorch_tanh`, and exact erf GELU are
  three different functions and none of them raises.
- **`LayerNorm` epsilon.** 1e-6 is common; torch defaults to 1e-5.
- **Subclassed `LayerNorm`** that upcasts to fp32 and casts back (CLIP).
- **Operator precedence.** `logit_scale * image @ text.t()` scales the *features*, not the product.
  1.9e-06.
- **`need_weights` and the attention implementation.** `nn.MultiheadAttention` with
  `need_weights=True` takes torch's unfused branch, where the query is scaled *before* the matmul —
  and never reaches `scaled_dot_product_attention`, so an `attn_implementation` pin does not govern
  it. eager vs sdpa cost 1.5e-06 on Grounding DINO's BERT.
- **A transpose is not a second matmul.** Follow upstream's order.
- **Batched and unbatched forwards are not bit-identical.** 2.5e-07 for OWLv2; **1.4e-05** for
  EasyOCR's recogniser, enough to flip a character. Batch exactly as upstream does — which may mean
  *not* batching.

### Checkpoint and geometry

- **Load strict.** Upstream often uses `strict=False`, which lets a key you named differently stay
  silently at its initialisation. Enumerate what is genuinely absent and drop it by prefix.
- **A strict load cannot catch aliasing.** Grounding DINO's encoder box head is a `deepcopy` of the
  decoder's, not the same object; aliasing them keeps both key sets matching while one overwrites
  the other. Cost 12.9 on the reference boxes, silently.
- **Geometry that looks derivable usually is not.** SigLIP 2's head dimension is 64, 72 *or* 96;
  its MLP is not `4 × width` for five of the fifteen; `resolution % patch == 0` is false for
  one variant, where the convolution strides past six pixels. Write the geometry down as frozen
  dataclasses and let the strict load hold it against the checkpoint.
- **Some keys must be dropped by name.** CLIP's `build_model` deletes `input_resolution`,
  `context_length`, `vocab_size` before its own strict load.
- **Weights that are present and unused.** Name them in PROVENANCE rather than leaving a tensor
  nobody can account for.
- **`load_state_dict` checks names and shapes, never dtypes.** DEIM's `reg_scale` is an int64
  parameter in every checkpoint (the YAML says `4`); a module built with `4.0` loads it silently as
  float and computes in a different dtype. Check dtypes yourself after the strict load.
- **`mmap=True` makes every tensor a view into one whole-file mapping.** Returning a 4-byte scalar
  from it keeps 1.5–7.5 GB resident for the life of the process. `.clone()` small reads.

### Postprocessing

- **Half-resolution heatmaps.** EasyOCR's coordinates are ×2 before the resize ratio is undone.
- **Which of two sizes multiplies which coordinate.** Boxes normalised to the *resized* image and
  scaled by the *original* is the classic failure and it does not raise.
- **Thresholds decide which crops exist**, not just which results survive.
- **Confidence formulas are not means.** EasyOCR's is `prod(p) ** (2 / sqrt(n))`, and it is also
  what the low-contrast second pass compares — so a wrong formula changes *strings*.
- **A leading blank in a CTC charset.** Index 0 is `[blank]` and the published charset string does
  not include it.
- **Second passes are part of the model.** EasyOCR re-runs low-contrast crops and keeps the better
  read.
- **The decode runs where upstream's runs, and PixelFlow's rounding hides it when it doesn't.**
  DEIM's upstream decodes (sigmoid, top-k) on the device its network ran on; mozo's adapter copied
  the raw outputs to the host first. CUDA's `sigmoid` is not the CPU's in the last bit (about a
  quarter of the 24,000 logits per image, up to 6e-08), so the 300 scores differed on every image
  and on one the top-k order changed too -- yet the gate's mozo check, which compared PixelFlow's
  rows (two or three decimals), passed every one. Compare the arrays the adapter hands to
  `pf.detections.from_arrays` exactly (spy on it), not only the rows it gives back.

- **`torchvision.ops.batched_nms` separates classes by `max + 1`.** For up to 1,000 boxes on the
  CPU it offsets each class by the largest coordinate plus one, which stops separating once boxes
  reach negative coordinates (a letterbox edge): two classes at -1000..10 suppress each other.
  `tests/test_vendor_agreement.py` catches the same bug in a hand-written NMS; a vendor whose
  upstream calls torchvision inherits it and must keep it to stay exact (Mayaku street).
- **Stale `.pyc` on the CIFS share.** An edit to `mozo/__init__.py` kept serving the old
  docstring until `__pycache__` was deleted. When a test reads text you just changed and disagrees,
  clear the cache before believing it.

### Things that are upstream's behaviour, not bugs to fix

Reproduce them, and record them:

- Grounding DINO's `previous_col` is not reset per batch row; its padding mask is commented out, so
  padded tokens are attended — and that is what the published weights were trained with.
- A prompt containing a literal `<|endoftext|>` pools there (CLIP). Upstream behaves the same way.

### Environment traps

- **An environment variable can change the reference's arithmetic.** timm's Swin window attention
  takes the explicit `q @ k.T` path unless `TIMM_FUSED_ATTN=2`, when it calls
  `scaled_dot_product_attention` -- different bits. Remove it in the gate before timm is imported.
- **transformers 5 renames published keys inside `from_pretrained`.** OmDet-Turbo's checkpoint
  says `language_backbone.model.text_model.*`; transformers 5's `CLIPTextModel` has no
  `text_model` level. A gate that loads the published file directly has to rename the same way.
- **transformers 5's default image processor is a different picture.** `AutoProcessor` returns
  the torchvision backend, which resizes unlike PIL and unlike the authors' pipeline; pin
  `backend="pil"` when PIL is what the weights were trained and evaluated with.
- **An upstream text cache can make the reference history-dependent.** OmDet-Turbo's model caches
  each class prompt across calls and encodes only the new ones, so the same prompt's embedding
  depends on which batch it was first encoded in. Empty the caches before every reference call.

- **Import-time side effects.** EasyOCR sets `os.environ["LRU_CACHE_CAPACITY"] = "1"` — a
  process-wide torch allocator setting that is not a library's to write. Do not carry it.
- **A default that silently does nothing.** EasyOCR's `quantize=True` swallows `NoQEngine` in a bare
  `except`; on a machine with fbgemm the gate would compare two different models.
- **Two entry points that disagree.** EasyOCR's array path documents BGR and its path entry decodes
  greyscale with libjpeg directly — 27 and 7 grey levels apart. Hand both sides the same array.
- **Mutable module globals** (`imgH`) become spec fields when vendored.
- **A config loader can leak between configs in one process.** DEIM's `YAMLConfig` merges into a
  module-level registry, so building `nano` then `large` in one process gives `large` nano's
  `use_lab` and decoder width. Build each reference config in a fresh process, and read the
  resolved values to write the spec rather than the YAML files.
- **CPU torch is not reproducible from one process to the next.** MKL's sgemm picks its kernel by
  buffer alignment, which changes per process: the same DEIM model on the same pixels gave a
  different encoder attention output in about one process in ten, on an idle machine (logits up
  to ~5e-4 apart). An exact gate across two processes needs `MKL_CBWR=AVX2,STRICT` set before
  torch loads, on both sides; it makes the result alignment-independent without favouring either.
  A gate that passes most runs and fails a few, always at the same matmul, is this.

---

## 6. Runtimes and export

Default to torch only. A graph artifact is published when it is measured to be worth it, and *not
publishing it is a result worth writing down*.

Ask, in order:

1. **Does it export at all?** Chase the blocker to its cause rather than quoting the error.
   EasyOCR's CRNN fails on `AdaptiveAvgPool2d((None, 1))` under a dynamic width; the only
   substitution that traces is a mean over the same axis, which is *not* the same in float (the
   pool divides by three, mean multiplies by a reciprocal) — implemented, gated, confidences moved
   1e-06, reverted.
2. **Is it faster?** CRAFT exports at parity 1.8e-07 and runs **0.66× torch** — 185 ms becomes
   280 ms. An artifact nothing would select is not worth publishing.
3. **Does it hold parity at the shapes you actually run?** CoreML was the fastest runtime measured
   for EasyOCR (13.8 ms vs torch-mps's 18.8) and is still not publishable: parity 1.6e-05, above
   the 1.4e-05 already known to flip a character, and only at a *fixed* shape. With flexible shapes
   neither model converts. Enumerating would be 6,241 combinations.
4. **How much of the call is even in the graph?** If postprocessing is OpenCV either way, a graph
   buys less than it would for a single-forward detector.

Record the answer in the plan's Results and in PROVENANCE, with numbers. "Not exported" with a
measured reason is a finding; without one it is a gap.

---

## 7. Integration checklist

The extraction is usually under half the work. These are the pieces that are easy to forget, in
roughly the order they bite.

- [ ] `mozo/registry.py` — one entry: `adapter_class`, `module`, `task_type`, description,
      variants. Add `ENCODES` if the family embeds. Add the task to `PROMPTED` if it takes text.
- [ ] `mozo/adapters/<family>.py` — its own class, `VARIANTS`, `EXECUTES`, `self.revision`.
      Accept `threshold` if the task type is prompted: the server forwards it, so omitting the
      parameter turns a documented query param into a 500.
- [ ] `tools/fetch/<family>.py` — download with a hash the *publisher* records (read from an API,
      not transcribed), place `LICENSE` and `NOTICE`, use `tools/common.py` (`digest`,
      `download_verified`, `require_licence`, `variant_parser`). Handle sharded downloads if the
      checkpoint has them.
- [ ] `tools/verify/<family>.py` — §4.
- [ ] `tools/bench/<family>.py` — §8.
- [ ] `tests/families/test_<family>.py` — registry agrees with the adapter; every variant ships its
      terms; geometry self-consistency; the refusals; the result shape.
- [ ] `tools/generate_manifest.py` — regenerate.
- [ ] **`pyproject.toml` `package-data`** — now a single `"*"` rule covering `LICENSE`, `NOTICE`,
      `PROVENANCE.md` and `assets/*`, so this needs no edit. It used to be per-family and had gone
      stale for five of fourteen families, shipping vendored code with none of its terms and, in
      SigLIP 2's case, without the tokenizer vocabulary it reads at construction. **If you add a
      data file in a new shape, check the wheel actually contains it** — build one and look.
- [ ] `PROVENANCE.md`, `NOTICE`, `LICENSE`, `README.md` inside the vendored package.
- [ ] `docs/models/<family>.md` page, in the shape the other family pages share, and a row in
      `docs/models/all-models.md`.
- [ ] Counts in `README.md` (six sites) and `mozo/__init__.py`, plus the licence breakdown.
      `tests/test_stated_counts.py` fails until they are right.
- [ ] `mozo/static/test_ui.html` — check whether it needs anything. It reads `prompted` and
      `encodes` from `/models`, so usually not; CLIP had to *make* that true.

A new task type additionally needs a branch in `mozo/server.py`'s dispatch. An existing one needs
nothing — Grounding DINO touched no shared code because OWLv2 had already named
`open_vocabulary_detection`.

---

## 8. Results — what to write down when it lands

Append to the plan. This is what makes the next family cheaper.

**Parity.** How many comparisons, against what, at which stages, and that the gate is falsifiable
with the perturbation you used.

**Traps found by measuring.** Separately from the predicted ones. EasyOCR found six it had not
predicted; SigLIP 2 found two. These are the most valuable paragraphs in any of these documents.

**Latency.** Against the reference, same weights, same device, same image — so the number means
"this extraction" and not "this model". Report the breakdown so a regression can be placed.

Two measurement traps worth inheriting:

- **Synchronise the device.** Metal queues asynchronously, so an unsynchronised timer measures
  submission. SigLIP 2 initially reported itself 23% slower on MPS purely because its own path
  ended in `.cpu()` and the reference's did not. Synchronised, it is 11% faster.
- **Interleave the two sides** across several trials, so thermal drift moves both together.

**Export.** §6, with numbers, including for what you did *not* publish.

**Size.** Lines vendored against lines upstream; published bytes per variant.

**Published.** Revision, artifacts, manifest counts, and whether the weights are actually uploaded.

---

## 9. The rules, in one place

1. A vendored package imports no other vendor and never `mozo`.
2. Each family has its own vendored package and its own adapter. No sharing.
3. The gate is exact, covers every stage, and has been made to fail on purpose.
4. Pin the reference: commit, versions, device, attention implementation.
5. Geometry is written down, never inferred. The strict load is what checks it.
6. Load strict. Never `strict=False`.
7. Reproduce upstream's behaviour, including its bugs. Record each one as a divergence *from the
   obvious reading*, not as a fix.
8. Refuse rather than silently truncate or silently drop.
9. Claim only the link you can measure, and say where the chain stops.
10. Weights licence is separate from code licence, and the claim gets a citation.
11. Say what the confidence number is, and what it is not.
12. Name what you left out, so it reads as a decision.

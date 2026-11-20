# granite_embedding_r2_deploy

Deployment-only Granite Embedding R2. Apache-2.0 code *and* weights, extracted from
`transformers/models/modernbert` and reduced to the encoder, with no `transformers`,
`tokenizers`, `sentence-transformers` or `safetensors` dependency at inference.

English text in, one vector out, and texts that mean the same thing land close together. Text
only: there is no image tower, so these vectors do not share a space with CLIP's or SigLIP 2's.

```python
from mozo.vendors.granite_embedding_r2_deploy import Encoder, SPECS

encoder = Encoder("torch-fp32.pth", SPECS["small"])          # device="cpu"

encoder.encode_text(["How do I count people crossing a line?",
                     "Track each person and record when they cross."])
# (2, 384) float32, L2-normalised: a dot product is the cosine
```

One variant, `small` — 47.7M parameters, 384-d vectors, up to 8,192 tokens a text. See
`PROVENANCE.md` for the larger `granite-embedding-english-r2` and why it is absent.

## What the vector is

The `[CLS]` position's final hidden state, L2-normalised: the published pooling, normalised the
way IBM's model card does for retrieval. The number two vectors give is their cosine — compare it
against other cosines, or a threshold you calibrate on your own data. It is not a probability.

## What it guarantees

- **Bit-identical to `transformers`' `ModernBertModel`** on the CPU, at every stage, with
  `torch.equal` and no tolerance. `tools/verify/granite_embedding_r2.py` is the gate.
- **The same text gets the same vector, whatever it was encoded beside.** Nothing pads; only texts
  of exactly equal length share a forward pass, and only on the CPU, where that was measured
  identical.
- **The same text gets the same tokens on every Python from 3.10 to 3.14.** The Unicode tables the
  tokenizer needs are carried, not borrowed from the interpreter.
- **Too long is refused, never truncated.** Chunking is the caller's.

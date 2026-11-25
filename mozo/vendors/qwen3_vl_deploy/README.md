# qwen3_vl_deploy

Deployment-only Qwen3-VL. Apache-2.0 code *and* weights, extracted from
`transformers/models/qwen3_vl` and reduced to one image, one prompt and the decoding loop, with no
`transformers`, `tokenizers`, `jinja2` or `safetensors` dependency at inference.

An image and a question in, an answer in words out, one token at a time.

```python
from mozo.vendors.qwen3_vl_deploy import Generator, Sampling, SPECS

model = Generator("torch-bf16.pth", SPECS["2b"])            # device="cpu", float32
answer = model.generate(rgb, "What is on the table?", max_new_tokens=64)
answer.text, answer.finish_reason                           # ('...', 'stop')

model.generate(rgb, "Write a caption.", 32, Sampling(temperature=0.7, top_p=0.8, top_k=20, seed=3))
```

Two variants, `2b` and `4b` (the Instruct models), sharing one vision tower.

## What it guarantees

- **Bit-identical to `transformers` 5.24.1's `generate`** on the CPU in float32: every stage, every
  step's logits, every token, greedy and sampled with the same seed. `tools/verify/qwen3_vl.py` is
  the gate.
- **The same text gets the same tokens on every Python from 3.10 up.** The Unicode tables the
  tokenizer needs are carried, not borrowed from the interpreter.
- **A prompt holding a control token is refused**, never interpreted.

See `PROVENANCE.md` for what was taken, what was left, and where this diverges.

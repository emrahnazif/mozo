#!/usr/bin/env python3
"""Check Granite Embedding R2's tokenizer against a recorded corpus, under whatever Python runs this.

The tokenizer is meant to give the reference's ids on every Python from 3.10 up, though each
ships different Unicode tables -- that is why the vendored package carries its own. The reference
(``tokenizers``) is not needed here, nor torch: ``granite_embedding_r2.py --fuzz --corpus PATH``
records the corpus and the reference's ids once, and this replays it with the standard library
alone, importing the tokenizer module by its path.

    python tools/verify/granite_embedding_r2.py --fuzz --corpus corpus.json
    python3.14 tools/verify/granite_embedding_r2_corpus.py corpus.json
"""

from __future__ import annotations

import importlib.util
import json
import sys
import unicodedata
from pathlib import Path

TOKENIZER = (Path(__file__).resolve().parent.parent.parent
             / "mozo" / "vendors" / "granite_embedding_r2_deploy" / "tokenizer.py")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    spec = importlib.util.spec_from_file_location("granite_tokenizer", TOKENIZER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tokenizer = module.Tokenizer()

    corpus = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    failures = [text for text, ids in zip(corpus["texts"], corpus["ids"])
                if tokenizer.encode(text) != ids]
    print(f"Python {sys.version.split()[0]}, Unicode {unicodedata.unidata_version}: "
          f"{len(corpus['texts']):,} texts, {len(failures)} differ from the reference")
    for text in failures[:10]:
        print(f"  {text[:60]!r}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Encode texts in a process that never imports FAISS.

PyTorch and FAISS each ship an OpenMP runtime. Loading both in one process
aborts on macOS. The parent keeps FAISS; this worker keeps the embedder.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["CODESEARCH_EMBED_WORKER"] = "1"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 3:
        print(
            "usage: python -m codesearch.embed_worker MODEL TEXTS.json VECTORS.npy",
            file=sys.stderr,
        )
        return 2
    model_name, texts_path, out_path = args
    from codesearch.embedder import Embedder
    import numpy as np

    texts = json.loads(Path(texts_path).read_text(encoding="utf-8"))
    if not isinstance(texts, list):
        raise ValueError("worker input must be a JSON list of strings")
    embedder = Embedder(model_name=model_name, cache_dir=None)
    vectors = embedder._embed_raw(texts, show_progress=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, vectors)
    revision = embedder.model_revision or ""
    print(f"MODEL_REVISION={revision}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

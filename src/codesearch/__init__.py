"""Semantic code search: meaning-based function retrieval with a measured ANN tradeoff."""

from __future__ import annotations

import os

# faiss-cpu and PyTorch each ship an OpenMP runtime; on macOS the process
# aborts unless one of them is allowed to continue.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "0.1.0"

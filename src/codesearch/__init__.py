"""Semantic code search: meaning-based function retrieval with a measured ANN tradeoff."""

__version__ = "0.1.0"

from codesearch.runtime import preload_single_openmp

# Load one OpenMP runtime before any extension module that ships its own copy.
# Embedding still runs in a subprocess so PyTorch never shares the FAISS process.
preload_single_openmp()

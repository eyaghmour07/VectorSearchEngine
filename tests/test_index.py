from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import faiss

from codesearch.embedder import l2_normalize
from codesearch.index import FaissFlatIndex, FaissHNSWIndex


def _distinct_vectors(n: int, dim: int) -> np.ndarray:
    # One-hot-ish rows so nearest neighbors are unambiguous.
    raw = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        raw[i, i] = 1.0
        raw[i, (i + 1) % dim] = 0.25 * (i + 1)
    return l2_normalize(raw)


def test_flat_returns_exact_nearest_neighbors() -> None:
    vectors = _distinct_vectors(8, 16)
    index = FaissFlatIndex(16)
    index.build(vectors)

    query = vectors[3]
    hits = index.search(query, k=3)
    assert hits[0][0] == 3
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)

    sims = vectors @ query
    expected = [int(i) for i in np.argsort(-sims)[:3]]
    assert [idx for idx, _ in hits] == expected


def test_flat_save_load_round_trip(tmp_path: Path) -> None:
    vectors = _distinct_vectors(6, 8)
    index = FaissFlatIndex(8)
    index.build(vectors)
    original = index.search(vectors[1], k=4)

    path = tmp_path / "index.faiss"
    index.save(path)

    restored = FaissFlatIndex(8)
    restored.load(path)
    assert restored.ntotal == 6
    assert restored.search(vectors[1], k=4) == original


def test_flat_rejects_dim_mismatch() -> None:
    index = FaissFlatIndex(4)
    with pytest.raises(ValueError, match="Expected dim"):
        index.build(np.zeros((3, 8), dtype=np.float32))


def test_hnsw_uses_inner_product_and_finds_the_query_vector() -> None:
    vectors = _distinct_vectors(12, 16)
    index = FaissHNSWIndex(16, ef_search=64)
    index.build(vectors)
    assert index._index.metric_type == faiss.METRIC_INNER_PRODUCT
    hits = index.search(vectors[4], k=1, ef_search=64)
    assert hits[0][0] == 4
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)

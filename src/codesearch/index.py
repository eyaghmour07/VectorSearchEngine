from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import faiss
import numpy as np

DEFAULT_HNSW_M = 32
DEFAULT_EF_CONSTRUCTION = 200
DEFAULT_EF_SEARCH = 64


class VectorIndex(ABC):
    @abstractmethod
    def build(self, vectors: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        k: int,
        *,
        ef_search: int | None = None,
    ) -> list[tuple[int, float]]:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str | Path) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def ntotal(self) -> int:
        raise NotImplementedError


class FaissFlatIndex(VectorIndex):
    """Exact inner-product search over L2-normalized vectors."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)

    def build(self, vectors: np.ndarray) -> None:
        if vectors.ndim != 2:
            raise ValueError(f"Expected 2D vectors, got shape {vectors.shape}")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {vectors.shape[1]}")
        if vectors.dtype != np.float32:
            vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        else:
            vectors = np.ascontiguousarray(vectors)
        self._index = faiss.IndexFlatIP(self.dim)
        if vectors.shape[0]:
            self._index.add(vectors)

    def search(
        self,
        query_vector: np.ndarray,
        k: int,
        *,
        ef_search: int | None = None,
    ) -> list[tuple[int, float]]:
        return _search_faiss(self._index, query_vector, k)

    def save(self, path: str | Path) -> None:
        faiss.write_index(self._index, str(path))

    def load(self, path: str | Path) -> None:
        index = faiss.read_index(str(path))
        if not isinstance(index, faiss.IndexFlatIP):
            raise TypeError(f"Expected IndexFlatIP at {path}, got {type(index).__name__}")
        if index.d != self.dim:
            raise ValueError(f"Index dim {index.d} does not match expected {self.dim}")
        self._index = index

    @property
    def ntotal(self) -> int:
        return int(self._index.ntotal)


class FaissHNSWIndex(VectorIndex):
    """Approximate inner-product search. Built only after the flat baseline is trusted."""

    def __init__(
        self,
        dim: int,
        m: int = DEFAULT_HNSW_M,
        ef_construction: int = DEFAULT_EF_CONSTRUCTION,
        ef_search: int = DEFAULT_EF_SEARCH,
    ) -> None:
        self.dim = dim
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._index = self._new_index()

    def _new_index(self) -> faiss.IndexHNSWFlat:
        index = faiss.IndexHNSWFlat(self.dim, self.m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.ef_construction
        index.hnsw.efSearch = self.ef_search
        return index

    def build(self, vectors: np.ndarray) -> None:
        if vectors.ndim != 2:
            raise ValueError(f"Expected 2D vectors, got shape {vectors.shape}")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {vectors.shape[1]}")
        if vectors.dtype != np.float32:
            vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        else:
            vectors = np.ascontiguousarray(vectors)
        self._index = self._new_index()
        if vectors.shape[0]:
            self._index.add(vectors)

    def search(
        self,
        query_vector: np.ndarray,
        k: int,
        *,
        ef_search: int | None = None,
    ) -> list[tuple[int, float]]:
        if ef_search is not None:
            self._index.hnsw.efSearch = int(ef_search)
        else:
            self._index.hnsw.efSearch = self.ef_search
        return _search_faiss(self._index, query_vector, k)

    def save(self, path: str | Path) -> None:
        faiss.write_index(self._index, str(path))

    def load(self, path: str | Path) -> None:
        index = faiss.read_index(str(path))
        if not isinstance(index, faiss.IndexHNSWFlat):
            raise TypeError(f"Expected IndexHNSWFlat at {path}, got {type(index).__name__}")
        if index.d != self.dim:
            raise ValueError(f"Index dim {index.d} does not match expected {self.dim}")
        if index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError(
                f"HNSW index at {path} uses metric {index.metric_type}, expected METRIC_INNER_PRODUCT"
            )
        index.hnsw.efSearch = self.ef_search
        self._index = index

    @property
    def ntotal(self) -> int:
        return int(self._index.ntotal)


def create_index(index_type: str, dim: int, ef_search: int = DEFAULT_EF_SEARCH) -> VectorIndex:
    if index_type == "flat":
        return FaissFlatIndex(dim)
    if index_type == "hnsw":
        return FaissHNSWIndex(dim, ef_search=ef_search)
    raise ValueError(f"Unknown index type {index_type!r}. Expected 'flat' or 'hnsw'.")


def _search_faiss(index, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
    if index.ntotal == 0:
        return []
    query = np.asarray(query_vector, dtype=np.float32)
    if query.ndim == 1:
        query = query.reshape(1, -1)
    if query.shape[1] != index.d:
        raise ValueError(f"Query dim {query.shape[1]} does not match index dim {index.d}")
    k = min(k, index.ntotal)
    scores, indices = index.search(np.ascontiguousarray(query), k)
    results: list[tuple[int, float]] = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < 0:
            continue
        results.append((int(idx), float(score)))
    return results

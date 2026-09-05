"""A from-scratch HNSW over L2-normalized vectors, inner-product metric.

Same M / efConstruction / efSearch defaults as the FAISS index so a
head-to-head measurement is meaningful. Python and row-at-a-time graphs
will lose to FAISS on latency; that gap is the point of the comparison.
"""

from __future__ import annotations

from heapq import heappop, heappush
from pathlib import Path

import numpy as np

from codesearch.index import DEFAULT_EF_CONSTRUCTION, DEFAULT_EF_SEARCH, DEFAULT_HNSW_M, VectorIndex


class NativeHNSWIndex(VectorIndex):
    def __init__(
        self,
        dim: int,
        m: int = DEFAULT_HNSW_M,
        ef_construction: int = DEFAULT_EF_CONSTRUCTION,
        ef_search: int = DEFAULT_EF_SEARCH,
    ) -> None:
        if ef_search < 1:
            raise ValueError("ef_search must be >= 1")
        if m < 2:
            raise ValueError("m must be >= 2")
        self.dim = dim
        self.m = m
        self.m_max = m
        self.m_max0 = m * 2
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.level_mult = 1.0 / np.log(m)
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._graph: list[list[list[int]]] = []
        self._enter = 0
        self._max_level = -1
        self._rng = np.random.default_rng(0)

    def build(self, vectors: np.ndarray) -> None:
        if vectors.ndim != 2:
            raise ValueError(f"Expected 2D vectors, got shape {vectors.shape}")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {vectors.shape[1]}")
        self._vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        n = int(self._vectors.shape[0])
        self._graph = [[] for _ in range(n)]
        self._enter = 0
        self._max_level = -1
        self._rng = np.random.default_rng(0)
        for node in range(n):
            self._insert_node(node)

    def _insert_node(self, node: int) -> None:
        level = int(self._rng.exponential(self.level_mult))
        self._graph[node] = [[] for _ in range(level + 1)]
        if node == 0:
            self._enter = 0
            self._max_level = level
            return

        enter = self._enter
        query = self._vectors[node]
        for lc in range(self._max_level, level, -1):
            enter = self._greedy(query, enter, lc)

        for lc in range(min(level, self._max_level), -1, -1):
            candidates = self._search_layer(query, [enter], self.ef_construction, lc)
            neighbors = self._select(query, candidates, self.m)
            self._graph[node][lc] = neighbors
            max_deg = self.m_max0 if lc == 0 else self.m_max
            for neighbor in neighbors:
                links = self._graph[neighbor]
                while len(links) <= lc:
                    links.append([])
                links[lc].append(node)
                if len(links[lc]) > max_deg:
                    links[lc] = self._select(self._vectors[neighbor], links[lc], max_deg)
            enter = neighbors[0] if neighbors else enter

        if level > self._max_level:
            self._max_level = level
            self._enter = node

    def _greedy(self, query: np.ndarray, enter: int, level: int) -> int:
        current = enter
        best = self._score(current, query)
        changed = True
        while changed:
            changed = False
            for neighbor, sim in self._neighbor_scores(current, level, query):
                if sim > best:
                    best = sim
                    current = neighbor
                    changed = True
        return current

    def _search_layer(
        self, query: np.ndarray, enters: list[int], ef: int, level: int
    ) -> list[int]:
        visited = set(enters)
        candidates: list[tuple[float, int]] = []
        w: list[tuple[float, int]] = []
        for node in enters:
            sim = self._score(node, query)
            heappush(candidates, (-sim, node))
            heappush(w, (sim, node))
        while candidates:
            sim, node = heappop(candidates)
            sim = -sim
            worst = w[0][0]
            if sim < worst and len(w) >= ef:
                break
            for neighbor, nsim in self._neighbor_scores(node, level, query):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                if nsim > worst or len(w) < ef:
                    heappush(candidates, (-nsim, neighbor))
                    heappush(w, (nsim, neighbor))
                    if len(w) > ef:
                        heappop(w)
                    worst = w[0][0]
        return [idx for _, idx in sorted(w, reverse=True)]

    def _select(self, query: np.ndarray, candidates: list[int], m: int) -> list[int]:
        if not candidates:
            return []
        ids = np.asarray(candidates, dtype=np.int64)
        scores = self._vectors[ids] @ query
        order = np.argsort(-scores)[:m]
        return [int(ids[i]) for i in order]

    def _neighbors(self, node: int, level: int) -> list[int]:
        layers = self._graph[node]
        if level >= len(layers):
            return []
        return layers[level]

    def _neighbor_scores(
        self, node: int, level: int, query: np.ndarray
    ) -> list[tuple[int, float]]:
        ids = self._neighbors(node, level)
        if not ids:
            return []
        arr = np.asarray(ids, dtype=np.int64)
        scores = self._vectors[arr] @ query
        return [(int(i), float(s)) for i, s in zip(arr, scores)]

    def _score(self, idx: int, query: np.ndarray) -> float:
        return float(self._vectors[idx] @ query)

    def search(
        self,
        query_vector: np.ndarray,
        k: int,
        *,
        ef_search: int | None = None,
    ) -> list[tuple[int, float]]:
        if k < 1:
            raise ValueError("k must be >= 1")
        resolved = self.ef_search if ef_search is None else int(ef_search)
        if resolved < 1:
            raise ValueError("ef_search must be >= 1")
        if self._vectors.shape[0] == 0:
            return []
        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim == 1:
            pass
        elif query.ndim == 2 and query.shape[0] == 1:
            query = query[0]
        else:
            raise ValueError(f"Expected a single query vector, got shape {query.shape}")
        if query.shape[0] != self.dim:
            raise ValueError(f"Query dim {query.shape[0]} does not match index dim {self.dim}")

        enter = self._enter
        for lc in range(self._max_level, 0, -1):
            enter = self._greedy(query, enter, lc)
        hits = self._search_layer(query, [enter], max(resolved, k), 0)
        scored = [(idx, self._score(idx, query)) for idx in hits]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as handle:
            np.savez(
                handle,
                vectors=self._vectors,
                enter=np.asarray([self._enter], dtype=np.int64),
                max_level=np.asarray([self._max_level], dtype=np.int64),
                m=np.asarray([self.m], dtype=np.int64),
                graph=np.asarray(self._graph, dtype=object),
            )

    def load(self, path: str | Path) -> None:
        data = np.load(path, allow_pickle=True)
        self._vectors = np.ascontiguousarray(data["vectors"], dtype=np.float32)
        if self._vectors.ndim != 2 or self._vectors.shape[1] != self.dim:
            raise ValueError(f"Native HNSW at {path} has shape {self._vectors.shape}")
        self._enter = int(data["enter"][0])
        self._max_level = int(data["max_level"][0])
        self.m = int(data["m"][0])
        self._graph = [list(layers) for layers in data["graph"].tolist()]

    @property
    def ntotal(self) -> int:
        return int(self._vectors.shape[0])

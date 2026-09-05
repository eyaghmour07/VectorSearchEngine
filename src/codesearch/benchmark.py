from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import numpy as np

from codesearch.chunker import ChunkStrategy, apply_strategy
from codesearch.embedder import DEFAULT_CACHE_DIR, MODEL_NAME, Embedder
from codesearch.index import DEFAULT_EF_SEARCH, FaissFlatIndex, FaissHNSWIndex
from codesearch.models import CodeChunk
from codesearch.store import DEFAULT_STORE_DIR, load_index

DEFAULT_GROUND_TRUTH = Path("data") / "ground_truth.json"
DEFAULT_RESULTS_PATH = Path("benchmark_results.json")
EF_SWEEP = [16, 32, 64, 128, 256]
WARMUP_RUNS = 5
TIMED_RUNS = 20
EVAL_KS = (1, 5, 10)
MAX_K = 10


def run_benchmark(
    *,
    ground_truth_path: str | Path = DEFAULT_GROUND_TRUTH,
    store_dir: str | Path = DEFAULT_STORE_DIR,
    output_path: str | Path = DEFAULT_RESULTS_PATH,
) -> dict:
    entries = load_ground_truth(ground_truth_path)
    stored = load_index(
        store_dir,
        model_name=MODEL_NAME,
        chunk_strategy=_read_strategy(store_dir),
    )
    chunks = stored.chunks
    if stored.index.ntotal < 2:
        raise ValueError("Index is too small to benchmark.")

    embedder = Embedder(model_name=MODEL_NAME, cache_dir=DEFAULT_CACHE_DIR)
    texts = [chunk.embedding_text for chunk in chunks]
    vectors = embedder.encode(texts)
    queries = [embedder.encode_query(entry["query"]) for entry in entries]

    flat, flat_build_s, flat_bytes = _build_and_size(FaissFlatIndex(vectors.shape[1]), vectors)
    hnsw, hnsw_build_s, hnsw_bytes = _build_and_size(FaissHNSWIndex(vectors.shape[1]), vectors)

    exact_topk = [flat.search(query, MAX_K) for query in queries]
    results = [
        _measure_row(
            index_type="flat",
            ef_search=None,
            index=flat,
            queries=queries,
            exact_topk=exact_topk,
            recall_is_exact=True,
        )
    ]
    for ef in EF_SWEEP:
        results.append(
            _measure_row(
                index_type="hnsw",
                ef_search=ef,
                index=hnsw,
                queries=queries,
                exact_topk=exact_topk,
                recall_is_exact=False,
            )
        )

    human_hits = _human_hit_at_k(flat, chunks, queries, entries, k=10)
    strategy_hits = _strategy_hit_rates(chunks, entries, embedder)

    payload = {
        "repo_path": stored.meta.repo_path,
        "model_name": stored.meta.model_name,
        "chunk_strategy": stored.meta.chunk_strategy,
        "chunk_count": len(chunks),
        "query_count": len(entries),
        "ground_truth_path": str(ground_truth_path),
        "indexes": [
            {
                "index_type": "flat",
                "build_seconds": flat_build_s,
                "disk_bytes": flat_bytes,
                "ntotal": flat.ntotal,
            },
            {
                "index_type": "hnsw",
                "build_seconds": hnsw_build_s,
                "disk_bytes": hnsw_bytes,
                "ntotal": hnsw.ntotal,
                "M": 32,
                "efConstruction": 200,
            },
        ],
        "results": results,
        "human_hit@10": human_hits,
        "strategy_hit@10": strategy_hits,
        "output_path": str(output_path),
        "recall_warning": _perfect_recall_warning(results, chunk_count=len(chunks)),
    }
    Path(output_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def load_ground_truth(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Create it by hand; do not generate it from search output."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not raw:
        raise ValueError(
            f"{path} is empty. Ground truth must be written by hand against the target repo. "
            "Leaving it empty is correct until then — the benchmark will not invent queries."
        )
    if not isinstance(raw, list):
        raise ValueError(f"{path} must be a JSON list of {{query, relevant_ids}} objects.")
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict) or "query" not in entry or "relevant_ids" not in entry:
            raise ValueError(f"Invalid ground-truth entry at index {i}")
        if not entry["query"] or not entry["relevant_ids"]:
            raise ValueError(f"Ground-truth entry {i} is missing query or relevant_ids")
    return raw


def _read_strategy(store_dir: str | Path) -> str:
    meta_path = Path(store_dir) / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}. Run `codesearch index` first.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta["chunk_strategy"]


def _build_and_size(index, vectors: np.ndarray) -> tuple:
    start = time.perf_counter()
    index.build(vectors)
    build_s = time.perf_counter() - start
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "index.faiss"
        index.save(path)
        disk_bytes = path.stat().st_size
    return index, build_s, disk_bytes


def _measure_row(
    *,
    index_type: str,
    ef_search: int | None,
    index,
    queries: list[np.ndarray],
    exact_topk: list[list[tuple[int, float]]],
    recall_is_exact: bool,
) -> dict:
    approx_topk: list[list[tuple[int, float]]] = []
    for query in queries:
        approx_topk.append(index.search(query, MAX_K, ef_search=ef_search))

    if recall_is_exact:
        recalls = {f"recall@{k}": 1.0 for k in EVAL_KS}
    else:
        recalls = {
            f"recall@{k}": _mean_recall(exact_topk, approx_topk, k) for k in EVAL_KS
        }

    samples_ms = _latency_samples_ms(index, queries, ef_search=ef_search)
    p50 = float(np.percentile(samples_ms, 50))
    p95 = float(np.percentile(samples_ms, 95))
    total_s = float(np.sum(samples_ms) / 1000.0)
    qps = (len(samples_ms) / total_s) if total_s > 0 else 0.0
    return {
        "index_type": index_type,
        "ef_search": ef_search,
        "recall@1": recalls["recall@1"],
        "recall@5": recalls["recall@5"],
        "recall@10": recalls["recall@10"],
        "p50_ms": p50,
        "p95_ms": p95,
        "qps": qps,
        "timed_runs": TIMED_RUNS,
        "warmup_runs": WARMUP_RUNS,
        "sample_count": len(samples_ms),
    }


def _mean_recall(
    exact_topk: list[list[tuple[int, float]]],
    approx_topk: list[list[tuple[int, float]]],
    k: int,
) -> float:
    scores = []
    for exact, approx in zip(exact_topk, approx_topk):
        truth = {idx for idx, _ in exact[:k]}
        pred = {idx for idx, _ in approx[:k]}
        if not truth:
            scores.append(0.0)
        else:
            scores.append(len(truth & pred) / len(truth))
    return float(np.mean(scores)) if scores else 0.0


def _latency_samples_ms(index, queries: list[np.ndarray], *, ef_search: int | None) -> np.ndarray:
    for query in queries:
        for _ in range(WARMUP_RUNS):
            index.search(query, MAX_K, ef_search=ef_search)
    samples: list[float] = []
    for query in queries:
        for _ in range(TIMED_RUNS):
            start = time.perf_counter()
            index.search(query, MAX_K, ef_search=ef_search)
            samples.append((time.perf_counter() - start) * 1000.0)
    return np.asarray(samples, dtype=np.float64)


def _human_hit_at_k(
    index,
    chunks: list[CodeChunk],
    queries: list[np.ndarray],
    entries: list[dict],
    k: int,
) -> dict:
    hits = 0
    for query_vec, entry in zip(queries, entries):
        result_ids = {chunks[idx].id for idx, _ in index.search(query_vec, k)}
        if result_ids & set(entry["relevant_ids"]):
            hits += 1
    return {
        "k": k,
        "hits": hits,
        "queries": len(entries),
        "rate": hits / len(entries) if entries else 0.0,
    }


def _strategy_hit_rates(
    chunks: list[CodeChunk],
    entries: list[dict],
    embedder: Embedder,
) -> dict[str, float]:
    original = [chunk.embedding_text for chunk in chunks]
    rates: dict[str, float] = {}
    try:
        for strategy in ChunkStrategy:
            apply_strategy(chunks, strategy)
            vectors = embedder.encode([chunk.embedding_text for chunk in chunks])
            flat = FaissFlatIndex(vectors.shape[1])
            flat.build(vectors)
            query_vecs = [embedder.encode_query(entry["query"]) for entry in entries]
            rates[strategy.value] = _human_hit_at_k(flat, chunks, query_vecs, entries, k=10)["rate"]
    finally:
        for chunk, text in zip(chunks, original):
            chunk.embedding_text = text
    return rates


def _perfect_recall_warning(results: list[dict], *, chunk_count: int) -> str | None:
    hnsw_rows = [row for row in results if row["index_type"] == "hnsw"]
    if not hnsw_rows:
        return None
    if any(row["recall@10"] < 1.0 for row in hnsw_rows):
        return None
    return (
        "HNSW recall@10 was 1.0 at every efSearch. That usually means the corpus is too "
        f"small (n={chunk_count}) or the HNSW index is not actually approximating."
    )

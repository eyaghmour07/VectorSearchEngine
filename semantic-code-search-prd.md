# PRD: Semantic Code Search CLI

## Overview

A command-line tool that indexes a Python repository and lets a user search it by natural-language description of behavior rather than by identifier name. Grep finds `retryAuth`; this finds `handleLoginBackoff` when the user asks for "retry logic after failed login."

The tool also ships a benchmark mode that compares an exact (brute-force) index against an approximate nearest-neighbor index on recall and latency, so the accuracy/speed tradeoff is measured rather than assumed.

## The point of this project

**The deliverable is not "a search tool that works." It is a measured accuracy/latency tradeoff.**

Anyone can wire an embedding model to a vector library and get plausible-looking results. What makes this project worth building is the benchmark: an honest, reproducible measurement of how much exactness approximate search gives up in exchange for speed, on a real corpus, against an exact baseline.

A finished project produces a sentence like: *"HNSW returned results 40x faster than exact search at 97% recall@10, and here is the curve showing how that degrades as efSearch drops."*

If at the end there is a working search tool but no trustworthy numbers, the project has failed at its actual purpose. Prioritize accordingly: correctness of the benchmark matters more than polish of the CLI.

## Goals

1. Index a mid-sized Python repo (target: 1k–10k functions) into a persistent vector index.
2. Answer natural-language queries with ranked function results, each with file path and line numbers.
3. Provide a reproducible benchmark comparing exact vs. approximate search on recall@k and query latency.

## Non-goals

- No web UI. CLI only.
- No multi-language support. Python source only.
- No multi-repo indexing (single repo per index).
- No LLM generation/summarization of results. Retrieval only.

## Do not do these things

These are the failure modes that would quietly ruin the project. They matter more than any feature below.

**Do not generate the ground truth set.** `data/ground_truth.json` is written by hand by a human reading the repo. Do not populate it from the tool's own search output, do not generate queries with an LLM, and do not infer relevant functions from embedding similarity. Ground truth derived from the system under test measures nothing. If this file is empty, leave it empty and say so — do not fill it to make the benchmark run.

**Do not fabricate, estimate, or placeholder any benchmark number.** Every figure in `benchmark_results.json` and the README must come from an actual measured run on this machine. No illustrative examples, no "typical" values, no numbers copied from FAISS documentation. If a benchmark has not been run yet, the README table stays empty.

**Do not report recall as 100% or omit it.** HNSW is approximate. If the measured recall is 100%, that is a bug in the measurement — most likely the flat and HNSW indexes are returning identical results because the corpus is too small or the index fell back to exact search. Investigate rather than reporting it.

**Do not skip normalization or mismatch the metric.** All vectors must be L2-normalized, and both indexes must use inner product. A mismatch here produces results that look reasonable and are silently wrong — the worst possible failure for this project.

**Do not swallow exceptions to keep things running.** A `SyntaxError` in a parsed source file is expected and should warn-and-continue. Everything else — a failed index load, a metadata mismatch, an embedding dimension mismatch — must raise loudly. No bare `except:`, no `except Exception: pass`.

**Do not add features that are not in this document.** No web UI, no LLM summarization of results, no query rewriting, no reranking layer, no multi-language parsing, no Docker setup, no CI config. Scope creep is the main risk to finishing.

**Do not build components out of order.** Exact (flat) search must work end to end and be manually verified before HNSW is written. HNSW is an optimization measured against the flat baseline; without a trusted baseline there is nothing to measure against.

**Do not pad the index with noise.** Test files, `setup.py`, virtualenv contents, and sub-3-line function bodies stay out. A bloated index makes both search quality and benchmark numbers worse and less interpretable.

## Tech stack

- Python 3.11+
- `ast` (stdlib) for parsing Python source
- `sentence-transformers` for embeddings — model: `all-MiniLM-L6-v2` (fast, 384-dim, CPU-friendly)
- `faiss-cpu` for indexing
- `typer` for the CLI
- `rich` for result formatting
- `numpy`, `pytest`

Everything runs locally. No API keys, no network calls at query time.

## Architecture

```
src/
  cli.py            # typer entrypoint, three commands
  parser.py         # repo -> list[CodeChunk]
  chunker.py        # CodeChunk -> embedding input text
  embedder.py       # text -> vectors (batched)
  index.py          # FaissFlatIndex, FaissHNSWIndex (shared interface)
  store.py          # persist/load index + chunk metadata
  benchmark.py      # recall@k and latency measurement
  models.py         # CodeChunk, SearchResult dataclasses
tests/
data/
  ground_truth.json # query -> expected function ids
```

### Data model

```python
@dataclass
class CodeChunk:
    id: str              # f"{relpath}::{qualname}"
    repo_path: str
    file_path: str       # relative to repo root
    qualname: str        # e.g. "Session.request"
    start_line: int
    end_line: int
    signature: str
    docstring: str | None
    body: str
    embedding_text: str  # what actually gets embedded

@dataclass
class SearchResult:
    chunk: CodeChunk
    score: float         # cosine similarity, higher is better
    rank: int
```

## Component requirements

### parser.py

Walk the repo, parse each `.py` file with `ast`, and extract every `FunctionDef`, `AsyncFunctionDef`, and `ClassDef` method as a `CodeChunk`.

- Skip: `test_*.py`, `*_test.py`, `setup.py`, anything under `.venv/`, `venv/`, `node_modules/`, `.git/`, `build/`, `dist/`.
- Skip functions under 3 lines of body (getters, `pass` stubs) — they add index noise.
- Use `ast.get_docstring()` for docstrings.
- Reconstruct signatures from the AST args node (name, args, defaults, return annotation if present).
- On a `SyntaxError` in any file, log a warning and continue. One bad file must not abort indexing.
- `qualname` must include the class for methods: `Session.request`, not `request`.

### chunker.py

Build the `embedding_text` for each chunk. Default strategy:

```
{qualname}
{signature}
{docstring or ""}
{first 30 lines of body}
```

Truncate the whole thing to 512 tokens' worth of characters (approximate with a 2000-char cap; the model truncates anyway).

Make the strategy swappable via an enum (`SIGNATURE_ONLY`, `SIGNATURE_DOCSTRING`, `FULL`) with `FULL` as the default, so the benchmark can compare them.

### embedder.py

- Load `all-MiniLM-L6-v2` once, reuse it.
- Batch encode (batch size 64) with a progress bar.
- L2-normalize all vectors so inner product equals cosine similarity.
- Cache embeddings on disk keyed by a hash of `embedding_text`, so re-indexing an unchanged repo is fast.

### index.py

Define an abstract `VectorIndex` with `build(vectors)`, `search(query_vector, k) -> list[(idx, score)]`, `save(path)`, `load(path)`.

Two implementations:
- `FaissFlatIndex` — `IndexFlatIP`. Exact. This is the ground-truth baseline.
- `FaissHNSWIndex` — `IndexHNSWFlat`, `M=32`, `efConstruction=200`, `efSearch` configurable at query time (default 64).

Both take normalized vectors and use inner product.

### store.py

Persist to `.codesearch/` in the current working directory:
- `index.faiss` — the FAISS index
- `chunks.jsonl` — one `CodeChunk` per line, in index order
- `meta.json` — repo path, model name, chunk strategy, index type, chunk count, build timestamp

On load, validate that the model name and chunk strategy in `meta.json` match the current config; error clearly if they don't.

### benchmark.py

Two measurements, both reported as a table:

**Recall@k:** treat `FaissFlatIndex` results as ground truth. For each query in `data/ground_truth.json`, compute what fraction of the flat index's top-k appears in the HNSW top-k. Report recall@1, @5, @10.

**Latency:** run each query 20 times per index (5 warmup runs discarded), report p50 and p95 in milliseconds, plus queries/second.

Also report index build time and on-disk index size for each type.

Run the sweep across `efSearch` in `[16, 32, 64, 128, 256]` so the recall/latency curve is visible, and write results to `benchmark_results.json`.

## CLI commands

```bash
# Index a repo
codesearch index /path/to/repo [--index-type flat|hnsw] [--strategy full|sig|sig-doc]

# Query
codesearch search "retry logic after a failed request" [-k 10] [--ef-search 64]

# Benchmark
codesearch benchmark [--ground-truth data/ground_truth.json]
```

Search output, via `rich`:

```
1. Session.request                              score 0.782
   requests/sessions.py:502-589
   Constructs a Request, prepares it, and sends it.

2. HTTPAdapter.send                             score 0.741
   requests/adapters.py:434-501
   Sends PreparedRequest object. Returns Response object.
```

## Ground truth set

Create `data/ground_truth.json` with 25–30 hand-written entries against the target repo:

```json
[
  {
    "query": "retry a request after a connection failure",
    "relevant_ids": ["requests/adapters.py::HTTPAdapter.send"]
  }
]
```

These are written by hand by inspecting the repo. They are the honest evaluation set — do not generate them from the search tool's own output.

## Acceptance criteria

- [ ] `codesearch index` on a repo of ~2k functions completes without crashing and reports chunk count.
- [ ] Re-running `index` on an unchanged repo is significantly faster (embedding cache hit).
- [ ] `codesearch search` returns ranked results with correct file paths and line numbers.
- [ ] Line numbers in output match the actual function location in the source file.
- [ ] `codesearch benchmark` produces a table with recall@k and p50/p95 latency for both index types across the `efSearch` sweep.
- [ ] HNSW is measurably faster than flat at scale, with recall reported honestly (not claimed as 100%).
- [ ] A malformed Python file in the repo produces a warning, not a crash.
- [ ] `pytest` passes.

## Tests

- `test_parser.py` — nested functions, async functions, class methods get correct qualnames; decorators don't break parsing; syntax errors are skipped gracefully.
- `test_chunker.py` — each strategy produces expected text; truncation works.
- `test_index.py` — flat index returns exact nearest neighbors on a small synthetic set; save/load round-trips.
- `test_store.py` — metadata mismatch on load raises a clear error.

## README requirements

Must include:
- One-line description of the problem (grep matches names, not behavior).
- Install + quickstart (index, search, benchmark).
- **A benchmark results table with real numbers** — recall@10 and p50 latency for flat vs. HNSW at several `efSearch` values.
- A short "tradeoffs and what I learned" section: why HNSW trades exactness for speed, what recall level the tool actually achieves, and what chunking strategy performed best.
- Which repo was indexed and how large the index is.

## Build order

1. `models.py`, `parser.py` + tests — verify chunk extraction on the target repo first.
2. `chunker.py`, `embedder.py` — get embeddings working and cached.
3. `index.py` (flat only), `store.py`, `cli.py` `index` and `search` — end-to-end working search.
4. `index.py` (HNSW).
5. `data/ground_truth.json` — hand-written.
6. `benchmark.py` + `cli.py` `benchmark`.
7. README with real numbers.

Get step 3 fully working before touching HNSW. A working exact search is the foundation; approximate search is an optimization on top of it.

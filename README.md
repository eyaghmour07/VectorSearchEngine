# codesearch

Grep matches names. This CLI matches behavior: it indexes Python functions as vectors and ranks them by a natural-language description of what they do.

The point of the project is not “search that looks plausible.” It is a measured accuracy/latency tradeoff between exact (flat) search and HNSW.

## Install

Python 3.11+. No API keys. The embedding model downloads once, then everything is local.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The CLI exits if it is launched with an older interpreter. Python 3.9 skipped four Django production files (`asgi.py`, `defaulttags.py`, `choices.py`, `json.py`) and produced a different corpus.

## Quickstart

Index [psf/requests](https://github.com/psf/requests). Clone it first — `/path/to/requests` is not a real folder.

```bash
git clone --depth 1 https://github.com/psf/requests.git /tmp/requests-corpus
codesearch index /tmp/requests-corpus/src --store-dir .codesearch-requests --index-type flat --strategy full
codesearch search "retry a request after a connection failure" --store-dir .codesearch-requests
codesearch benchmark --store-dir .codesearch-requests --ground-truth data/ground_truth.json --output benchmark_results_requests.json
```

Django is a separate corpus with its own labels and its own result file:

```bash
git clone --depth 1 https://github.com/django/django.git /tmp/django-corpus
codesearch index /tmp/django-corpus/django --store-dir .codesearch-django
codesearch benchmark --store-dir .codesearch-django --ground-truth data/ground_truth_django.json --output benchmark_results_django.json
```

Do not point the Django index at `data/ground_truth.json`. Those queries are Requests-specific.

Label provenance is in `data/GROUND_TRUTH.md`. Review those files before treating them as a paper-grade annotation set.

## What was indexed

Measured 2026-09-05 on this machine (macOS arm64, 8 cores, FAISS pinned to 1 thread). Model `all-MiniLM-L6-v2` revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Python 3.12.14, faiss-cpu 1.15.0, sentence-transformers 6.0.1, torch 2.14.0.

| corpus | git SHA | functions after filters | labels | result file |
| --- | --- | --- | --- | --- |
| `psf/requests` `src/` | `dae7ef63b4df6eded86637f251fc4e3a06c3b479` | 151 | `data/ground_truth.json` (30) | `benchmark_results_requests.json` |
| `django/django` `django/` | `b3f4d83aad7f589f165a6d8b020b7acba4936f35` | 6187 | `data/ground_truth_django.json` (28) | `benchmark_results_django.json` |

`requests` is the PRD example corpus, but 151 vectors is too small for HNSW to beat exact search. The published speed/recall numbers are from Django.

On-disk Django index size: flat **9.1 MB**, HNSW **10.7 MB**. HNSW build time: **0.692 s**.

## Benchmark results

Recall@k is HNSW vs the flat index (flat is the referee), not vs the JSON labels. Each query ran 5 warmup + 20 timed iterations. p50/p95 are over all timed samples.

HNSW rows with recall@10 = 1.0 are stored in the JSON `results` array and omitted here. The PRD treats perfect ANN recall as a measurement bug unless investigated; on this run that happened at `efSearch` 64 and above.

### Django (6187 vectors) — published table

From `benchmark_results_django.json` `publishable_results`:

| index | efSearch | recall@10 | p50 latency (ms) | speedup vs flat |
| --- | --- | --- | --- | --- |
| flat | — | 1.000 | 0.144 | 1.0× |
| hnsw | 16 | 0.982 | 0.016 | 9.2× |
| hnsw | 32 | 0.996 | 0.022 | 6.6× |

Headline: **HNSW at `efSearch=16` was 9.2× faster than exact search at 98.2% recall@10.** Raising `efSearch` to 32 recovered 99.6% recall and was still 6.6× faster.

Labeled hit@10 on the Django set (flat, `full` chunking): **20/28**.

### Requests (151 vectors) — small-corpus check

From `benchmark_results_requests.json` `publishable_results`:

| index | efSearch | recall@10 | p50 latency (ms) | speedup vs flat |
| --- | --- | --- | --- | --- |
| flat | — | 1.000 | 0.009 | 1.0× |
| hnsw | 16 | 0.983 | 0.009 | 1.0× |
| hnsw | 32 | 0.997 | 0.010 | 0.8× |

HNSW is not faster than flat when the corpus fits in a single brute-force scan. That is why the published headline uses Django.

Labeled hit@10 on the Requests set (flat, `full` chunking): **29/30**.

`benchmark_results.json` is not a measured artifact. Use the two corpus-specific files above.

## Tradeoffs and what I learned

HNSW trades exactness for speed by walking a sparse neighbor graph instead of scoring every vector. `efSearch` is “how hard should I look?” Lower is faster and missier; higher converges to the flat ranking and gives the speedup back.

On 6187 functions, `efSearch=16` is the useful published operating point: 98.2% of the exact top-10, 9.2× faster. `efSearch=32` is the conservative point. Settings that reported 100% recall@10 are not published.

On 151 functions the approximation is pointless.

Chunking strategy, measured as labeled hit@10 on each hand-written set (flat index):

| strategy | requests (30) | django (28) |
| --- | --- | --- |
| `full` (qualname + signature + docstring + first 30 body lines) | 29/30 | 20/28 |
| `sig` (qualname + signature) | 28/30 | 12/28 |
| `sig-doc` (qualname + signature + docstring) | 27/30 | 19/28 |

`full` was best on both corpora. Including a slice of the body helped more than the docstring alone.

Vectors are L2-normalized and both indexes use inner product, so a score is cosine similarity. Mixing L2 distance with IP would look reasonable and be silently wrong.

A `SyntaxError` in one source file logs a warning and indexing continues. That is the only exception the parser swallows.

FAISS and PyTorch each ship an OpenMP runtime. Embedding runs in a worker process so those libraries never initialize two copies in the same address space. The process does not set `KMP_DUPLICATE_LIB_OK`.

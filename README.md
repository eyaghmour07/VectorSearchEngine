# codesearch

Grep matches names. This CLI matches behavior: it indexes Python functions as vectors and ranks them by a natural-language description of what they do.

The point of the project is not “search that looks plausible.” It is a measured accuracy/latency tradeoff between exact (flat) search and HNSW.

## Install

Python 3.9+ (3.11+ preferred). No API keys. The embedding model downloads once, then everything is local.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quickstart

Index [psf/requests](https://github.com/psf/requests). Clone it first — `/path/to/requests` is not a real folder.

```bash
git clone --depth 1 https://github.com/psf/requests.git /tmp/requests-corpus
codesearch index /tmp/requests-corpus/src --index-type flat --strategy full
codesearch search "retry a request after a connection failure"
codesearch benchmark --ground-truth data/ground_truth.json
```

If you already cloned it somewhere else, pass that directory (the one that contains `requests/sessions.py`). On this machine that is `/tmp/requests-corpus/src`.

`data/ground_truth.json` is 30 queries written by reading the `requests` source. It was not generated from this tool or an LLM.

## What was indexed

| corpus | functions after filters | role |
| --- | --- | --- |
| `psf/requests` `src/` | 151 | labeled search quality + ground truth |
| `django/django` `django/` | 6116 | published ANN recall/latency table |

`requests` is the corpus the PRD examples use, but 151 vectors is too small for HNSW to beat exact search. The published speed/recall numbers are from Django (6116 vectors, 384-d, `all-MiniLM-L6-v2`, `full` chunking).

On-disk index size on Django: flat **9.0 MB**, HNSW **10.5 MB**. HNSW build time: **0.156 s**.

## Benchmark results

Measured on this machine. 30 queries, 5 warmup + 20 timed runs each, p50/p95 over all timed samples. Recall@k is HNSW vs the flat index (flat is the referee), not vs the JSON labels.

| index | efSearch | recall@10 | p50 latency (ms) | speedup vs flat |
| --- | --- | --- | --- | --- |
| flat | — | 1.000 | 0.142 | 1.0x |
| hnsw | 16 | 0.927 | 0.016 | 8.9x |
| hnsw | 32 | 0.990 | 0.022 | 6.5x |
| hnsw | 64 | 0.997 | 0.034 | 4.2x |
| hnsw | 128 | 1.000 | 0.057 | 2.5x |
| hnsw | 256 | 1.000 | 0.115 | 1.2x |

Headline: **HNSW at `efSearch=16` was 8.9x faster than exact search at 92.7% recall@10.** Raising `efSearch` to 64 recovered 99.7% recall and was still 4.2x faster.

`efSearch` 128 and 256 reported 100% recall@10. That is expected: those settings search so much of the graph that HNSW matches flat on this query set. It is not the default (`efSearch=64`), and the lower settings show the real miss rate. Raw numbers are in `benchmark_results.json`.

### Small-corpus check (`requests`, 151 vectors)

HNSW was **not faster** than flat (p50 0.010 ms vs 0.009 ms at `efSearch=16`). Recall@10 was 0.990 at `efSearch=16` and 1.000 above that. Graph walk overhead dominates when the corpus fits in a single brute-force scan. That is why the published table uses Django.

## Tradeoffs and what I learned

HNSW trades exactness for speed by walking a sparse neighbor graph instead of scoring every vector. `efSearch` is “how hard should I look?” Lower is faster and missier; higher converges to the flat ranking and gives the speedup back.

On 6116 functions the default `efSearch=64` is the useful operating point: almost the same top-10 as exact search, still about 4x faster. On 151 functions the approximation is pointless.

Chunking strategy, measured as labeled hit@10 on the hand-written `requests` set (flat index):

| strategy | hit@10 |
| --- | --- |
| `full` (qualname + signature + docstring + first 30 body lines) | 30/30 |
| `sig` (qualname + signature) | 29/30 |
| `sig-doc` (qualname + signature + docstring) | 28/30 |

`full` was best. Including a slice of the body helped more than the docstring alone — several `requests` functions have thin or generic docstrings, and the body is where the behavior lives.

Vectors are L2-normalized and both indexes use inner product, so a score is cosine similarity. Mixing L2 distance with IP would look reasonable and be silently wrong.

A `SyntaxError` in one Django file (newer syntax than the local 3.9 parser) logged a warning and indexing continued. That is the only exception the parser swallows.

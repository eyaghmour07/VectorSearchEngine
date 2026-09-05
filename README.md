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

The CLI exits if it is launched with an older interpreter. Python 3.9 skipped four Django production files and produced a different corpus.

```bash
pytest
```

44 tests (43 pass on 3.12; one skip is the Python 3.11+ CLI guard). There is no CI config; the PRD forbids adding one.

## Quickstart

```bash
git clone --depth 1 https://github.com/psf/requests.git /tmp/requests-corpus
codesearch index /tmp/requests-corpus/src --store-dir .codesearch-requests
codesearch search "retry a request after a connection failure" --store-dir .codesearch-requests
codesearch benchmark --store-dir .codesearch-requests --ground-truth data/ground_truth.json --output benchmark_results_requests.json
```

Django and the 52k mixed corpus use their own stores, labels, and result files. Do not point a Django index at `data/ground_truth.json`.

```
$ codesearch search "retry a request after a connection failure" --store-dir .codesearch-requests -k 5

1. SessionRedirectMixin.resolve_redirects    score 0.444
   requests/sessions.py:186-307
   Receives a Response. Returns a generator of Responses or Requests.

2. Session.send                              score 0.432
   requests/sessions.py:752-829
   Send a given PreparedRequest.

3. HTTPAdapter.send                          score 0.425
   requests/adapters.py:634-748
   Sends PreparedRequest object. Returns Response object.

4. Response.ok                               score 0.387
   requests/models.py:862-874
   Returns True if :attr:`status_code` is less than 400, False if not.

5. rewind_body                               score 0.373
   requests/utils.py:1139-1155
   Move file pointer back to its recorded starting position
```

![codesearch search on the Requests index](docs/cli-search.png)

Label provenance is in `data/GROUND_TRUTH.md`.

## What was indexed

Measured 2026-09-05 on this machine (macOS arm64, 8 cores, FAISS pinned to 1 thread). Model `all-MiniLM-L6-v2` revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Python 3.12.14, faiss-cpu 1.15.0, sentence-transformers 6.0.1, torch 2.14.0.

| corpus | functions | labels | result file |
| --- | --- | --- | --- |
| `psf/requests` `src/` (`dae7ef63`) | 151 | `data/ground_truth.json` (30) | `benchmark_results_requests.json` |
| `django/django` `django/` (`b3f4d83a`) | 6187 | `data/ground_truth_django.json` (28) | `benchmark_results_django.json` |
| mixed 11-repo tree (below) | 52800 | remapped Requests+Django (58) | `benchmark_results_scale.json` |

The mixed tree is one `codesearch index` root at `/tmp/scale-src`:

| tree | git | functions |
| --- | --- | --- |
| CPython | `7a918411a300` | 15328 |
| SQLAlchemy | `de83fa72d787` | 6300 |
| Django | `b3f4d83aad7f` | 6187 |
| pandas | `70e6c48ff317` | 5892 |
| Ansible | `3827d66bfbdc` | 5043 |
| matplotlib | `f3877c54e25b` | 4633 |
| scikit-learn | `9bafc1c9cab9` | 3890 |
| Sphinx | `e44a40eb2f81` | 2896 |
| NumPy | `22c80bc405bc` | 2214 |
| Flask | `d318b6834711` | 266 |
| Requests | `dae7ef63b4df` | 151 |

CPython main uses syntax newer than 3.12; those files warned and were skipped. That shrinks the stdlib slice but does not invent vectors.

## The finding: the tradeoff appears at 50k, not at 6k

Recall@k is HNSW vs the flat index (flat is the referee). Each query ran 5 warmup + 20 timed iterations. p50 and p95 are over all timed samples.

| corpus | n | flat p50 | HNSW ef16 p50 | speedup | recall@10 |
| --- | --- | --- | --- | --- | --- |
| Requests | 151 | 0.009 ms | 0.009 ms | 1.0× | 0.983 |
| Django | 6187 | 0.144 ms | 0.016 ms | 9.2× | 0.982 |
| mixed | 52800 | 1.483 ms | 0.022 ms | 66× | 0.955 |

At 151 vectors HNSW is not faster than a scan. At 6,187 it is 9.2× faster and saves 0.128 ms — a demo, not a decision. At 52,800 it is 66× faster and saves 1.46 ms. That is the scale where approximate search is a choice.

HNSW’s tail is worse than flat’s, and the gap grows with n. On the mixed corpus, flat p95/p50 is 1.5× (2.218 / 1.483). HNSW at `efSearch=16` is 4.0× (0.090 / 0.022). Faster typical queries, fatter worst-case queries.

### Mixed corpus (52800 vectors)

From `benchmark_results_scale.json` `results`. On-disk: flat 77.3 MB, HNSW 91.1 MB. HNSW build 25.7 s.

| index | efSearch | recall@10 | p50 (ms) | p95 (ms) | speedup |
| --- | --- | --- | --- | --- | --- |
| flat | — | 1.000 | 1.483 | 2.218 | 1.0× |
| hnsw | 16 | 0.955 | 0.022 | 0.090 | 66× |
| hnsw | 32 | 0.991 | 0.030 | 0.087 | 49× |
| hnsw | 64 | 1.000 | 0.048 | 0.181 | 31× |
| hnsw | 128 | 1.000 | 0.084 | 0.242 | 18× |
| hnsw | 256 | 1.000 | 0.175 | 0.425 | 8.5× |

Headline: **HNSW at `efSearch=16` was 66× faster than exact search at 95.5% recall@10.** `efSearch=32` is the conservative point: 99.1% recall, 49× faster.

`efSearch` 64 and above reported recall@10 = 1.0. That is not a silent fallback to exact search — those rows are still 8–31× faster than flat, so the graph walk is not scanning 52,800 vectors. It means this 58-query set did not expose a top-10 miss once the candidate list was wide enough. The misses live at 16 and 32, which is why the full curve stays in the table.

### Django (6187 vectors)

From `benchmark_results_django.json` `results`:

| index | efSearch | recall@10 | p50 (ms) | p95 (ms) | speedup |
| --- | --- | --- | --- | --- | --- |
| flat | — | 1.000 | 0.144 | 0.165 | 1.0× |
| hnsw | 16 | 0.982 | 0.016 | 0.022 | 9.2× |
| hnsw | 32 | 0.996 | 0.022 | 0.031 | 6.6× |
| hnsw | 64 | 1.000 | 0.034 | 0.052 | 4.2× |
| hnsw | 128 | 1.000 | 0.061 | 0.094 | 2.4× |
| hnsw | 256 | 1.000 | 0.122 | 0.207 | 1.2× |

Same 100% investigation as above, on a smaller n: high `efSearch` matches flat on this query set and gives the speedup back.

### Requests (151 vectors)

From `benchmark_results_requests.json` `results`:

| index | efSearch | recall@10 | p50 (ms) | p95 (ms) | speedup |
| --- | --- | --- | --- | --- | --- |
| flat | — | 1.000 | 0.009 | 0.010 | 1.0× |
| hnsw | 16 | 0.983 | 0.009 | 0.011 | 1.0× |
| hnsw | 32 | 0.997 | 0.010 | 0.013 | 0.8× |
| hnsw | 64 | 1.000 | 0.013 | 0.019 | 0.7× |
| hnsw | 128 | 1.000 | 0.021 | 0.032 | 0.4× |
| hnsw | 256 | 1.000 | 0.025 | 0.039 | 0.3× |

Graph walk overhead dominates when the corpus fits in one brute-force scan.

`benchmark_results.json` is a pointer, not a measured run.

## Labeled search quality

Flat hit@10 against the hand-written labels:

| corpus | hit@10 |
| --- | --- |
| Requests | 29/30 |
| Django | 20/28 |
| mixed (same 58 labels, more distractors) | 40/58 |

### Why Django is 71% and Requests is 97%

The eight Django misses were inspected against the top-10 (and top-50) of the flat index. They are not random rank noise. They fall into two classes.

**Overloaded framework vocabulary.** The paraphrase uses a word Django also uses for a different mechanism. “return an existing row or create it if the lookup misses” never retrieved `QuerySet.get_or_create`; it retrieved `Lookup`, `Query.build_lookup`, and `get_lookup` because “lookup” in Django means a field comparison, not a get-or-insert. “set a cookie” ranked `CookieStorage._update_cookie` and `set_signed_cookie` above `HttpResponseBase.set_cookie`. “send an email” ranked `AdminEmailHandler.send_mail` and `EmailBackend._send` above `send_mail`. “join related objects… N+1” ranked `prefetch_related` and SQL `join` helpers above `select_related`.

**Private hooks instead of the public name.** “run a block of ORM writes as one database transaction” is `transaction.atomic`, but the labeled chunk is `Atomic.__enter__` (rank 32). “temporarily change Django settings inside a test” ranked `override_settings.decorate_class` and `modify_settings`; `override_settings.enable` is not in the top 50. The query describes the feature; the label is the implementation entry the parser extracted.

Requests fails less because an HTTP client’s public verbs — retry, redirect, cookie jar, JSON body — line up with both the docstrings and the identifiers. MiniLM is matching shared words, not proving it understands ORM transactions.

Chunking strategy, labeled hit@10 on the per-repo sets (flat):

| strategy | requests (30) | django (28) |
| --- | --- | --- |
| `full` | 29/30 | 20/28 |
| `sig` | 28/30 | 12/28 |
| `sig-doc` | 27/30 | 19/28 |

`full` was best on both. The body slice helps more than the docstring alone.

## Tradeoffs

HNSW trades exactness for speed by walking a sparse neighbor graph. `efSearch` is how hard to look. Lower is faster and missier; higher converges to the flat ranking and gives the speedup back. Publish the whole curve, including the 100% rows — they show that convergence, and the leftover latency vs flat shows the walk is still approximate.

Vectors are L2-normalized and both indexes use inner product. A `SyntaxError` in one source file logs a warning and indexing continues. That is the only exception the parser swallows.

FAISS and PyTorch each ship an OpenMP runtime. Embedding runs in a worker process so those libraries never initialize two copies in the same address space. The process does not set `KMP_DUPLICATE_LIB_OK`.

`--index-type native` is a from-scratch HNSW (inner product, same M / ef defaults). The tables above are still FAISS; native is tested on synthetic vectors and is not the published curve yet.

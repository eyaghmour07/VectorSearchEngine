# Ground-truth provenance

These files are the labeled evaluation sets. Each query names a behavior; `relevant_ids` are function IDs taken from the source, not from `codesearch search` or embedding similarity.

| file | corpus | queries |
| --- | --- | --- |
| `ground_truth.json` | `psf/requests` (`src/`) | 30 |
| `ground_truth_django.json` | `django/django` (`django/`) | 28 |

How the labels were written:

1. Open the listed source file in the cloned corpus.
2. Read the function body and docstring.
3. Write a paraphrase of that behavior that does not copy the identifier.
4. Record the parser ID `relpath::qualname`.

IDs were then checked against a parsed corpus so every label exists after the parser’s filters (tests, `setup.py`, bodies shorter than 3 lines). That check is existence only — it does not choose which function is relevant.

Review the JSON before treating this as a paper-grade human annotation set. The PRD requires a person to own the labels; this file exists so that ownership is explicit rather than implied by a silent commit.

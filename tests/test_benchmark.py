from __future__ import annotations

from pathlib import Path

import pytest

from codesearch.benchmark import (
    GroundTruthError,
    _mean_recall,
    _recall_warnings,
    _validate_ground_truth_ids,
    load_ground_truth,
)
from codesearch.models import CodeChunk


def _chunk(chunk_id: str) -> CodeChunk:
    return CodeChunk(
        id=chunk_id,
        repo_path="/tmp/repo",
        file_path=chunk_id.split("::")[0],
        qualname=chunk_id.split("::")[1],
        start_line=1,
        end_line=10,
        signature="def fn():",
        docstring=None,
        body="a = 1\nb = 2\nreturn a + b",
        embedding_text="fn",
    )


def test_mean_recall_is_fraction_of_exact_topk() -> None:
    exact = [[(0, 1.0), (1, 0.9), (2, 0.8)], [(3, 1.0), (4, 0.5), (5, 0.1)]]
    approx = [[(0, 1.0), (7, 0.4), (2, 0.3)], [(9, 0.9), (4, 0.4), (5, 0.2)]]
    assert _mean_recall(exact, approx, k=1) == 0.5
    assert _mean_recall(exact, approx, k=3) == pytest.approx(2 / 3)


def test_validate_ground_truth_ids_rejects_unknown_chunks() -> None:
    entries = [{"query": "retry", "relevant_ids": ["mod.py::missing"]}]
    with pytest.raises(GroundTruthError, match="not in the indexed corpus"):
        _validate_ground_truth_ids(entries, [_chunk("mod.py::present")], "data/gt.json")


def test_validate_ground_truth_ids_accepts_present_chunks() -> None:
    entries = [{"query": "retry", "relevant_ids": ["mod.py::present"]}]
    _validate_ground_truth_ids(entries, [_chunk("mod.py::present")], "data/gt.json")


def test_perfect_hnsw_recall_is_not_publishable() -> None:
    results = [
        {"index_type": "flat", "ef_search": None, "recall@10": 1.0},
        {"index_type": "hnsw", "ef_search": 16, "recall@10": 0.9},
        {"index_type": "hnsw", "ef_search": 64, "recall@10": 1.0},
    ]
    warnings = _recall_warnings(results, chunk_count=80)
    assert any("efSearch=64" in item for item in warnings)
    publishable = [row for row in results if row["index_type"] == "flat" or row["recall@10"] < 1.0]
    assert [row["ef_search"] for row in publishable] == [None, 16]


def test_load_ground_truth_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_ground_truth(path)

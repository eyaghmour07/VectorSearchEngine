from __future__ import annotations

import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from codesearch.cli import app
from codesearch.embedder import l2_normalize
from codesearch.index import FaissFlatIndex
from codesearch.models import CodeChunk
from codesearch.store import save_index

import numpy as np

runner = CliRunner()
needs_311 = pytest.mark.skipif(sys.version_info < (3, 11), reason="CLI requires Python 3.11+")


def test_cli_refuses_python_below_311() -> None:
    if sys.version_info >= (3, 11):
        pytest.skip("this interpreter is already 3.11+")
    result = runner.invoke(app, ["search", "retry"])
    assert result.exit_code == 1
    assert "requires Python 3.11+" in result.output


def _chunk(i: int) -> CodeChunk:
    return CodeChunk(
        id=f"mod.py::fn_{i}",
        repo_path="/tmp/repo",
        file_path="mod.py",
        qualname=f"fn_{i}",
        start_line=i * 10,
        end_line=i * 10 + 5,
        signature=f"def fn_{i}():",
        docstring=None,
        body="a = 1\nb = 2\nreturn a + b",
        embedding_text=f"fn_{i}",
    )


@needs_311
def test_search_exits_when_store_is_missing(tmp_path: Path) -> None:
    result = runner.invoke(app, ["search", "retry", "--store-dir", str(tmp_path / "missing")])
    assert result.exit_code == 1
    assert "codesearch index" in result.output


@needs_311
def test_search_exits_on_strategy_mismatch(tmp_path: Path) -> None:
    vectors = l2_normalize(np.eye(4, dtype=np.float32))
    index = FaissFlatIndex(4)
    index.build(vectors)
    save_index(
        tmp_path,
        index,
        [_chunk(i) for i in range(4)],
        repo_path="/tmp/repo",
        model_name="all-MiniLM-L6-v2",
        chunk_strategy="full",
        index_type="flat",
    )
    result = runner.invoke(
        app,
        ["search", "retry", "--store-dir", str(tmp_path), "--strategy", "sig"],
    )
    assert result.exit_code == 1
    assert "chunk_strategy" in result.output


@needs_311
def test_benchmark_exits_when_ground_truth_missing(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "benchmark",
            "--store-dir",
            str(tmp_path / "store"),
            "--ground-truth",
            str(tmp_path / "absent.json"),
            "--output",
            str(tmp_path / "out.json"),
        ],
    )
    assert result.exit_code == 1

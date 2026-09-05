from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from codesearch.embedder import l2_normalize
from codesearch.index import FaissFlatIndex
from codesearch.models import CodeChunk
from codesearch.store import MetadataMismatchError, load_index, save_index


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


def _saved_index(tmp_path: Path, *, strategy: str = "full"):
    vectors = l2_normalize(np.eye(4, dtype=np.float32))
    index = FaissFlatIndex(4)
    index.build(vectors)
    chunks = [_chunk(i) for i in range(4)]
    save_index(
        tmp_path,
        index,
        chunks,
        repo_path="/tmp/repo",
        model_name="all-MiniLM-L6-v2",
        chunk_strategy=strategy,
        index_type="flat",
    )


def test_load_raises_on_strategy_mismatch(tmp_path: Path) -> None:
    _saved_index(tmp_path, strategy="full")
    with pytest.raises(MetadataMismatchError, match="chunk_strategy"):
        load_index(tmp_path, model_name="all-MiniLM-L6-v2", chunk_strategy="sig")


def test_load_raises_on_model_mismatch(tmp_path: Path) -> None:
    _saved_index(tmp_path, strategy="full")
    with pytest.raises(MetadataMismatchError, match="model_name"):
        load_index(tmp_path, model_name="other-model", chunk_strategy="full")


def test_load_succeeds_when_config_matches(tmp_path: Path) -> None:
    _saved_index(tmp_path, strategy="full")
    stored = load_index(tmp_path, model_name="all-MiniLM-L6-v2", chunk_strategy="full")
    assert stored.meta.chunk_count == 4
    assert stored.chunks[0].id == "mod.py::fn_0"
    assert stored.index.ntotal == 4


def test_failed_resave_leaves_previous_index(tmp_path: Path, monkeypatch) -> None:
    _saved_index(tmp_path, strategy="full")
    original = load_index(tmp_path, model_name="all-MiniLM-L6-v2", chunk_strategy="full")
    original_sha = original.meta.index_sha256

    vectors = l2_normalize(np.eye(4, dtype=np.float32))
    index = FaissFlatIndex(4)
    index.build(vectors)
    chunks = [_chunk(i) for i in range(4)]
    chunks[0].qualname = "replaced"

    def boom(path):
        Path(path).write_bytes(b"truncated")
        raise OSError("simulated interrupt")

    monkeypatch.setattr(index, "save", boom)
    with pytest.raises(OSError, match="simulated interrupt"):
        save_index(
            tmp_path,
            index,
            chunks,
            repo_path="/tmp/repo",
            model_name="all-MiniLM-L6-v2",
            chunk_strategy="full",
            index_type="flat",
        )

    restored = load_index(tmp_path, model_name="all-MiniLM-L6-v2", chunk_strategy="full")
    assert restored.chunks[0].qualname == "fn_0"
    assert restored.meta.index_sha256 == original_sha
    assert restored.index.ntotal == 4

from __future__ import annotations

from codesearch.chunker import (
    BODY_LINE_LIMIT,
    MAX_EMBEDDING_CHARS,
    ChunkStrategy,
    build_embedding_text,
)
from codesearch.models import CodeChunk


def _chunk(**overrides) -> CodeChunk:
    data = {
        "id": "mod.py::Foo.bar",
        "repo_path": "/tmp/repo",
        "file_path": "mod.py",
        "qualname": "Foo.bar",
        "start_line": 1,
        "end_line": 20,
        "signature": "def bar(self, x: int) -> int",
        "docstring": "Return x plus one.",
        "body": "y = x + 1\nreturn y",
        "embedding_text": "",
    }
    data.update(overrides)
    return CodeChunk(**data)


def test_signature_only_omits_docstring_and_body() -> None:
    text = build_embedding_text(_chunk(), ChunkStrategy.SIGNATURE_ONLY)
    assert text == "Foo.bar\ndef bar(self, x: int) -> int"
    assert "Return x" not in text
    assert "y = x" not in text


def test_signature_docstring_omits_body() -> None:
    text = build_embedding_text(_chunk(), ChunkStrategy.SIGNATURE_DOCSTRING)
    assert "Foo.bar" in text
    assert "def bar(self, x: int) -> int" in text
    assert "Return x plus one." in text
    assert "y = x" not in text


def test_full_includes_qualname_signature_docstring_and_body() -> None:
    text = build_embedding_text(_chunk(), ChunkStrategy.FULL)
    assert text.splitlines()[0] == "Foo.bar"
    assert "def bar(self, x: int) -> int" in text
    assert "Return x plus one." in text
    assert "y = x + 1" in text


def test_full_uses_empty_string_when_docstring_missing() -> None:
    text = build_embedding_text(_chunk(docstring=None), ChunkStrategy.FULL)
    assert "None" not in text


def test_full_truncates_body_to_first_30_lines() -> None:
    body = "\n".join(f"line_{i} = {i}" for i in range(40))
    text = build_embedding_text(_chunk(body=body), ChunkStrategy.FULL)
    assert "line_0 =" in text
    assert "line_29 =" in text
    assert "line_30 =" not in text
    assert text.count("line_") == BODY_LINE_LIMIT


def test_embedding_text_is_capped() -> None:
    body = "x" * (MAX_EMBEDDING_CHARS + 500)
    text = build_embedding_text(_chunk(body=body, docstring="d" * 100), ChunkStrategy.FULL)
    assert len(text) == MAX_EMBEDDING_CHARS

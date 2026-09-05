from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from codesearch.embedder import EMBED_DIM, MODEL_NAME
from codesearch.index import VectorIndex, create_index
from codesearch.models import CodeChunk
from codesearch.runtime import file_sha256

DEFAULT_STORE_DIR = Path(".codesearch")
INDEX_FILENAME = "index.faiss"
CHUNKS_FILENAME = "chunks.jsonl"
META_FILENAME = "meta.json"


class MetadataMismatchError(ValueError):
    """Raised when a saved index was built with a different model or chunk strategy."""


@dataclass
class IndexMeta:
    repo_path: str
    model_name: str
    chunk_strategy: str
    index_type: str
    chunk_count: int
    build_timestamp: str
    embedding_dim: int = EMBED_DIM
    generation_id: str = ""
    index_sha256: str = ""
    chunks_sha256: str = ""

    def to_dict(self) -> dict:
        return {
            "repo_path": self.repo_path,
            "model_name": self.model_name,
            "chunk_strategy": self.chunk_strategy,
            "index_type": self.index_type,
            "chunk_count": self.chunk_count,
            "build_timestamp": self.build_timestamp,
            "embedding_dim": self.embedding_dim,
            "generation_id": self.generation_id,
            "index_sha256": self.index_sha256,
            "chunks_sha256": self.chunks_sha256,
        }

    @classmethod
    def from_dict(cls, data: dict) -> IndexMeta:
        return cls(
            repo_path=data["repo_path"],
            model_name=data["model_name"],
            chunk_strategy=data["chunk_strategy"],
            index_type=data["index_type"],
            chunk_count=int(data["chunk_count"]),
            build_timestamp=data["build_timestamp"],
            embedding_dim=int(data.get("embedding_dim", EMBED_DIM)),
            generation_id=str(data.get("generation_id", "")),
            index_sha256=str(data.get("index_sha256", "")),
            chunks_sha256=str(data.get("chunks_sha256", "")),
        )


@dataclass
class StoredIndex:
    index: VectorIndex
    chunks: list[CodeChunk]
    meta: IndexMeta
    directory: Path


def save_index(
    directory: str | Path,
    index: VectorIndex,
    chunks: list[CodeChunk],
    *,
    repo_path: str,
    model_name: str,
    chunk_strategy: str,
    index_type: str,
) -> IndexMeta:
    directory = Path(directory)
    if index.ntotal != len(chunks):
        raise ValueError(
            f"Index size {index.ntotal} does not match chunk count {len(chunks)}"
        )
    dim = getattr(index, "dim", None)
    if dim is None:
        raise ValueError("Index is missing a dim attribute")

    parent = directory.parent if directory.parent != Path("") else Path(".")
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".{directory.name}.tmp.{uuid4().hex}"
    backup = parent / f".{directory.name}.bak.{uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        index_path = staging / INDEX_FILENAME
        chunks_path = staging / CHUNKS_FILENAME
        index.save(index_path)
        with chunks_path.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
            handle.flush()
            os_fsync(handle)
        meta = IndexMeta(
            repo_path=repo_path,
            model_name=model_name,
            chunk_strategy=chunk_strategy,
            index_type=index_type,
            chunk_count=len(chunks),
            build_timestamp=datetime.now(timezone.utc).isoformat(),
            embedding_dim=int(dim),
            generation_id=uuid4().hex,
            index_sha256=file_sha256(index_path),
            chunks_sha256=file_sha256(chunks_path),
        )
        meta_path = staging / META_FILENAME
        meta_path.write_text(json.dumps(meta.to_dict(), indent=2) + "\n", encoding="utf-8")
        with meta_path.open("rb") as handle:
            os_fsync(handle)
        if directory.exists():
            directory.rename(backup)
        staging.rename(directory)
        if backup.exists():
            shutil.rmtree(backup)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if backup.exists() and not directory.exists():
            backup.rename(directory)
        raise
    return meta


def os_fsync(handle) -> None:
    handle.flush()
    try:
        import os

        os.fsync(handle.fileno())
    except OSError:
        pass


def load_index(
    directory: str | Path,
    *,
    model_name: str = MODEL_NAME,
    chunk_strategy: str,
    ef_search: int | None = None,
) -> StoredIndex:
    directory = Path(directory)
    meta_path = directory / META_FILENAME
    index_path = directory / INDEX_FILENAME
    chunks_path = directory / CHUNKS_FILENAME
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}. Run `codesearch index` first.")
    if not index_path.exists():
        raise FileNotFoundError(f"Missing FAISS index at {index_path}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunk metadata at {chunks_path}")

    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    meta = IndexMeta.from_dict(raw)
    _validate_meta(meta, model_name=model_name, chunk_strategy=chunk_strategy)
    if meta.index_sha256:
        actual = file_sha256(index_path)
        if actual != meta.index_sha256:
            raise ValueError(
                f"index.faiss checksum mismatch: expected {meta.index_sha256}, got {actual}"
            )
    if meta.chunks_sha256:
        actual = file_sha256(chunks_path)
        if actual != meta.chunks_sha256:
            raise ValueError(
                f"chunks.jsonl checksum mismatch: expected {meta.chunks_sha256}, got {actual}"
            )

    chunks = _load_chunks(chunks_path)
    if len(chunks) != meta.chunk_count:
        raise ValueError(
            f"chunks.jsonl has {len(chunks)} rows but meta.json says {meta.chunk_count}"
        )

    index = create_index(meta.index_type, meta.embedding_dim, ef_search=ef_search or 64)
    index.load(index_path)
    if index.ntotal != len(chunks):
        raise ValueError(
            f"Loaded index ntotal={index.ntotal} does not match {len(chunks)} chunks"
        )
    return StoredIndex(index=index, chunks=chunks, meta=meta, directory=directory)


def _validate_meta(meta: IndexMeta, *, model_name: str, chunk_strategy: str) -> None:
    mismatches: list[str] = []
    if meta.model_name != model_name:
        mismatches.append(
            f"model_name: index has {meta.model_name!r}, current config is {model_name!r}"
        )
    if meta.chunk_strategy != chunk_strategy:
        mismatches.append(
            f"chunk_strategy: index has {meta.chunk_strategy!r}, current config is {chunk_strategy!r}"
        )
    if mismatches:
        raise MetadataMismatchError(
            "Saved index does not match the current config:\n  - "
            + "\n  - ".join(mismatches)
            + "\nRe-run `codesearch index` or pass the matching --strategy."
        )


def _load_chunks(path: Path) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(CodeChunk.from_dict(json.loads(line)))
            except (TypeError, KeyError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid chunk on line {line_no} of {path}: {exc}") from exc
    return chunks

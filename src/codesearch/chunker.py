from __future__ import annotations

from enum import Enum

from codesearch.models import CodeChunk

MAX_EMBEDDING_CHARS = 2000
BODY_LINE_LIMIT = 30


class ChunkStrategy(str, Enum):
    FULL = "full"
    SIGNATURE_ONLY = "sig"
    SIGNATURE_DOCSTRING = "sig-doc"


def apply_strategy(chunks: list[CodeChunk], strategy: ChunkStrategy) -> list[CodeChunk]:
    for chunk in chunks:
        chunk.embedding_text = build_embedding_text(chunk, strategy)
    return chunks


def build_embedding_text(chunk: CodeChunk, strategy: ChunkStrategy) -> str:
    if strategy is ChunkStrategy.SIGNATURE_ONLY:
        text = f"{chunk.qualname}\n{chunk.signature}"
    elif strategy is ChunkStrategy.SIGNATURE_DOCSTRING:
        text = f"{chunk.qualname}\n{chunk.signature}\n{chunk.docstring or ''}"
    elif strategy is ChunkStrategy.FULL:
        body_preview = "\n".join(chunk.body.splitlines()[:BODY_LINE_LIMIT])
        text = (
            f"{chunk.qualname}\n"
            f"{chunk.signature}\n"
            f"{chunk.docstring or ''}\n"
            f"{body_preview}"
        )
    else:
        raise ValueError(f"Unknown chunk strategy: {strategy}")
    return text[:MAX_EMBEDDING_CHARS]


def parse_strategy(value: str) -> ChunkStrategy:
    try:
        return ChunkStrategy(value)
    except ValueError as exc:
        valid = ", ".join(s.value for s in ChunkStrategy)
        raise ValueError(f"Unknown chunk strategy {value!r}. Expected one of: {valid}") from exc

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class CodeChunk:
    id: str
    repo_path: str
    file_path: str
    qualname: str
    start_line: int
    end_line: int
    signature: str
    docstring: str | None
    body: str
    embedding_text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CodeChunk:
        return cls(**data)


@dataclass
class SearchResult:
    chunk: CodeChunk
    score: float
    rank: int

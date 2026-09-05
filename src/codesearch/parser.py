from __future__ import annotations

import ast
import logging
import os
import tokenize
from pathlib import Path

from codesearch.models import CodeChunk

logger = logging.getLogger("codesearch")

SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "build",
    "dist",
    ".codesearch",
    "__pycache__",
}

SKIP_EXACT_FILES = {"setup.py"}

MIN_BODY_LINES = 3


def parse_repo(repo_path: str | Path) -> list[CodeChunk]:
    """Walk a Python repo and extract function/method chunks."""
    repo = Path(repo_path).resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"Repo path is not a directory: {repo}")

    chunks: list[CodeChunk] = []
    for path in _iter_python_files(repo):
        source = _read_source(path)
        if source is None:
            continue
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            logger.warning("Skipping %s: SyntaxError: %s", path, exc)
            continue
        rel = path.relative_to(repo).as_posix()
        chunks.extend(_extract_chunks(tree, source, repo, rel))
    _disambiguate_ids(chunks)
    return chunks


def _read_source(path: Path) -> str | None:
    try:
        with tokenize.open(path) as handle:
            return handle.read()
    except (SyntaxError, UnicodeDecodeError, tokenize.TokenError) as exc:
        logger.warning("Skipping %s: could not decode source: %s", path, exc)
        return None


def _iter_python_files(repo: Path) -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(repo):
        dirnames[:] = [
            name for name in dirnames if name not in SKIP_DIR_NAMES and not name.startswith(".")
        ]
        current = Path(dirpath)
        for name in filenames:
            if not name.endswith(".py"):
                continue
            if name in SKIP_EXACT_FILES:
                continue
            if _is_test_filename(name):
                continue
            files.append(current / name)
    files.sort()
    return files


def _is_test_filename(name: str) -> bool:
    return name.startswith("test_") or name.endswith("_test.py")


def _extract_chunks(
    tree: ast.AST,
    source: str,
    repo: Path,
    file_path: str,
) -> list[CodeChunk]:
    source_lines = source.splitlines()
    chunks: list[CodeChunk] = []

    def visit(node: ast.AST, scope: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, scope + [child.name])
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunk = _chunk_from_function(child, source_lines, repo, file_path, scope)
                if chunk is not None:
                    chunks.append(chunk)
                visit(child, scope + [child.name])
            else:
                visit(child, scope)

    visit(tree, [])
    return chunks


def _chunk_from_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    source_lines: list[str],
    repo: Path,
    file_path: str,
    scope: list[str],
) -> CodeChunk | None:
    body = _function_body(node, source_lines)
    if _nonempty_line_count(body) < MIN_BODY_LINES:
        return None

    qualname = ".".join([*scope, node.name])
    role = _decorator_role(node)
    if role:
        qualname = f"{qualname}@{role}"
    docstring = ast.get_docstring(node)
    start_line = node.lineno
    end_line = node.end_lineno if node.end_lineno is not None else node.lineno
    return CodeChunk(
        id=f"{file_path}::{qualname}",
        repo_path=str(repo),
        file_path=file_path,
        qualname=qualname,
        start_line=start_line,
        end_line=end_line,
        signature=_reconstruct_signature(node),
        docstring=docstring,
        body=body,
        embedding_text="",
    )


def _decorator_role(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    for decorator in node.decorator_list:
        name = _decorator_name(decorator)
        if name is None:
            continue
        simple = name.rsplit(".", 1)[-1]
        if simple in {"setter", "deleter", "overload"}:
            return simple
    return None


def _decorator_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _decorator_name(node.value)
        if parent is None:
            return node.attr
        return f"{parent}.{node.attr}"
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    return None


def _disambiguate_ids(chunks: list[CodeChunk]) -> None:
    seen: dict[str, list[CodeChunk]] = {}
    for chunk in chunks:
        seen.setdefault(chunk.id, []).append(chunk)
    for group in seen.values():
        if len(group) < 2:
            continue
        for chunk in group:
            suffix = f"#L{chunk.start_line}"
            chunk.qualname = f"{chunk.qualname}{suffix}"
            chunk.id = f"{chunk.file_path}::{chunk.qualname}"


def _function_body(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    source_lines: list[str],
) -> str:
    stmts = list(node.body)
    if not stmts:
        return ""
    first = stmts[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        stmts = stmts[1:]
    if not stmts:
        return ""
    start = stmts[0].lineno
    end = node.end_lineno if node.end_lineno is not None else stmts[-1].lineno
    return "\n".join(source_lines[start - 1 : end])


def _nonempty_line_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


def _reconstruct_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    args = ast.unparse(node.args)
    ret = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    return f"{prefix} {node.name}({args}){ret}"

from __future__ import annotations

import ast
import logging
import os
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
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning("Skipping %s: could not decode as UTF-8", path)
            continue
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            logger.warning("Skipping %s: SyntaxError: %s", path, exc)
            continue
        rel = path.relative_to(repo).as_posix()
        chunks.extend(_extract_chunks(tree, source, repo, rel))
    return chunks


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

    def visit(node: ast.AST, class_parts: list[str], func_parts: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, class_parts + [child.name], func_parts)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunk = _chunk_from_function(
                    child, source_lines, repo, file_path, class_parts, func_parts
                )
                if chunk is not None:
                    chunks.append(chunk)
                visit(child, class_parts, func_parts + [child.name])
            else:
                visit(child, class_parts, func_parts)

    visit(tree, [], [])
    return chunks


def _chunk_from_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    source_lines: list[str],
    repo: Path,
    file_path: str,
    class_parts: list[str],
    func_parts: list[str],
) -> CodeChunk | None:
    body = _function_body(node, source_lines)
    if _nonempty_line_count(body) < MIN_BODY_LINES:
        return None

    qualname = ".".join([*class_parts, *func_parts, node.name])
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

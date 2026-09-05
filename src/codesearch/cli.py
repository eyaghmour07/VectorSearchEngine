from __future__ import annotations

import json
import logging
import sys
from enum import Enum
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from codesearch.benchmark import (
    DEFAULT_GROUND_TRUTH,
    DEFAULT_RESULTS_PATH,
    GroundTruthError,
    run_benchmark,
)
from codesearch.chunker import apply_strategy, parse_strategy
from codesearch.embedder import MODEL_NAME, Embedder
from codesearch.index import DEFAULT_EF_SEARCH, create_index
from codesearch.models import SearchResult
from codesearch.parser import parse_repo
from codesearch.runtime import configure_measurement_threads
from codesearch.store import DEFAULT_STORE_DIR, MetadataMismatchError, load_index, save_index

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()
err_console = Console(stderr=True)


class IndexType(str, Enum):
    flat = "flat"
    hnsw = "hnsw"


class StrategyOpt(str, Enum):
    full = "full"
    sig = "sig"
    sig_doc = "sig-doc"


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def _require_python() -> None:
    if sys.version_info < (3, 11):
        err_console.print(
            "[red]codesearch requires Python 3.11+. "
            f"This interpreter is {sys.version.split()[0]}. "
            "Older parsers skip newer syntax and change the corpus.[/red]"
        )
        raise typer.Exit(code=1)


@app.command("index")
def index_cmd(
    repo: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True),
    index_type: IndexType = typer.Option(IndexType.flat, "--index-type"),
    strategy: StrategyOpt = typer.Option(StrategyOpt.full, "--strategy"),
    store_dir: Path = typer.Option(DEFAULT_STORE_DIR, "--store-dir"),
) -> None:
    """Index a Python repository into a persistent vector index."""
    _require_python()
    _configure_logging()
    configure_measurement_threads()
    repo_path = str(repo.resolve())
    chunk_strategy = parse_strategy(strategy.value)

    console.print(f"Parsing [bold]{repo_path}[/bold] ...")
    chunks = parse_repo(repo_path)
    apply_strategy(chunks, chunk_strategy)
    console.print(f"Extracted [bold]{len(chunks)}[/bold] chunks (strategy={chunk_strategy.value})")
    if not chunks:
        raise typer.Exit(code=1)

    embedder = Embedder(model_name=MODEL_NAME, cache_dir=store_dir / "embed_cache")
    vectors = embedder.encode([chunk.embedding_text for chunk in chunks])
    if vectors.shape[0] != len(chunks):
        raise RuntimeError(
            f"Embedding count {vectors.shape[0]} does not match chunk count {len(chunks)}"
        )

    vec_index = create_index(index_type.value, vectors.shape[1])
    vec_index.build(vectors)
    save_index(
        store_dir,
        vec_index,
        chunks,
        repo_path=repo_path,
        model_name=MODEL_NAME,
        chunk_strategy=chunk_strategy.value,
        index_type=index_type.value,
    )
    console.print(f"Wrote {index_type.value} index with {len(chunks)} chunks to {store_dir}/")


@app.command()
def search(
    query: str = typer.Argument(...),
    k: int = typer.Option(10, "-k", "--k", min=1),
    ef_search: int = typer.Option(DEFAULT_EF_SEARCH, "--ef-search", min=1),
    strategy: StrategyOpt = typer.Option(StrategyOpt.full, "--strategy"),
    store_dir: Path = typer.Option(DEFAULT_STORE_DIR, "--store-dir"),
) -> None:
    """Search the current index with a natural-language description of behavior."""
    _require_python()
    _configure_logging()
    configure_measurement_threads()
    try:
        stored = load_index(
            store_dir,
            model_name=MODEL_NAME,
            chunk_strategy=strategy.value,
            ef_search=ef_search,
        )
    except FileNotFoundError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    except (MetadataMismatchError, ValueError, TypeError, json.JSONDecodeError) as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    embedder = Embedder(model_name=MODEL_NAME, cache_dir=store_dir / "embed_cache")
    query_vec = embedder.encode_query(query)
    hits = stored.index.search(query_vec, k, ef_search=ef_search)
    results = [
        SearchResult(chunk=stored.chunks[idx], score=score, rank=rank)
        for rank, (idx, score) in enumerate(hits, start=1)
    ]
    _print_results(results)


@app.command()
def benchmark(
    ground_truth: Path = typer.Option(
        DEFAULT_GROUND_TRUTH,
        "--ground-truth",
        exists=False,
        dir_okay=False,
    ),
    store_dir: Path = typer.Option(DEFAULT_STORE_DIR, "--store-dir"),
    output: Path = typer.Option(DEFAULT_RESULTS_PATH, "--output"),
) -> None:
    """Compare exact vs HNSW search on recall@k and latency."""
    _require_python()
    _configure_logging()
    try:
        payload = run_benchmark(
            ground_truth_path=ground_truth,
            store_dir=store_dir,
            output_path=output,
        )
    except FileNotFoundError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    except (ValueError, GroundTruthError, TypeError) as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    _print_benchmark(payload)


def _print_results(results: list[SearchResult]) -> None:
    if not results:
        console.print("No results.")
        return
    for result in results:
        snippet = _first_doc_line(result.chunk.docstring)
        console.print(
            f"[bold]{result.rank}. {result.chunk.qualname:<40}[/bold]  "
            f"score {result.score:.3f}"
        )
        console.print(
            f"   {result.chunk.file_path}:{result.chunk.start_line}-{result.chunk.end_line}"
        )
        if snippet:
            console.print(f"   {snippet}")
        console.print()


def _first_doc_line(docstring: str | None) -> str:
    if not docstring:
        return ""
    for line in docstring.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _print_benchmark(payload: dict) -> None:
    info = Table(title="Index stats")
    info.add_column("index")
    info.add_column("build time (s)")
    info.add_column("on-disk size")
    info.add_column("vectors")
    for row in payload["indexes"]:
        info.add_row(
            row["index_type"],
            f"{row['build_seconds']:.3f}",
            _format_bytes(row["disk_bytes"]),
            str(row["ntotal"]),
        )
    console.print(info)

    table = Table(title="Publishable recall and latency")
    table.add_column("index")
    table.add_column("efSearch")
    table.add_column("recall@1")
    table.add_column("recall@5")
    table.add_column("recall@10")
    table.add_column("p50 ms")
    table.add_column("p95 ms")
    table.add_column("qps")
    for row in payload["publishable_results"]:
        table.add_row(
            row["index_type"],
            "-" if row["ef_search"] is None else str(row["ef_search"]),
            f"{row['recall@1']:.3f}",
            f"{row['recall@5']:.3f}",
            f"{row['recall@10']:.3f}",
            f"{row['p50_ms']:.3f}",
            f"{row['p95_ms']:.3f}",
            f"{row['qps']:.1f}",
        )
    console.print(table)
    for warning in payload.get("recall_warnings") or []:
        err_console.print(f"[yellow]{warning}[/yellow]")
    console.print(f"Wrote {payload['output_path']}")


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


if __name__ == "__main__":
    app()

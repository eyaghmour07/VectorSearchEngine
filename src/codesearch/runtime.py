from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from codesearch import __version__

FAISS_THREADS = 1
_LOADED_OMP: str | None = None

# Do not set KMP_DUPLICATE_LIB_OK. That workaround can produce silent wrong
# results. Preload one OpenMP runtime instead so faiss and torch share it.
_OMP_FILENAMES = (
    "libomp.dylib",
    "libiomp5.dylib",
    "libomp.so",
    "libomp.so.5",
    "libiomp5.so",
    "libgomp.so.1",
)


def _discover_omp_paths() -> list[Path]:
    paths: list[Path] = []
    # Torch's copy must win. Preloading faiss's libomp first makes PyTorch
    # initialize a second runtime and abort on macOS.
    for package in ("torch", "sklearn", "faiss"):
        spec = importlib.util.find_spec(package)
        if spec is None or not spec.origin:
            continue
        root = Path(spec.origin).resolve().parent
        for name in _OMP_FILENAMES:
            paths.append(root / name)
            paths.append(root / ".dylibs" / name)
            paths.append(root / "lib" / name)
    extra_roots = [
        Path("/opt/homebrew/opt/libomp/lib"),
        Path("/usr/local/opt/libomp/lib"),
        Path("/opt/homebrew/lib"),
        Path("/usr/lib"),
    ]
    for root in extra_roots:
        for name in _OMP_FILENAMES:
            paths.append(root / name)
    return paths


def preload_single_openmp() -> str | None:
    """Load one OpenMP dylib globally before faiss or torch initialize their copies."""
    global _LOADED_OMP
    if _LOADED_OMP:
        return _LOADED_OMP
    for path in _discover_omp_paths():
        if not path.is_file():
            continue
        try:
            ctypes.CDLL(str(path), mode=os.RTLD_GLOBAL)
        except OSError:
            continue
        _LOADED_OMP = str(path)
        return _LOADED_OMP
    return None


def configure_measurement_threads() -> int:
    """Pin BLAS/OpenMP to one thread so latency numbers are comparable."""
    preload_single_openmp()
    os.environ.setdefault("OMP_NUM_THREADS", str(FAISS_THREADS))
    os.environ.setdefault("MKL_NUM_THREADS", str(FAISS_THREADS))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(FAISS_THREADS))
    try:
        import faiss

        faiss.omp_set_num_threads(FAISS_THREADS)
    except ImportError:
        pass
    return FAISS_THREADS


def git_head(repo_path: str | Path) -> str | None:
    root = Path(repo_path).resolve()
    for candidate in (root, *root.parents):
        if (candidate / ".git").exists():
            try:
                result = subprocess.run(
                    ["git", "-C", str(candidate), "rev-parse", "HEAD"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except (OSError, subprocess.CalledProcessError):
                return None
            return result.stdout.strip() or None
    return None


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_run_metadata(
    *,
    repo_path: str,
    model_name: str,
    model_revision: str | None = None,
) -> dict:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "codesearch_version": __version__,
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "faiss_threads": FAISS_THREADS,
        "openmp_library": _LOADED_OMP,
        "numpy_version": np.__version__,
        "faiss_version": package_version("faiss-cpu") or package_version("faiss"),
        "sentence_transformers_version": package_version("sentence-transformers"),
        "torch_version": package_version("torch"),
        "model_name": model_name,
        "model_revision": model_revision,
        "corpus_path": repo_path,
        "corpus_git_sha": git_head(repo_path),
    }

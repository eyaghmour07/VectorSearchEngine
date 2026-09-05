from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384
BATCH_SIZE = 64
DEFAULT_CACHE_DIR = Path(".codesearch") / "embed_cache"


class Embedder:
    """Encode texts with MiniLM, L2-normalize, and cache vectors by text hash."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._model = None
        self._dim = EMBED_DIM if model_name == MODEL_NAME else None
        self._model_revision: str | None = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            dim = self._model.get_sentence_embedding_dimension()
            if dim is not None:
                self._dim = int(dim)
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._load_model()
        return int(self._dim or EMBED_DIM)

    @property
    def model_revision(self) -> str | None:
        if self._model_revision:
            return self._model_revision
        if self._model is None:
            return None
        revision = getattr(
            getattr(self._model, "model_card_data", None), "base_model_revision", None
        ) or getattr(self._model, "_model_revision", None)
        self._model_revision = revision
        return revision

    def encode(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        vectors = [None] * len(texts)
        missing_indices: list[int] = []
        for i, text in enumerate(texts):
            cached = self._read_cache(text)
            if cached is None:
                missing_indices.append(i)
            else:
                vectors[i] = cached

        if missing_indices:
            missing_texts = [texts[i] for i in missing_indices]
            encoded = self._embed_raw(missing_texts, show_progress=show_progress)
            encoded = l2_normalize(encoded)
            for offset, idx in enumerate(missing_indices):
                vec = encoded[offset]
                vectors[idx] = vec
                self._write_cache(texts[idx], vec)

        stacked = np.stack(vectors).astype(np.float32, copy=False)
        return l2_normalize(stacked)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text], show_progress=False)[0]

    def _embed_raw(self, texts: list[str], show_progress: bool) -> np.ndarray:
        if os.environ.get("CODESEARCH_EMBED_WORKER") == "1":
            return self._embed_raw_local(texts, show_progress)
        return self._embed_raw_in_worker(texts)

    def _embed_raw_local(self, texts: list[str], show_progress: bool) -> np.ndarray:
        model = self._load_model()
        encoded = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return np.asarray(encoded, dtype=np.float32)

    def _embed_raw_in_worker(self, texts: list[str]) -> np.ndarray:
        with tempfile.TemporaryDirectory(prefix="codesearch-embed-") as tmp:
            tmp_path = Path(tmp)
            texts_path = tmp_path / "texts.json"
            out_path = tmp_path / "vectors.npy"
            texts_path.write_text(json.dumps(texts), encoding="utf-8")
            env = os.environ.copy()
            env["CODESEARCH_EMBED_WORKER"] = "1"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "codesearch.embed_worker",
                    self.model_name,
                    str(texts_path),
                    str(out_path),
                ],
                check=False,
                env=env,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "").strip()
                raise RuntimeError(
                    f"Embedding worker exited {result.returncode}"
                    + (f": {detail}" if detail else "")
                )
            for line in (result.stdout or "").splitlines():
                if line.startswith("MODEL_REVISION="):
                    revision = line.split("=", 1)[1].strip()
                    if revision:
                        self._model_revision = revision
            vectors = np.load(out_path)
        return np.asarray(vectors, dtype=np.float32)

    def _cache_path(self, text: str) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256(f"{self.model_name}\n{text}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.npy"

    def _read_cache(self, text: str) -> np.ndarray | None:
        path = self._cache_path(text)
        if path is None or not path.exists():
            return None
        vec = np.load(path)
        expected = (self.dim,)
        if vec.shape != expected:
            raise ValueError(
                f"Cached embedding at {path} has shape {vec.shape}, expected {expected}"
            )
        return np.asarray(vec, dtype=np.float32)

    def _write_cache(self, text: str, vector: np.ndarray) -> None:
        path = self._cache_path(text)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.stem}.{os.getpid()}.{uuid4().hex}.tmp.npy")
        np.save(tmp, vector.astype(np.float32, copy=False))
        tmp.replace(path)


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    out = (vectors / norms).astype(np.float32, copy=False)
    if squeeze:
        return out[0]
    return out

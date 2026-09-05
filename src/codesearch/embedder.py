from __future__ import annotations

import hashlib
from pathlib import Path

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

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        if not texts:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)

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
        model = self._load_model()
        encoded = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return np.asarray(encoded, dtype=np.float32)

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
        if vec.shape != (EMBED_DIM,):
            raise ValueError(
                f"Cached embedding at {path} has shape {vec.shape}, expected ({EMBED_DIM},)"
            )
        return np.asarray(vec, dtype=np.float32)

    def _write_cache(self, text: str, vector: np.ndarray) -> None:
        path = self._cache_path(text)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npy")
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

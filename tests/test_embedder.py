from __future__ import annotations

from pathlib import Path

import numpy as np

from codesearch.embedder import Embedder, l2_normalize


def test_cache_writes_use_unique_temp_files(tmp_path: Path, monkeypatch) -> None:
    embedder = Embedder(cache_dir=tmp_path)
    seen: list[Path] = []
    original_save = np.save

    def tracking_save(path, arr, *args, **kwargs):
        seen.append(Path(path))
        return original_save(path, arr, *args, **kwargs)

    monkeypatch.setattr(np, "save", tracking_save)
    vector = np.ones(384, dtype=np.float32)
    embedder._write_cache("alpha", vector)
    embedder._write_cache("beta", vector)

    assert len(seen) == 2
    assert seen[0] != seen[1]
    assert all(path.name.startswith(".") and ".tmp.npy" in path.name for path in seen)
    assert (tmp_path / f"{embedder._cache_path('alpha').stem}.npy").exists()
    assert np.allclose(np.load(embedder._cache_path("alpha")), vector)


def test_embed_raw_uses_worker_outside_worker_process(tmp_path: Path, monkeypatch) -> None:
    embedder = Embedder(cache_dir=tmp_path)
    seen: dict = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        out = Path(cmd[-1])
        np.save(out, np.ones((1, 384), dtype=np.float32))
        return type("R", (), {"returncode": 0, "stdout": "MODEL_REVISION=abc\n", "stderr": ""})()

    monkeypatch.setattr("codesearch.embedder.subprocess.run", fake_run)
    vectors = embedder._embed_raw(["hello"], show_progress=False)
    assert "codesearch.embed_worker" in seen["cmd"]
    assert vectors.shape == (1, 384)
    assert embedder.model_revision == "abc"


def test_openmp_preload_finds_a_library() -> None:
    from codesearch.runtime import _LOADED_OMP, configure_measurement_threads, preload_single_openmp

    path = preload_single_openmp()
    assert configure_measurement_threads() == 1
    assert path is None or Path(path).exists()
    assert _LOADED_OMP == path


def test_l2_normalize_unit_length() -> None:
    out = l2_normalize(np.array([[3.0, 4.0]], dtype=np.float32))
    assert out.shape == (1, 2)
    assert np.isclose(np.linalg.norm(out[0]), 1.0)

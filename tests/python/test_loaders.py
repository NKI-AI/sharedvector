import json
from pathlib import Path

import numpy as np
import pytest

from sharedvector.loaders import NumpyLoader, ZarrLoader, cast_volume, resolve_numpy_dtype


def _volume(value: int = 0) -> np.ndarray:
    return np.full((4, 3, 2), value, dtype=np.int16)


def _make_zarr_db(root: Path, volumes: dict[str, np.ndarray], manifest: dict | None) -> Path | None:
    import zarr

    root.mkdir()
    for name, volume in volumes.items():
        zarr.create_array(store=str(root / name), data=volume, zarr_format=3)
    if manifest is None:
        return None
    manifest_path = root / "spacing.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


def test_resolve_numpy_dtype_rejects_unsupported() -> None:
    with pytest.raises(ValueError):
        resolve_numpy_dtype("float64")


def test_cast_volume_returns_same_object_when_dtype_matches() -> None:
    volume = _volume(3)

    assert cast_volume(volume, np.dtype(np.int16)) is volume


def test_cast_volume_clips_integer_targets() -> None:
    wide = np.array([40000, -40000, 12], dtype=np.int32)

    np.testing.assert_array_equal(cast_volume(wide, np.dtype(np.int16)), [32767, -32768, 12])


def test_numpy_loader_sorted_discovery_and_cast(tmp_path: Path) -> None:
    np.save(tmp_path / "b.npy", np.full((2, 2), 2.0))
    np.save(tmp_path / "a.npy", np.full((2, 2), 1.0))
    loader = NumpyLoader(tmp_path)

    chunks = loader(0)

    assert len(loader) == 2
    assert len(chunks) == 1
    assert chunks[0].data.dtype == np.float32
    assert chunks[0].data.flags.c_contiguous
    assert chunks[0].meta.shape == (0,)
    np.testing.assert_array_equal(chunks[0].data, 1.0)


def test_numpy_loader_raises_without_matches(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        NumpyLoader(tmp_path)
    with pytest.raises(FileNotFoundError):
        NumpyLoader(tmp_path / "missing")


@pytest.mark.zarr
def test_zarr_loader_returns_int16_chunks_with_reversed_spacing(tmp_path: Path) -> None:
    root = tmp_path / "db.zarr"
    manifest = _make_zarr_db(root, {"a.Vol": _volume()}, {"a.Vol": [0.7, 0.8, 2.5]})
    loader = ZarrLoader(root, slices_per_chunk=2, spacing_file=manifest)

    chunks = loader(0)

    assert len(chunks) == 2
    assert all(chunk.data.dtype == np.int16 for chunk in chunks)
    assert all(chunk.data.shape == (2, 3, 2) for chunk in chunks)
    np.testing.assert_array_equal(chunks[0].meta, [2.5, 0.8, 0.7])


@pytest.mark.zarr
def test_zarr_loader_clips_wide_values_into_int16(tmp_path: Path) -> None:
    root = tmp_path / "db.zarr"
    _make_zarr_db(root, {"a": np.array([[[40000, -40000, 12]]], dtype=np.int32)}, None)
    loader = ZarrLoader(root, slices_per_chunk=0)

    chunk = loader(0)[0]

    np.testing.assert_array_equal(chunk.data, [[[32767, -32768, 12]]])


@pytest.mark.zarr
def test_zarr_loader_float32_target_does_not_clip(tmp_path: Path) -> None:
    root = tmp_path / "db.zarr"
    _make_zarr_db(root, {"a": np.array([[[40000, -40000]]], dtype=np.int32)}, None)
    loader = ZarrLoader(root, slices_per_chunk=0, dtype="float32")

    chunk = loader(0)[0]

    assert chunk.data.dtype == np.float32
    np.testing.assert_array_equal(chunk.data, [[[40000.0, -40000.0]]])


@pytest.mark.zarr
def test_zarr_loader_counts_only_array_directories(tmp_path: Path) -> None:
    root = tmp_path / "db.zarr"
    _make_zarr_db(root, {"a": _volume(), "b": _volume()}, {"a": [1, 1, 1], "b": [1, 1, 1]})

    assert len(ZarrLoader(root, slices_per_chunk=1)) == 2


@pytest.mark.zarr
def test_zarr_loader_missing_volume_in_manifest_raises(tmp_path: Path) -> None:
    root = tmp_path / "db.zarr"
    manifest = _make_zarr_db(root, {"a": _volume(), "b": _volume()}, {"a": [1, 1, 1]})

    with pytest.raises(ValueError, match="b"):
        ZarrLoader(root, slices_per_chunk=1, spacing_file=manifest)


@pytest.mark.zarr
def test_zarr_loader_without_manifest_publishes_no_meta(tmp_path: Path) -> None:
    root = tmp_path / "db.zarr"
    _make_zarr_db(root, {"a": _volume()}, None)

    chunks = ZarrLoader(root, slices_per_chunk=2)(0)

    assert all(chunk.meta.shape == (0,) for chunk in chunks)


@pytest.mark.zarr
def test_zarr_loader_too_small_volume_yields_no_chunks(tmp_path: Path) -> None:
    root = tmp_path / "db.zarr"
    _make_zarr_db(root, {"a": _volume()}, None)

    assert ZarrLoader(root, slices_per_chunk=5)(0) == []


@pytest.mark.zarr
def test_zarr_loader_missing_manifest_path_raises(tmp_path: Path) -> None:
    root = tmp_path / "db.zarr"
    _make_zarr_db(root, {"a": _volume()}, None)

    with pytest.raises(FileNotFoundError):
        ZarrLoader(root, slices_per_chunk=1, spacing_file=root / "spacing.json")


@pytest.mark.zarr
def test_zarr_loader_flip_is_seedable(tmp_path: Path) -> None:
    root = tmp_path / "db.zarr"
    volume = np.arange(24, dtype=np.int16).reshape(4, 3, 2)
    _make_zarr_db(root, {"a": volume}, None)

    first = [ZarrLoader(root, slices_per_chunk=2, seed=0)(0)[0].data for _ in range(3)]

    assert all(np.array_equal(first[0], other) for other in first[1:])

import json
from pathlib import Path

import pytest

from helpers import TaggedLoader
from sharedvector.cli import EXIT_BUSY, instantiate_loader, producer_main, resolve_source_size, spacing_manifest_main
from sharedvector.coordination import OwnerLock, PoolPaths


def test_instantiate_loader_from_spec_and_json() -> None:
    loader = instantiate_loader("helpers:TaggedLoader", json.dumps({"source_size": 3, "element_count": 2}))

    assert isinstance(loader, TaggedLoader)
    assert len(loader) == 3
    assert loader.element_count == 2


def test_instantiate_loader_requires_class_name() -> None:
    with pytest.raises(ValueError):
        instantiate_loader("helpers", None)


def test_resolve_source_size_prefers_override() -> None:
    assert resolve_source_size(TaggedLoader(3), 7) == 7
    assert resolve_source_size(TaggedLoader(3), None) == 3
    with pytest.raises(ValueError):
        resolve_source_size(lambda source_index: [], None)  # ty: ignore[invalid-argument-type]


def test_producer_main_exits_busy_when_pool_is_owned(shm_name: str, lock_dir: Path) -> None:
    lock = OwnerLock.try_acquire(PoolPaths.for_name(shm_name, lock_dir))
    assert lock is not None
    try:
        code = producer_main(
            [
                "--name", shm_name, "--slot-count", "4", "--chunk-size-mb", "0.01",
                "--loader-class", "helpers:TaggedLoader", "--loader-args", '{"source_size": 4}',
                "--lock-dir", str(lock_dir),
            ]
        )  # fmt: skip
    finally:
        lock.release()

    assert code == EXIT_BUSY


@pytest.mark.zarr
def test_spacing_manifest_main_writes_file(tmp_path: Path, capsys) -> None:
    import numpy as np
    import zarr

    root = tmp_path / "db.zarr"
    root.mkdir()
    zarr.create_array(store=str(root / "a"), data=np.zeros((2, 2, 2), dtype=np.int16), zarr_format=3).attrs[
        "spacing"
    ] = [1, 2, 3]

    code = spacing_manifest_main([str(root), "--output", str(tmp_path / "out.json")])

    assert code == 0
    assert json.loads((tmp_path / "out.json").read_text()) == {"a": [1.0, 2.0, 3.0]}
    assert "wrote" in capsys.readouterr().out

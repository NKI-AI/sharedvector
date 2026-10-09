"""Shared fixtures: optional-dependency skipping and segment cleanup."""

import contextlib
import importlib.util
from pathlib import Path

import pytest

from helpers import unique_name
from sharedvector.coordination import PoolPaths
from sharedvector.vector import remove_shared_memory

_OPTIONAL = {"torch": "torch", "zarr": "zarr"}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    missing = {marker for marker, module in _OPTIONAL.items() if importlib.util.find_spec(module) is None}
    for item in items:
        for marker in missing:
            if marker in item.keywords:
                item.add_marker(pytest.mark.skip(reason=f"{marker} is not installed"))


@pytest.fixture
def lock_dir(tmp_path: Path) -> Path:
    return tmp_path / "locks"


@pytest.fixture
def shm_name(request: pytest.FixtureRequest, lock_dir: Path):
    """Unique segment name; removes the segment and coordination files afterwards."""
    name = unique_name(request.node.name[:20].replace("[", "_").replace("]", "_"))
    remove_shared_memory(name)
    yield name
    remove_shared_memory(name)
    paths = PoolPaths.for_name(name, lock_dir)
    for path in (paths.marker, paths.lock):
        with contextlib.suppress(FileNotFoundError):
            path.unlink()

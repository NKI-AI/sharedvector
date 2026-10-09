import json
from pathlib import Path

import numpy as np
import pytest

from sharedvector.spacing import collect_spacings, write_spacing_manifest

pytestmark = pytest.mark.zarr


def _make_db(root: Path, spacings: dict[str, list[float] | None]) -> None:
    import zarr

    root.mkdir()
    for name, spacing in spacings.items():
        array = zarr.create_array(store=str(root / name), data=np.zeros((2, 2, 2), dtype=np.int16), zarr_format=3)
        if spacing is not None:
            array.attrs["spacing"] = spacing


def test_collect_spacings_reads_every_array(tmp_path: Path) -> None:
    root = tmp_path / "db.zarr"
    _make_db(root, {"b": [0.7, 0.8, 2.5], "a.Vol": [1, 1, 1]})

    assert collect_spacings(root) == {"a.Vol": [1.0, 1.0, 1.0], "b": [0.7, 0.8, 2.5]}


def test_collect_spacings_strict_raises_on_missing_attribute(tmp_path: Path) -> None:
    root = tmp_path / "db.zarr"
    _make_db(root, {"a": [1, 1, 1], "b": None})

    with pytest.raises(ValueError, match="b"):
        collect_spacings(root)


def test_collect_spacings_lenient_skips_missing(tmp_path: Path) -> None:
    root = tmp_path / "db.zarr"
    _make_db(root, {"a": [1, 1, 1], "b": None})

    assert collect_spacings(root, strict=False) == {"a": [1.0, 1.0, 1.0]}


def test_collect_spacings_requires_arrays(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        collect_spacings(tmp_path)


def test_write_spacing_manifest_defaults_to_root(tmp_path: Path) -> None:
    root = tmp_path / "db.zarr"
    _make_db(root, {"a": [1, 2, 3]})

    output = write_spacing_manifest(root)

    assert output == root / "spacing.json"
    assert json.loads(output.read_text()) == {"a": [1.0, 2.0, 3.0]}


def test_write_spacing_manifest_honours_output(tmp_path: Path) -> None:
    root = tmp_path / "db.zarr"
    _make_db(root, {"a": [1, 2, 3]})

    output = write_spacing_manifest(root, tmp_path / "elsewhere.json")

    assert output == tmp_path / "elsewhere.json" and output.is_file()

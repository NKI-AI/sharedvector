# Copyright 2026 AI for Oncology. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Spacing manifests for zarr databases consumed by :class:`~sharedvector.loaders.ZarrLoader`."""

import json
import logging
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _progress(items: Iterable[T], description: str) -> Iterable[T]:
    """Wrap ``items`` in a tqdm bar when tqdm is installed, else return them unchanged."""
    try:
        from tqdm import tqdm
    except ImportError:
        return items
    return tqdm(items, desc=description, unit="array")


def _read_spacing(path: Path, attr_key: str) -> Any:
    """Return the ``attr_key`` attribute from ``path/zarr.json`` or ``None``."""
    try:
        with open(path / "zarr.json", "rb") as handle:
            metadata = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    attributes = metadata.get("attributes")
    return attributes.get(attr_key) if isinstance(attributes, dict) else None


def collect_spacings(root: Path | str, attr_key: str = "spacing", strict: bool = True) -> dict[str, list[float]]:
    """Read one spacing triple per zarr array directory under ``root``.

    Args:
        root: Zarr database root holding one array directory per volume.
        attr_key: Array attribute holding the ``(x, y, z)`` spacing.
        strict: Raise on a missing or malformed attribute instead of skipping the volume with a warning.

    Returns:
        Mapping from array directory name to its ``[x, y, z]`` spacing.

    Raises:
        FileNotFoundError: If ``root`` is not a directory or holds no arrays.
        ValueError: In strict mode, if an array lacks a valid spacing attribute.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"Not a directory: {root_path}")
    with os.scandir(root_path) as entries:
        names = [entry.name for entry in entries if entry.is_dir()]
    if not names:
        raise FileNotFoundError(f"No arrays found under {root_path}")

    spacings: dict[str, list[float]] = {}
    for name in _progress(names, "collecting spacings"):
        spacing = _read_spacing(root_path / name, attr_key)
        if not isinstance(spacing, list | tuple) or len(spacing) != 3:
            message = f"{name}: attribute {attr_key!r} missing or not a triple"
            if strict:
                raise ValueError(message)
            logger.warning("skipping %s", message)
            continue
        spacings[name] = [float(value) for value in spacing]
    return spacings


def write_spacing_manifest(
    root: Path | str,
    output: Path | str | None = None,
    attr_key: str = "spacing",
    strict: bool = True,
) -> Path:
    """Write the ``{name: [x, y, z]}`` manifest :class:`~sharedvector.loaders.ZarrLoader` consumes.

    Args:
        root: Zarr database root.
        output: Manifest path; defaults to ``root / "spacing.json"``.
        attr_key: Array attribute holding the spacing.
        strict: Forwarded to :func:`collect_spacings`.

    Returns:
        The written manifest path.
    """
    output_path = Path(root) / "spacing.json" if output is None else Path(output)
    spacings = collect_spacings(root, attr_key, strict)
    output_path.write_text(json.dumps(spacings, indent=1))
    return output_path


__all__ = ["collect_spacings", "write_spacing_manifest"]

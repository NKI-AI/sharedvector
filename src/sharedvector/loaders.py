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
"""Loaders map a source index to one or more chunks destined for the shared pool."""

import json
import random
from pathlib import Path
from typing import NamedTuple, Protocol

import numpy as np

_SUPPORTED_DTYPES = (np.dtype(np.float32), np.dtype(np.int16))


class Chunk(NamedTuple):
    """One contiguous array destined for a slot, with its per-slot metadata.

    Attributes:
        data: C-contiguous array in the segment dtype.
        meta: 1-D float64 metadata of at most eight values; empty when absent.
    """

    data: np.ndarray
    meta: np.ndarray


def no_meta() -> np.ndarray:
    """Return empty metadata for chunks without any."""
    return np.empty(0, dtype=np.float64)


def resolve_numpy_dtype(dtype: str) -> np.dtype:
    """Parse a segment dtype name, restricted to what the shared vector stores.

    Args:
        dtype: ``"float32"`` or ``"int16"``.

    Returns:
        The numpy dtype.

    Raises:
        ValueError: If the dtype is not supported by the shared vector.
    """
    resolved = np.dtype(dtype)
    if resolved not in _SUPPORTED_DTYPES:
        raise ValueError(f"dtype must be float32 or int16, got {dtype!r}")
    return resolved


def cast_volume(volume: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Cast to ``dtype``, clipping integer targets to their representable range.

    Args:
        volume: Source array.
        dtype: Target dtype.

    Returns:
        ``volume`` itself when already in ``dtype``, else a cast copy.
    """
    if volume.dtype == dtype:
        return volume
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(volume, info.min, info.max).astype(dtype)
    return volume.astype(dtype)


class Loader(Protocol):
    """Strategy callable mapping a source index to one or more chunks.

    A single source (typically a file) may yield multiple chunks, e.g. when a volume is split along
    axis 0. Returning ``[]`` signals that the source produced no usable chunks; callers skip it with
    a warning. Loaders are pickled into producer processes, so they must be picklable.
    """

    def __call__(self, source_index: int) -> list[Chunk]:
        """Return the chunks for ``source_index``."""
        ...

    def __len__(self) -> int:
        """Return the number of source indices."""
        ...


class NumpyLoader:
    """Reference loader that scans a directory for ``.npy`` files.

    Files are discovered via ``Path.glob(pattern)`` at construction time and sorted for deterministic
    source indexing across processes. Each file becomes a single chunk without metadata.
    """

    def __init__(self, directory: Path | str, pattern: str = "*.npy", dtype: str = "float32") -> None:
        """Discover ``.npy`` files under ``directory``.

        Args:
            directory: Folder to scan.
            pattern: Glob pattern relative to ``directory`` (use ``**/*.npy`` for recursive discovery).
            dtype: Segment dtype the arrays are cast to.

        Raises:
            FileNotFoundError: If ``directory`` does not exist or contains no matching files.
        """
        root = Path(directory)
        if not root.is_dir():
            raise FileNotFoundError(f"Not a directory: {root}")
        self._paths = sorted(root.glob(pattern))
        if not self._paths:
            raise FileNotFoundError(f"No files matching {pattern!r} under {root}")
        self._dtype = resolve_numpy_dtype(dtype)

    def __len__(self) -> int:
        return len(self._paths)

    def __call__(self, source_index: int) -> list[Chunk]:
        """Load one ``.npy`` file as a single chunk."""
        arr = np.load(self._paths[source_index])
        return [Chunk(np.ascontiguousarray(cast_volume(arr, self._dtype)), no_meta())]


class ZarrLoader:
    """Loader splitting zarr v3 volumes into slabs, optionally tagged with spacing.

    Spacing comes from one manifest per database, ``{"<array name>": [x, y, z]}``, parsed once at
    construction and shipped reversed as ``(z, y, x)`` so index ``i`` of the metadata matches axis
    ``i`` of the array. Without a manifest the chunks carry no metadata.

    Requires the ``zarr`` extra (``pip install sharedvector[zarr]``).
    """

    def __init__(
        self,
        directory: Path | str,
        slices_per_chunk: int,
        spacing_file: Path | str | None = None,
        dtype: str = "int16",
        seed: int | None = None,
    ) -> None:
        """Discover zarr arrays under ``directory`` and load the spacing manifest.

        Args:
            directory: Zarr database root holding one array directory per volume.
            slices_per_chunk: Slices per emitted chunk; ``0`` emits whole volumes.
            spacing_file: Optional spacing manifest; ``None`` publishes no spacing.
            dtype: Segment dtype; integer targets are clipped before casting.
            seed: Seed for the random axis flip applied before slabbing; ``None`` draws from the OS.

        Raises:
            FileNotFoundError: If ``directory`` has no arrays or ``spacing_file`` does not exist.
            ValueError: If a volume has no manifest entry or an entry is not a triple.
        """
        root = Path(directory)
        if not root.is_dir():
            raise FileNotFoundError(f"Not a directory: {root}")
        self._paths = sorted(f for f in root.glob("*") if f.is_dir())
        if not self._paths:
            raise FileNotFoundError(f"No arrays found under {root}")

        self._slices_per_chunk = slices_per_chunk
        self._dtype = resolve_numpy_dtype(dtype)
        self._spacings = None if spacing_file is None else self._load_spacings(Path(spacing_file))
        self._rng = random.Random(seed)

    def _load_spacings(self, manifest_path: Path) -> np.ndarray:
        """Read ``(len(paths), 3)`` float64 spacings in ``(z, y, x)`` order."""
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Spacing manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        missing = [path.name for path in self._paths if path.name not in manifest]
        if missing:
            raise ValueError(f"{len(missing)} volumes missing from {manifest_path}, e.g. {missing[:5]}")
        entries = [manifest[path.name] for path in self._paths]
        if any(len(entry) != 3 for entry in entries):
            raise ValueError(f"Spacing entries in {manifest_path} must be (x, y, z)")
        spacings = np.asarray(entries, dtype=np.float64)
        return np.ascontiguousarray(spacings[:, ::-1])

    def __len__(self) -> int:
        return len(self._paths)

    def __call__(self, source_index: int) -> list[Chunk]:
        """Load one volume and split it into slabs of ``slices_per_chunk``."""
        try:
            import zarr
        except ImportError as exc:
            raise ImportError("ZarrLoader requires the zarr extra: pip install sharedvector[zarr]") from exc

        sample_path = self._paths[source_index]
        volume: np.ndarray = zarr.open_array(sample_path, zarr_format=3, mode="r")[:]  # ty: ignore[invalid-assignment]
        volume = cast_volume(volume, self._dtype)
        meta = no_meta() if self._spacings is None else self._spacings[source_index]

        if self._slices_per_chunk == 0:
            return [Chunk(np.ascontiguousarray(volume), meta)]

        if self._rng.random() >= 0.5:
            volume = volume[::-1]

        num_full_chunks = len(volume) // self._slices_per_chunk
        if num_full_chunks == 0:
            return []

        volume = volume[: num_full_chunks * self._slices_per_chunk]
        return [Chunk(np.ascontiguousarray(chunk), meta) for chunk in np.array_split(volume, num_full_chunks)]


__all__ = ["Chunk", "Loader", "NumpyLoader", "ZarrLoader", "cast_volume", "no_meta", "resolve_numpy_dtype"]
